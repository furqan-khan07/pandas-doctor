"""Find the statements `pandas-doctor run` can check, and how to build rewrites for them.

A *site* is one statement in the user's code (an iterrows loop, a row-wise apply, a
concat loop) that the runtime will wrap: copy its inputs before it runs, let it run for
real, then run candidate rewrites on the copies and compare.

Candidates are built only after the original ran, because the best rewrite depends on the
data: which dtype iterrows gave each row, which dtype the result column ended up with.
Each site offers a short list from plainest to most faithful; the runtime keeps the
first one that checks out.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider

from pandas_doctor.findings import Fix
from pandas_doctor.static.analysis import (
    body_statements,
    code,
    names_in,
    small_statements,
    target_names,
    walk,
)
from pandas_doctor.static.rules.base import FileContext
from pandas_doctor.static.rules.pdd001_iterrows import LoopShape, _p, assigned_names, iterrows_frame
from pandas_doctor.static.rules.pdd002_apply_rowwise import row_function, rowwise_apply
from pandas_doctor.static.rules.pdd003_concat_in_loop import ConcatInLoop
from pandas_doctor.static.translate import Translation, Untranslatable, translate
from pandas_doctor.verify.purity import is_pure

Builder = Callable[[Mapping[str, Any], Mapping[str, Any]], list["Candidate"]]


@dataclass(frozen=True)
class Candidate:
    code: str
    label: str
    uses_numpy: bool


@dataclass
class ConcatInfo:
    target: str
    callee: str
    kwargs: str
    assign_line: int
    assign_col: int


@dataclass
class Site:
    id: int
    rule: str
    message: str
    line: int
    end_line: int
    col: int
    node: cst.BaseStatement  # in `SiteMap.module`
    source: str
    inputs: list[str]
    outputs: list[str]
    loop_vars: list[str] = field(default_factory=list)
    build: Builder | None = None
    repeatable: bool = False
    concat: ConcatInfo | None = None
    fix: Fix | None = None


@dataclass
class SiteMap:
    module: cst.Module
    sites: list[Site]


def _row_dtype(frame: Any) -> Any:
    """The dtype iterrows / apply(axis=1) give each row: the frame's common dtype."""
    return frame.iloc[:0].to_numpy().dtype


def _faithful_renderer(frame_name: str, frame: Any) -> Callable[[str], str] | None:
    """Render columns cast to the row dtype, if any referenced column differs from it."""
    row_dtype = _row_dtype(frame)
    if all(dtype == row_dtype for dtype in frame.dtypes):
        return None

    def render(literal: str) -> str:
        name = ast.literal_eval(literal)
        if name in frame.columns and frame[name].dtype != row_dtype:
            return f'{frame_name}[{literal}].astype("{row_dtype}")'
        return f"{frame_name}[{literal}]"

    return render


def _dtype_text(dtype: Any) -> str:
    return f'"{dtype}"'


class _Finder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self, module: cst.Module) -> None:
        super().__init__()
        self.module = module
        self.sites: list[Site] = []

    def _pos(self, node: cst.CSTNode) -> tuple[int, int, int]:
        r = self.get_metadata(PositionProvider, node)
        return r.start.line, r.start.column, r.end.line

    def _add(self, node: cst.BaseStatement, rule: str, message: str, **kw: Any) -> None:
        line, col, end = self._pos(node)
        source = code(node)
        self.sites.append(
            Site(
                id=len(self.sites),
                rule=rule,
                message=message,
                line=line,
                end_line=end,
                col=col,
                node=node,
                source=source,
                repeatable=is_pure(source),
                **kw,
            )
        )

    # -- PDD001: for i, row in df.iterrows(): ... ----------------------------------

    def visit_For(self, node: cst.For) -> None:
        frame_node = iterrows_frame(node.iter)
        if not isinstance(frame_node, cst.Name) or node.orelse is not None:
            return
        names = target_names(node.target)
        stmts = body_statements(node.body)
        if names is None or len(names) != 2 or len(stmts) != 1:
            return
        idx, row = names
        frame = frame_node.value
        blocked = frozenset((assigned_names(node.body) | {idx}) - {row})
        probe = LoopShape(frame, idx, row, blocked)
        try:
            if probe.statement(stmts[0]) is None:
                return
        except Untranslatable:
            return
        stmt = stmts[0]
        outputs = list(probe.outputs)

        def build(snap: Mapping[str, Any], after: Mapping[str, Any]) -> list[Candidate]:
            df = snap.get(frame)
            if df is None or not hasattr(df, "iterrows"):
                return []
            renderers: list[tuple[str, Callable[[str], str] | None]] = [("plain", None)]
            faithful = _faithful_renderer(frame, df)
            if faithful is not None:
                renderers.append(("row dtype like iterrows", faithful))
            cell_dtypes: list[str | None] = [None]
            written = after.get(frame)
            column = _written_column(stmt)
            if (
                written is not None
                and column is not None
                and column in getattr(written, "columns", [])
            ):
                cell_dtypes.append(_dtype_text(written[column].dtype))
            out: list[Candidate] = []
            for label, render in renderers:
                for elements in ("tolist", "numpy"):
                    for total in ("sum", "exact"):
                        for cell_dtype in cell_dtypes:
                            shape = LoopShape(
                                frame,
                                idx,
                                row,
                                blocked,
                                column=render,
                                elements=elements,
                                total=total,
                            )
                            try:
                                text = shape.statement(stmt)
                            except Untranslatable:
                                continue
                            if text is None:
                                continue
                            if cell_dtype is not None:
                                text = _cast_column_write(text, frame, cell_dtype)
                                if text is None:
                                    continue
                            parts = [label]
                            if elements == "numpy" and "extend" in text:
                                parts.append("numpy scalars")
                            if total == "exact" and "accumulate" in text:
                                parts.append("same summation order")
                            if cell_dtype:
                                parts.append(f"result dtype {cell_dtype}")
                            out.append(
                                Candidate(text, ", ".join(parts), shape.uses_numpy or "np." in text)
                            )
            return _dedupe(out)

        self._add(
            node,
            "PDD001",
            f"`{frame}.iterrows()` loop",
            inputs=_inputs(node, set(names)),
            outputs=outputs,
            loop_vars=list(names),
            build=build,
        )

    # -- statements: comprehension over iterrows, row-wise apply ---------------------

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        if len(node.body) != 1 or not isinstance(node.body[0], cst.Assign):
            return
        assign = node.body[0]
        if len(assign.targets) != 1:
            return
        target = assign.targets[0].target
        if isinstance(assign.value, cst.ListComp) and isinstance(target, cst.Name):
            self._comprehension(node, target.value, assign.value)
            return
        for n in walk(assign.value):
            if isinstance(n, cst.Call) and rowwise_apply(n):
                self._apply(node, target, n)
                return

    def _comprehension(
        self, node: cst.SimpleStatementLine, out_name: str, comp: cst.ListComp
    ) -> None:
        for_in = comp.for_in
        frame_node = iterrows_frame(for_in.iter)
        names = target_names(for_in.target)
        if not isinstance(frame_node, cst.Name) or names is None or len(names) != 2:
            return
        if for_in.inner_for_in is not None or len(for_in.ifs) > 1:
            return
        frame = frame_node.value
        idx, row = names
        try:
            translate(comp.elt, row, frame, blocked=frozenset({idx}))
            if for_in.ifs:
                translate(for_in.ifs[0].test, row, frame, blocked=frozenset({idx}))
        except Untranslatable:
            return

        def build(snap: Mapping[str, Any], after: Mapping[str, Any]) -> list[Candidate]:
            df = snap.get(frame)
            if df is None:
                return []
            out: list[Candidate] = []
            renderers: list[tuple[str, Callable[[str], str] | None]] = [("plain", None)]
            faithful = _faithful_renderer(frame, df)
            if faithful is not None:
                renderers.append(("row dtype like iterrows", faithful))
            for label, render in renderers:
                try:
                    value = translate(comp.elt, row, frame, blocked=frozenset({idx}), column=render)
                    mask = ""
                    if for_in.ifs:
                        cond = translate(
                            for_in.ifs[0].test, row, frame, blocked=frozenset({idx}), column=render
                        )
                        mask = f"[{cond.code}]"
                except Untranslatable:
                    continue
                out.append(
                    Candidate(f"{out_name} = {_p(value)}{mask}.tolist()", label, value.uses_numpy)
                )
                inner = f"{_p(value)}{mask}" if mask else value.code
                out.append(
                    Candidate(
                        f"{out_name} = list(np.asarray({inner}))", f"{label}, numpy scalars", True
                    )
                )
            return _dedupe(out)

        self._add(
            node,
            "PDD001",
            f"list comprehension over `{frame}.iterrows()`",
            inputs=_inputs(node, set(names) | {out_name}),
            outputs=[out_name],
            build=build,
        )

    def _apply(
        self, node: cst.SimpleStatementLine, target: cst.BaseAssignTargetExpression, call: cst.Call
    ) -> None:
        assert isinstance(call.func, cst.Attribute)
        if not isinstance(call.func.value, cst.Name):
            return
        if any(
            a.keyword is not None and a.keyword.value not in {"axis", "func"} for a in call.args
        ):
            return
        found = row_function(call, self.module)
        if found is None:
            return
        row, expr = found
        frame = call.func.value.value
        try:
            translate(expr, row, frame)
        except Untranslatable:
            return
        if isinstance(target, cst.Name):
            output, column = target.value, None
        elif (
            isinstance(target, cst.Subscript)
            and isinstance(target.value, cst.Name)
            and len(target.slice) == 1
            and isinstance(target.slice[0].slice, cst.Index)
            and isinstance(target.slice[0].slice.value, cst.SimpleString)
        ):
            output, column = target.value.value, target.slice[0].slice.value.evaluated_value
        else:
            return

        def build(snap: Mapping[str, Any], after: Mapping[str, Any]) -> list[Candidate]:
            df = snap.get(frame)
            if df is None:
                return []
            renderers: list[tuple[str, Callable[[str], str] | None]] = [("plain", None)]
            faithful = _faithful_renderer(frame, df)
            if faithful is not None:
                renderers.append(("row dtype like apply", faithful))
            observed = after.get(output)
            if (
                column is not None
                and observed is not None
                and column in getattr(observed, "columns", [])
            ):
                observed_dtype: str | None = _dtype_text(observed[column].dtype)
            elif column is None and observed is not None and hasattr(observed, "dtype"):
                observed_dtype = _dtype_text(observed.dtype)
            else:
                observed_dtype = None
            out: list[Candidate] = []
            for label, render in renderers:
                try:
                    t = translate(expr, row, frame, column=render)
                except Untranslatable:
                    continue
                base = _as_series(t, frame)
                wrappers = [("", "")]
                if column is None:
                    wrappers.append((".rename(None)", "unnamed like apply"))
                if observed_dtype is not None:
                    wrappers.append(
                        (f".astype({observed_dtype})", f"result dtype {observed_dtype}")
                    )
                    if column is None:
                        wrappers.append(
                            (
                                f".astype({observed_dtype}).rename(None)",
                                f"result dtype {observed_dtype}, unnamed",
                            )
                        )
                for suffix, extra in wrappers:
                    replacement = f"({base}){suffix}" if suffix else base
                    stmt = node.visit(_Swap(call, cst.parse_expression(replacement)))
                    assert isinstance(stmt, cst.SimpleStatementLine)
                    text = code(stmt.with_changes(leading_lines=[]))
                    out.append(
                        Candidate(
                            text,
                            ", ".join(p for p in (label, extra) if p),
                            t.uses_numpy or "np." in text,
                        )
                    )
            return _dedupe(out)

        self._add(
            node,
            "PDD002",
            f"`{frame}.apply(..., axis=1)`",
            inputs=_inputs(node, set()),
            outputs=[output],
            build=build,
        )


class _Swap(cst.CSTTransformer):
    def __init__(self, target: cst.CSTNode, new: cst.BaseExpression) -> None:
        super().__init__()
        self.target = target
        self.new = new

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:  # type: ignore[override]
        return self.new if original_node is self.target else updated_node


def _as_series(t: Translation, frame: str) -> str:
    return f"pd.Series({t.code}, index={frame}.index)" if t.is_array else t.code


def _written_column(stmt: cst.BaseStatement) -> str | None:
    """The column a loop body writes with `df.at[i, "col"] = ...` (either branch)."""
    for n in walk(stmt):
        if (
            isinstance(n, cst.AssignTarget)
            and isinstance(n.target, cst.Subscript)
            and len(n.target.slice) == 2
        ):
            key = n.target.slice[1].slice
            if isinstance(key, cst.Index) and isinstance(key.value, cst.SimpleString):
                value = key.value.evaluated_value
                return value if isinstance(value, str) else None
    return None


def _cast_column_write(text: str, frame: str, dtype: str) -> str | None:
    """`df["c"] = expr`  ->  `df["c"] = pd.Series(expr, index=df.index).astype(dtype)`."""
    try:
        node = cst.parse_statement(text + "\n")
    except cst.ParserSyntaxError:
        return None
    smalls = small_statements(node)
    if len(smalls) != 1 or not isinstance(smalls[0], cst.Assign):
        return None
    target = smalls[0].targets[0].target
    if not (isinstance(target, cst.Subscript) and isinstance(target.value, cst.Name)):
        return None  # .loc[mask, col] writes keep the existing column's dtype
    value = code(smalls[0].value)
    return f"{code(target)} = pd.Series({value}, index={frame}.index).astype({dtype})"


def _inputs(node: cst.CSTNode, exclude: set[str]) -> list[str]:
    seen: list[str] = []
    for n in names_in(node):
        if (
            n.value not in exclude
            and n.value not in seen
            and n.value not in {"True", "False", "None"}
        ):
            seen.append(n.value)
    return seen


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out = []
    for c in cands:
        if c.code not in seen:
            seen.add(c.code)
            out.append(c)
    return out


def find_sites(source: str, path: str = "<string>") -> SiteMap:
    """Every statement in `source` that `run` knows how to check."""
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    finder = _Finder(wrapper.module)
    wrapper.visit(finder)

    concat_rule = ConcatInLoop(FileContext(path=path, module=wrapper.module))
    wrapper.visit(concat_rule)
    positions = wrapper.resolve(PositionProvider)
    for finding in concat_rule.findings:
        fix = finding.fix
        if fix is None or not isinstance(fix.target, cst.For):
            continue
        meta = dict(fix.meta)
        loop = fix.target
        assign = _growing_assign(loop, meta["target"], meta["callee"])
        if assign is None:
            continue
        a_pos = positions[assign].start
        loop_pos = positions[loop]
        finder.sites.append(
            Site(
                id=len(finder.sites),
                rule="PDD003",
                message=f"`{meta['target']}` grown with concat in a loop",
                line=loop_pos.start.line,
                end_line=loop_pos.end.line,
                col=loop_pos.start.column,
                node=loop,
                source=code(loop),
                inputs=[meta["target"]],
                outputs=[meta["target"]],
                concat=ConcatInfo(
                    meta["target"], meta["callee"], meta["kwargs"], a_pos.line, a_pos.column
                ),
                fix=fix,
            )
        )
    finder.sites.sort(key=lambda s: (s.line, s.col))
    for i, site in enumerate(finder.sites):
        site.id = i
    return SiteMap(wrapper.module, finder.sites)


def _growing_assign(loop: cst.For, target: str, callee: str) -> cst.Assign | None:
    for n in walk(loop.body):
        if (
            isinstance(n, cst.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0].target, cst.Name)
            and n.targets[0].target.value == target
            and isinstance(n.value, cst.Call)
            and code(n.value.func) == callee
        ):
            return n
    return None
