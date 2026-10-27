"""PDD001: looping over `DataFrame.iterrows()`.

`iterrows` builds a new Series for every row, so the loop body runs at Python speed with
extra allocation on top. For the common loop shapes the rule prints the column-wise
rewrite. It never applies it on its own: rows are upcast to a common dtype (an int column
becomes float when the frame also has a float column), so the rewrite can change dtypes.
`pandas-doctor run` checks the rewrite on the real data.
"""

from __future__ import annotations

from collections.abc import Callable

import libcst as cst

from pandas_doctor.static.analysis import (
    body_statements,
    code,
    dotted_name,
    is_literal,
    mentions,
    small_statements,
    target_names,
    walk,
)
from pandas_doctor.static.rules.base import Rule
from pandas_doctor.static.translate import Translation, Untranslatable, translate

NOTE = (
    "Not applied automatically: iterrows turns ints into floats when the frame also has "
    "float columns, so the rewrite can change dtypes. `pandas-doctor run` checks it on your data."
)


def assigned_names(node: cst.CSTNode) -> set[str]:
    """Names bound anywhere inside `node` (assignments, loop targets, `with ... as`, `:=`)."""
    out: set[str] = set()

    def bind(target: cst.BaseExpression) -> None:
        if isinstance(target, cst.Name):
            out.add(target.value)
        elif isinstance(target, (cst.Tuple, cst.List)):
            for el in target.elements:
                bind(el.value)
        elif isinstance(target, cst.StarredElement):
            bind(target.value)

    for n in walk(node):
        if isinstance(n, cst.Assign):
            for t in n.targets:
                bind(t.target)
        elif isinstance(n, (cst.AugAssign, cst.AnnAssign, cst.For, cst.CompFor, cst.NamedExpr)):
            bind(n.target)
        elif isinstance(n, cst.AsName) and isinstance(n.name, (cst.Name, cst.Tuple, cst.List)):
            bind(n.name)
    return out


def iterrows_frame(call: cst.BaseExpression) -> cst.BaseExpression | None:
    """`df.iterrows()` -> `df`."""
    if (
        isinstance(call, cst.Call)
        and isinstance(call.func, cst.Attribute)
        and call.func.attr.value == "iterrows"
        and not call.args
    ):
        return call.func.value
    return None


def _p(t: Translation) -> str:
    """Parenthesise a translation when a method call is chained onto it."""
    text = t.code
    try:
        node = cst.parse_expression(text)
    except cst.ParserSyntaxError:
        return f"({text})"
    if isinstance(node, (cst.Name, cst.Attribute, cst.Subscript, cst.Call)) or node.lpar:
        return text
    return f"({text})"


class LoopShape:
    """Recognises the loop bodies that have a known column-wise form.

    The static rule uses the plainest rewrite. `pandas-doctor run` builds the same shape
    in a few variants and keeps the first one that checks out as identical:

    * `column` renders a column reference, e.g. with `.astype(...)` to match the dtype
      iterrows gives each row;
    * `elements="numpy"` keeps numpy scalars in lists (iterrows yields numpy scalars,
      `.tolist()` yields Python ones);
    * `total="exact"` sums in the same left-to-right order as the loop, so the result is
      bit-for-bit the same instead of equal up to rounding.

    After `statement()` returns, `outputs` names the variables the rewrite assigns or
    changes and `uses_numpy` says whether it needs `np`.
    """

    def __init__(
        self,
        frame: str,
        idx: str | None,
        row: str,
        blocked: frozenset[str],
        *,
        column: Callable[[str], str] | None = None,
        elements: str = "tolist",
        total: str = "sum",
    ) -> None:
        self.frame = frame
        self.idx = idx
        self.row = row
        self.blocked = blocked
        self.column = column
        self.elements = elements
        self.total = total
        self.outputs: list[str] = []
        self.uses_numpy = False

    def tr(self, expr: cst.BaseExpression) -> Translation:
        t = translate(expr, self.row, self.frame, blocked=self.blocked, column=self.column)
        self.uses_numpy |= t.uses_numpy
        return t

    def _extend(self, acc: str, value: Translation, mask: str = "") -> str:
        self.outputs = [acc]
        if self.elements == "numpy":
            self.uses_numpy = True
            inner = f"{_p(value)}{mask}" if mask else value.code
            return f"{acc}.extend(np.asarray({inner}))"
        return f"{acc}.extend({_p(value)}{mask}.tolist())"

    def _total(self, acc: str, add: bool, value: Translation, mask: str = "") -> str:
        self.outputs = [acc]
        if self.total == "exact":
            self.uses_numpy = True
            ufunc = "add" if add else "subtract"
            inner = f"{_p(value)}{mask}" if mask else value.code
            return (
                f"{acc} = np.{ufunc}.accumulate(np.concatenate(([{acc}], np.asarray({inner}))))[-1]"
            )
        return f"{acc} {'+=' if add else '-='} {_p(value)}{mask}.sum()"

    def cell_target(self, target: cst.BaseExpression) -> str | None:
        """`df.at[i, "col"]` or `df.loc[i, "col"]` on the frame being looped -> `"col"`."""
        if not (
            isinstance(target, cst.Subscript)
            and isinstance(target.value, cst.Attribute)
            and target.value.attr.value in {"at", "loc"}
            and code(target.value.value) == self.frame
            and len(target.slice) == 2
        ):
            return None
        first, second = (el.slice for el in target.slice)
        if not (isinstance(first, cst.Index) and isinstance(second, cst.Index)):
            return None
        if not (isinstance(first.value, cst.Name) and first.value.value == self.idx):
            return None
        if not isinstance(second.value, (cst.SimpleString, cst.ConcatenatedString)):
            return None
        return code(second.value)

    def value_code(self, expr: cst.BaseExpression) -> str:
        """Column form of a value; plain literals and loop-invariant names pass through."""
        if mentions(expr, self.row) == 0:
            if is_literal(expr) or isinstance(expr, (cst.Integer, cst.Float, cst.SimpleString)):
                return code(expr)
            if isinstance(expr, cst.Name) and expr.value not in self.blocked:
                return code(expr)
            raise Untranslatable("value doesn't come from the row")
        return self.tr(expr).code

    def statement(self, stmt: cst.BaseStatement) -> str | None:
        """Suggestion for a loop body that is this single statement."""
        if isinstance(stmt, cst.If):
            return self._if(stmt)
        smalls = small_statements(stmt)
        if len(smalls) != 1:
            return None
        small = smalls[0]
        acc = self._append_target(small)
        if acc is not None:
            assert isinstance(small, cst.Expr) and isinstance(small.value, cst.Call)
            return self._extend(acc, self.tr(small.value.args[0].value))
        if isinstance(small, cst.AugAssign) and isinstance(small.target, cst.Name):
            if isinstance(small.operator, (cst.AddAssign, cst.SubtractAssign)):
                if small.target.value == self.row or small.target.value == self.idx:
                    return None
                add = isinstance(small.operator, cst.AddAssign)
                return self._total(small.target.value, add, self.tr(small.value))
            return None
        if isinstance(small, cst.Assign) and len(small.targets) == 1:
            column = self.cell_target(small.targets[0].target)
            if column is not None:
                self.outputs = [self.frame]
                return f"{self.frame}[{column}] = {self.value_code(small.value)}"
        return None

    def _append_target(self, small: cst.BaseSmallStatement) -> str | None:
        if (
            isinstance(small, cst.Expr)
            and isinstance(small.value, cst.Call)
            and isinstance(small.value.func, cst.Attribute)
            and small.value.func.attr.value == "append"
            and isinstance(small.value.func.value, cst.Name)
            and len(small.value.args) == 1
            and small.value.args[0].keyword is None
            and not small.value.args[0].star
        ):
            name = small.value.func.value.value
            if name not in {self.row, self.idx}:
                return name
        return None

    def _if(self, stmt: cst.If) -> str | None:
        cond = self.tr(stmt.test)
        body = body_statements(stmt.body)
        if len(body) != 1:
            return None
        smalls = small_statements(body[0])
        if len(smalls) != 1:
            return None
        small = smalls[0]
        mask = f"[{cond.code}]"
        if stmt.orelse is None:
            acc = self._append_target(small)
            if acc is not None:
                assert isinstance(small, cst.Expr) and isinstance(small.value, cst.Call)
                return self._extend(acc, self.tr(small.value.args[0].value), mask)
            if (
                isinstance(small, cst.AugAssign)
                and isinstance(small.target, cst.Name)
                and isinstance(small.operator, cst.AddAssign)
                and small.target.value not in {self.row, self.idx}
            ):
                if isinstance(small.value, cst.Integer):
                    self.outputs = [small.target.value]
                    count = f"int(({cond.code}).sum())"
                    k = code(small.value)
                    return f"{small.target.value} += {count if k == '1' else f'{k} * {count}'}"
                return self._total(small.target.value, True, self.tr(small.value), mask)
            if isinstance(small, cst.Assign) and len(small.targets) == 1:
                column = self.cell_target(small.targets[0].target)
                if column is not None:
                    self.outputs = [self.frame]
                    return (
                        f"{self.frame}.loc[{cond.code}, {column}] = {self.value_code(small.value)}"
                    )
            return None
        # if/else that writes the same cell either way -> np.where
        if not isinstance(stmt.orelse, cst.Else):
            return None
        other = body_statements(stmt.orelse.body)
        if len(other) != 1:
            return None
        other_smalls = small_statements(other[0])
        if len(other_smalls) != 1:
            return None
        a, b = small, other_smalls[0]
        if not (isinstance(a, cst.Assign) and isinstance(b, cst.Assign)):
            return None
        if len(a.targets) != 1 or len(b.targets) != 1:
            return None
        col_a = self.cell_target(a.targets[0].target)
        col_b = self.cell_target(b.targets[0].target)
        if col_a is None or col_a != col_b:
            return None
        self.outputs = [self.frame]
        self.uses_numpy = True
        return (
            f"{self.frame}[{col_a}] = np.where({cond.code}, "
            f"{self.value_code(a.value)}, {self.value_code(b.value)})"
        )


class IterrowsLoop(Rule):
    code = "PDD001"
    name = "iterrows-loop"
    summary = "Loop over DataFrame.iterrows(); vectorise it"
    triggers = ("iterrows",)

    def visit_For(self, node: cst.For) -> None:
        frame_node = iterrows_frame(node.iter)
        if frame_node is None:
            return
        frame = code(frame_node)
        message = (
            f"`{frame}.iterrows()` builds a Series for every row, so this loop runs at Python speed"
        )
        self.report(node, message, suggestion=self._suggest(node, frame_node), note=NOTE)

    def visit_ListComp(self, node: cst.ListComp) -> None:
        comp = node.for_in
        frame_node = iterrows_frame(comp.iter)
        if frame_node is None:
            return
        frame = code(frame_node)
        suggestion = None
        names = target_names(comp.target)
        if (
            names is not None
            and len(names) == 2
            and comp.inner_for_in is None
            and len(comp.ifs) <= 1
            and dotted_name(frame_node) is not None
        ):
            shape = LoopShape(frame, names[0], names[1], frozenset({names[0]}))
            try:
                value = shape.tr(node.elt)
                if comp.ifs:
                    cond = shape.tr(comp.ifs[0].test)
                    suggestion = f"{_p(value)}[{cond.code}].tolist()"
                else:
                    suggestion = f"{_p(value)}.tolist()"
            except Untranslatable:
                suggestion = None
        self.report(
            node,
            f"list comprehension over `{frame}.iterrows()` builds a Series for every row",
            suggestion=suggestion,
            note=NOTE,
        )

    def _suggest(self, node: cst.For, frame_node: cst.BaseExpression) -> str | None:
        if dotted_name(frame_node) is None or node.orelse is not None:
            return None
        names = target_names(node.target)
        if names is None or len(names) != 2:
            return None
        idx, row = names
        stmts = body_statements(node.body)
        if len(stmts) != 1:
            return None
        blocked = frozenset((assigned_names(node.body) | {idx}) - {row})
        shape = LoopShape(code(frame_node), idx, row, blocked)
        try:
            return shape.statement(stmts[0])
        except Untranslatable:
            return None
