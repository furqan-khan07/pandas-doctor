"""PDD003: growing a DataFrame with `pd.concat` (or `.append`) inside a loop.

Every `X = pd.concat([X, part])` copies everything collected so far, so the loop is
quadratic. Collecting the parts in a list and concatenating once gives the same frame.

The rule offers a rewrite for the canonical shape. It is an *unsafe* fix (applied only with
`--unsafe-fixes`) because it is identical only when every part has the same columns and
dtypes; the property test found frames with different columns where repeated concat fills
gaps with NaT/<NA> but a single concat uses NaN. `pandas-doctor run` verifies it on real
data. The shape:

    for p in parts:                       X_parts = []
        X = pd.concat([X, p])      ->     for p in parts:
                                              X_parts.append(p)
                                          if X_parts:
                                              X = pd.concat([X, *X_parts])

Conditions, each checked below:

* a plain `for` loop with no `else:`, and the concat is the whole right-hand side;
* `X` is a plain local name and appears nowhere else in the loop (so nothing reads the
  half-built frame), isn't `global`/`nonlocal`, and isn't read by a nested function;
* the call is `concat([X, part])` with at most `ignore_index=`/`sort=False`/`axis=0`;
* nothing can observe the frame half-built: no enclosing `try`/`with` in the same
  function, and no `return`/`yield`/`await` in the loop;
* the appended `part` can't change after it's appended: either a fresh call result, a
  name freshly assigned from a call earlier in the same iteration and not touched after
  the concat, or the loop variable itself when it's used nowhere else.

The fix also assumes the functions you call inside the loop don't keep and later modify
the frames they return. `tests/equivalence/test_pdd003_property.py` checks the
same-columns case across dtypes on every supported pandas version, and keeps the
different-columns counterexamples as regression tests.
"""

from __future__ import annotations

import libcst as cst
from libcst.metadata import PositionProvider

from pandas_doctor.findings import Fix
from pandas_doctor.static.analysis import (
    all_names,
    code,
    contains,
    literal_value,
    mentions,
    names_in,
    resolves_to,
    target_names,
    unique_name,
    walk,
)
from pandas_doctor.static.rules.base import Rule

_ALLOWED_KWARGS = {
    "ignore_index": (True, False),
    "sort": (False,),
    "axis": (0, "index", "rows"),
}
CAVEAT = (
    "Identical when every part has the same columns and dtypes. When they differ, "
    "concatenating one at a time fills the gaps with NaT/<NA> while one concat uses NaN, "
    "so the result can change. `pandas-doctor run` checks it on your data."
)
_SCOPE_BOUNDARY = (cst.FunctionDef, cst.ClassDef, cst.Lambda, cst.Module)


class _NotSafe(Exception):
    """Why the automatic fix isn't offered."""


class _Replace(cst.CSTTransformer):
    def __init__(self, target: cst.CSTNode, new: cst.CSTNode) -> None:
        super().__init__()
        self.target = target
        self.new = new

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:  # type: ignore[override]
        return self.new if original_node is self.target else updated_node


def _objs(call: cst.Call) -> cst.BaseExpression | None:
    if call.args and call.args[0].keyword is None and not call.args[0].star:
        return call.args[0].value
    for arg in call.args:
        if arg.keyword is not None and arg.keyword.value == "objs":
            return arg.value
    return None


def _first_elements(objs: cst.BaseExpression | None) -> list[cst.BaseExpression] | None:
    if isinstance(objs, (cst.List, cst.Tuple)):
        out: list[cst.BaseExpression] = []
        for el in objs.elements:
            if not isinstance(el, cst.Element):
                return None
            out.append(el.value)
        return out
    return None


class ConcatInLoop(Rule):
    code = "PDD003"
    name = "concat-in-loop"
    summary = "pd.concat / append growth inside a loop; collect parts and concat once"
    fix_safety = "unsafe"

    def visit_Assign(self, node: cst.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0].target, cst.Name):
            return
        name = node.targets[0].target.value
        loop = self._loop(node)
        if loop is None:
            return
        # X = X.append(...): DataFrame.append is gone since pandas 2.0
        value = node.value
        if (
            isinstance(value, cst.Call)
            and isinstance(value.func, cst.Attribute)
            and value.func.attr.value == "append"
            and isinstance(value.func.value, cst.Name)
            and value.func.value.value == name
        ):
            self.report(
                value,
                f"`{name} = {name}.append(...)` in a loop: DataFrame.append was removed in "
                "pandas 2.0, and growing a frame this way copies it on every iteration",
                note=f"Collect the pieces in a list, then call `pd.concat([{name}, *pieces])` once after the loop.",
            )
            return
        call = self._concat_growing(node, name)
        if call is None:
            return
        try:
            fix = self._fix(node, call, name, loop)
            skipped = None
        except _NotSafe as why:
            fix, skipped = None, str(why)
        self.report(
            call,
            f"`{name}` grows with pd.concat inside a loop, which copies it on every iteration",
            note=(
                "Collect the parts in a list and call pd.concat once after the loop."
                if fix is None
                else None
            ),
            fix=fix,
            fix_skipped=skipped,
        )

    # -- detection --------------------------------------------------------------

    def _loop(self, node: cst.CSTNode) -> cst.For | cst.While | None:
        for anc in self.ancestors(node):
            if isinstance(anc, (cst.For, cst.While)):
                return anc
            if isinstance(anc, _SCOPE_BOUNDARY):
                return None
        return None

    def _concat_growing(self, node: cst.Assign, name: str) -> cst.Call | None:
        for n in walk(node.value):
            if isinstance(n, cst.Call) and resolves_to(self, n.func, "pandas.concat"):
                elements = _first_elements(_objs(n))
                if elements and any(isinstance(e, cst.Name) and e.value == name for e in elements):
                    return n
        return None

    # -- the fix ------------------------------------------------------------------

    def _fix(self, node: cst.Assign, call: cst.Call, name: str, loop: cst.For | cst.While) -> Fix:
        if not isinstance(loop, cst.For):
            raise _NotSafe("not a `for` loop")
        if loop.asynchronous is not None:
            raise _NotSafe("async for loop")
        if loop.orelse is not None:
            raise _NotSafe("the loop has an `else:` block")
        if node.value is not call:
            raise _NotSafe("the concat result is used inside a bigger expression")
        if not call.args or call.args[0].keyword is not None:
            raise _NotSafe("objs isn't the first positional argument")
        elements = _first_elements(call.args[0].value)
        if elements is None or len(elements) != 2:
            raise _NotSafe("concat isn't called with exactly [frame, part]")
        first, part = elements
        if not (isinstance(first, cst.Name) and first.value == name):
            raise _NotSafe(f"`{name}` isn't the first item, so parts are prepended")
        for arg in call.args[1:]:
            if arg.keyword is None or arg.star:
                raise _NotSafe("extra positional arguments to concat")
            allowed = _ALLOWED_KWARGS.get(arg.keyword.value)
            if allowed is None:
                raise _NotSafe(f"`{arg.keyword.value}=` changes how pieces combine")
            if literal_value(arg.value) not in allowed:
                raise _NotSafe(f"`{arg.keyword.value}=` isn't a plain literal we can keep")
        if mentions(loop, name) != 2:
            raise _NotSafe(f"`{name}` is read elsewhere in the loop")
        if contains(loop.body, (cst.Yield, cst.Await, cst.Return)):
            raise _NotSafe("the loop can exit or pause midway (return/yield/await)")
        self._check_scope(loop, name)
        self._check_part(loop, node, part)

        module = self.ctx.module
        parts = unique_name(f"{name}_parts", all_names(module))
        append = cst.Expr(
            value=cst.Call(
                func=cst.Attribute(value=cst.Name(parts), attr=cst.Name("append")),
                args=[cst.Arg(value=part)],
            )
        )
        new_loop = loop.visit(_Replace(node, append))
        assert isinstance(new_loop, cst.For)
        kwargs = "".join(
            f", {code(a.with_changes(comma=cst.MaybeSentinel.DEFAULT))}" for a in call.args[1:]
        )
        cfg = module.config_for_parsing
        init = cst.parse_statement(f"{parts} = []\n", config=cfg)
        after = cst.parse_statement(
            f"if {parts}:\n{module.default_indent}{name} = {code(call.func)}([{name}, *{parts}]{kwargs})\n",
            config=cfg,
        )
        replacement = cst.FlattenSentinel(
            [
                init.with_changes(leading_lines=loop.leading_lines),
                new_loop.with_changes(leading_lines=[]),
                after,
            ]
        )
        return Fix(
            target=loop,
            replacement=replacement,
            description=f"collect parts in `{parts}` and concat once after the loop",
            safe=False,
            caveat=CAVEAT,
        )

    def _check_scope(self, loop: cst.For, name: str) -> None:
        scope_node: cst.CSTNode | None = None
        for anc in self.ancestors(loop):
            if isinstance(anc, (cst.Try, cst.TryStar, cst.With)):
                raise _NotSafe(
                    "inside try/with, so a half-built frame could be observed after an error"
                )
            if isinstance(anc, _SCOPE_BOUNDARY):
                scope_node = anc
                break
        if isinstance(scope_node, cst.ClassDef):
            raise _NotSafe("class body")
        if isinstance(scope_node, cst.FunctionDef):
            for n in walk(scope_node.body):
                if isinstance(n, (cst.Global, cst.Nonlocal)) and any(
                    item.name.value == name for item in n.names
                ):
                    raise _NotSafe(f"`{name}` is global/nonlocal")
                if isinstance(n, (cst.FunctionDef, cst.Lambda, cst.ClassDef)) and mentions(n, name):
                    raise _NotSafe(f"a nested function reads `{name}`")

    def _check_part(self, loop: cst.For, assign: cst.Assign, part: cst.BaseExpression) -> None:
        """The appended object must not change between being appended and the final concat."""
        if isinstance(part, cst.Call):
            return  # a fresh object nobody else holds
        if not isinstance(part, cst.Name):
            raise _NotSafe("can't prove the appended object isn't changed later in the loop")
        pname = part.value
        loop_vars = target_names(loop.target) or []
        if pname in loop_vars:
            if mentions(loop.body, pname) != 1:
                raise _NotSafe(
                    f"loop variable `{pname}` is used elsewhere in the loop, so it might be modified"
                )
            return
        # A name assigned from a call earlier in this iteration and untouched after the concat.
        end = self.get_metadata(PositionProvider, assign).end
        concat_at = (end.line, end.column)
        uses: list[tuple[tuple[int, int], cst.Name]] = []
        for n in names_in(loop.body):
            if n.value == pname:
                pos = self.get_metadata(PositionProvider, n).start
                uses.append(((pos.line, pos.column), n))
        uses.sort(key=lambda item: item[0])
        if not uses:
            raise _NotSafe(f"`{pname}` isn't defined in the loop")
        binding = self._binding_of(loop, uses[0][1])
        if binding is None or not isinstance(binding.value, cst.Call):
            raise _NotSafe(f"`{pname}` isn't freshly created from a call in each iteration")
        if any(pos > concat_at for pos, _ in uses):
            raise _NotSafe(f"`{pname}` is used after being appended, so it might be modified")

    def _binding_of(self, loop: cst.For, name_node: cst.Name) -> cst.Assign | None:
        """The `name = ...` statement whose target is `name_node`, directly in the loop body."""
        for n in walk(loop.body):
            if (
                isinstance(n, cst.Assign)
                and len(n.targets) == 1
                and n.targets[0].target is name_node
            ):
                parent = self.parent(n)
                grand = self.parent(parent) if parent is not None else None
                if grand is loop.body or parent is loop.body:
                    return n
        return None
