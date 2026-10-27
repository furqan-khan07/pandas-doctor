"""PDD005: `groupby(...).apply(lambda g: g["col"].sum())` and friends.

`apply` hands every group to Python as a DataFrame. When the function just aggregates a
column (or does arithmetic on a few aggregates), the built-in aggregations do the same
work without leaving C:

    df.groupby("k").apply(lambda g: g["x"].sum() / g["y"].sum())
    ->  df.groupby("k")["x"].sum() / df.groupby("k")["y"].sum()

Not applied automatically: the result's name (and for some functions the dtype) can
differ from what `apply` returns. `pandas-doctor run` checks it on your data.
"""

from __future__ import annotations

import libcst as cst

from pandas_doctor.static.analysis import code, mentions
from pandas_doctor.static.rules.base import Rule
from pandas_doctor.static.rules.pdd004_elementwise_apply import _single_param_lambda

AGGREGATIONS = frozenset(
    {
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "std",
        "var",
        "nunique",
        "first",
        "last",
        "prod",
        "size",
        "sem",
    }
)
NOTE = (
    "Not applied automatically: `apply` names its result differently and can infer a different "
    "dtype. `pandas-doctor run` checks it on your data."
)


class _Rewrite(cst.CSTTransformer):
    """`g["x"].sum()` -> `<groupby>["x"].sum()`; anything else that touches `g` fails."""

    def __init__(self, group: str, groupby: str) -> None:
        super().__init__()
        self.group = group
        self.groupby = groupby
        self.leaves = 0

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        func = original_node.func
        if not (
            isinstance(func, cst.Attribute)
            and func.attr.value in AGGREGATIONS
            and not original_node.args
        ):
            return updated_node
        column = _column_of(func.value, self.group)
        if column is None:
            if (
                isinstance(func.value, cst.Name)
                and func.value.value == self.group
                and func.attr.value == "size"
            ):
                self.leaves += 1
                return cst.parse_expression(f"{self.groupby}.size()")
            return updated_node
        self.leaves += 1
        return cst.parse_expression(f"{self.groupby}[{column}].{func.attr.value}()").with_changes(
            lpar=original_node.lpar, rpar=original_node.rpar
        )


def _column_of(node: cst.BaseExpression, group: str) -> str | None:
    if (
        isinstance(node, cst.Subscript)
        and isinstance(node.value, cst.Name)
        and node.value.value == group
        and len(node.slice) == 1
        and isinstance(node.slice[0].slice, cst.Index)
        and isinstance(node.slice[0].slice.value, cst.SimpleString)
    ):
        return code(node.slice[0].slice.value)
    if (
        isinstance(node, cst.Attribute)
        and isinstance(node.value, cst.Name)
        and node.value.value == group
    ):
        return f'"{node.attr.value}"'
    return None


def _walk(node: cst.CSTNode):  # type: ignore[no-untyped-def]
    yield node
    for child in node.children:
        yield from _walk(child)


def _is_leaf(call: cst.Call, group: str) -> bool:
    func = call.func
    if not (isinstance(func, cst.Attribute) and func.attr.value in AGGREGATIONS and not call.args):
        return False
    if _column_of(func.value, group) is not None:
        return True
    return (
        isinstance(func.value, cst.Name) and func.value.value == group and func.attr.value == "size"
    )


_ALLOWED = (
    cst.BinaryOperation,
    cst.UnaryOperation,
    cst.Integer,
    cst.Float,
    cst.Name,
    cst.Attribute,
    cst.Subscript,
    cst.Call,
    cst.SimpleString,
)


def groupby_apply_rewrite(call: cst.Call) -> str | None:
    if not (isinstance(call.func, cst.Attribute) and call.func.attr.value == "apply"):
        return None
    receiver = call.func.value
    if not (
        isinstance(receiver, cst.Call)
        and isinstance(receiver.func, cst.Attribute)
        and receiver.func.attr.value == "groupby"
    ):
        return None
    if len(call.args) != 1:
        return None
    found = _single_param_lambda(call.args[0].value)
    if found is None:
        return None
    group, body = found
    for n in _walk(body):
        # every call must be an aggregation on the group: anything else (len(x), helper(),
        # random()) would run once instead of once per group
        if isinstance(n, cst.Call) and not _is_leaf(n, group):
            return None
    rewriter = _Rewrite(group, code(receiver))
    new_body = body.visit(rewriter)
    assert isinstance(new_body, cst.BaseExpression)
    if rewriter.leaves == 0 or mentions(new_body, group):
        return None  # `g` is used in some other way
    if not isinstance(body, _ALLOWED):
        return None
    return code(new_body)


class GroupbyApply(Rule):
    code = "PDD005"
    name = "groupby-apply-agg"
    summary = "groupby().apply(lambda g: g[col].agg()); use the built-in aggregation"

    def visit_Call(self, node: cst.Call) -> None:
        replacement = groupby_apply_rewrite(node)
        if replacement is None:
            return
        self.report(
            node,
            "`groupby(...).apply(...)` runs Python once per group just to aggregate",
            suggestion=replacement,
            note=NOTE,
        )
