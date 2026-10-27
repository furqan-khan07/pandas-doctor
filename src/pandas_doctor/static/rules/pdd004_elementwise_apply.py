"""PDD004: `s.apply(lambda x: ...)` / `.map(lambda x: ...)` where pandas has a vector method.

`df["name"].apply(lambda x: x.strip().lower())` calls Python once per element;
`df["name"].str.strip().str.lower()` does the same work in one pass. Likewise for
arithmetic, comparisons, `len(x)` and datetime fields.

Not applied automatically: the two differ on missing values (`x.lower()` raises on NaN,
`.str.lower()` returns NaN) and on non-string elements. `pandas-doctor run` checks it on
your data.

To avoid flagging `DataFrame.apply`, which passes whole columns and is already
column-wise, the rule only fires on `.map(...)`/`.applymap(...)`, or on `.apply(...)`
called on an obvious column (`df["col"]`, `df.col`), or when the lambda uses string or
datetime methods that only make sense on single values.
"""

from __future__ import annotations

import libcst as cst

from pandas_doctor.findings import Severity
from pandas_doctor.static.analysis import code
from pandas_doctor.static.rules.base import Rule
from pandas_doctor.static.translate import DT_ATTRS, STR_METHODS, Untranslatable, translate

NOTE = (
    "Not applied automatically: on missing values `x.lower()` raises while `.str.lower()` "
    "gives NaN, and non-string values behave differently. `pandas-doctor run` checks it on your data."
)


def _single_param_lambda(arg: cst.BaseExpression) -> tuple[str, cst.BaseExpression] | None:
    if not isinstance(arg, cst.Lambda):
        return None
    p = arg.params
    if len(p.params) != 1 or p.kwonly_params or p.posonly_params or p.star_kwarg is not None:
        return None
    if isinstance(p.star_arg, (cst.Param, cst.ParamStar)) or p.params[0].default is not None:
        return None
    return p.params[0].name.value, arg.body


def _is_column(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.Subscript) and len(node.slice) == 1:
        inner = node.slice[0].slice
        return isinstance(inner, cst.Index) and isinstance(
            inner.value, (cst.SimpleString, cst.Integer)
        )
    return isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)


# Methods/fields that exist on single str/Timestamp values but not on Series, so using
# them proves the lambda receives single elements. (`replace` is left out: Series has it.)
_SCALAR_ONLY = (STR_METHODS - {"replace"}) | DT_ATTRS
_GROUPERS = {"groupby", "resample", "rolling", "expanding", "ewm", "pipe"}


def _uses_scalar_methods(body: cst.BaseExpression, param: str) -> bool:
    """Whether the lambda calls a str method / reads a datetime field on its argument."""
    return any(
        isinstance(n, cst.Attribute)
        and n.attr.value in _SCALAR_ONLY
        and isinstance(n.value, cst.Name)
        and n.value.value == param
        for n in _walk(body)
    )


def _from_grouper(node: cst.BaseExpression) -> bool:
    """`df.groupby(...)`, `s.rolling(3)`, ... anywhere in the receiver chain."""
    for n in _walk(node):
        if (
            isinstance(n, cst.Call)
            and isinstance(n.func, cst.Attribute)
            and n.func.attr.value in _GROUPERS
        ):
            return True
    return False


def _walk(node: cst.CSTNode):  # type: ignore[no-untyped-def]
    yield node
    for child in node.children:
        yield from _walk(child)


def elementwise_apply(call: cst.Call) -> tuple[cst.BaseExpression, str, cst.BaseExpression] | None:
    """(receiver, lambda parameter, lambda body) if this is a vectorisable element-wise apply."""
    if not isinstance(call.func, cst.Attribute):
        return None
    method = call.func.attr.value
    if method not in {"apply", "map", "applymap"}:
        return None
    if len(call.args) != 1 or call.args[0].keyword is not None or call.args[0].star:
        return None
    found = _single_param_lambda(call.args[0].value)
    if found is None:
        return None
    param, body = found
    receiver = call.func.value
    if _from_grouper(receiver):
        return None  # each call gets a whole group, not one element
    if method == "apply" and not (_is_column(receiver) or _uses_scalar_methods(body, param)):
        return None
    if isinstance(receiver, cst.Attribute) and receiver.attr.value == "str":
        return None
    return receiver, param, body


def vectorised(call: cst.Call) -> str | None:
    found = elementwise_apply(call)
    if found is None:
        return None
    receiver, param, body = found
    try:
        t = translate(body, param, code(receiver), elementwise=True)
    except Untranslatable:
        return None
    if t.is_array:
        return f"pd.Series({t.code}, index={code(receiver)}.index)"
    return t.code


class ElementwiseApply(Rule):
    code = "PDD004"
    name = "elementwise-apply"
    summary = "s.apply/map(lambda x: ...) with a vector equivalent (.str, .dt, arithmetic)"
    severity = Severity.WARNING

    def visit_Call(self, node: cst.Call) -> None:
        if elementwise_apply(node) is None:
            return
        replacement = vectorised(node)
        if replacement is None:
            return  # nothing to suggest: don't flag lambdas we can't map one to one
        assert isinstance(node.func, cst.Attribute)
        self.report(
            node,
            f"`.{node.func.attr.value}(lambda ...)` calls Python once per element",
            suggestion=replacement,
            note=NOTE,
        )
