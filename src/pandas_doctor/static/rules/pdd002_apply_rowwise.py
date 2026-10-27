"""PDD002: `DataFrame.apply(func, axis=1)`.

Row-wise apply calls a Python function once per row with the row as a Series. When the
function is plain arithmetic, comparisons or a conditional on columns, the rule prints the
column-wise expression. Like PDD001 it is never applied automatically, because each row is
upcast to a common dtype before your function sees it.
"""

from __future__ import annotations

import libcst as cst

from pandas_doctor.static.analysis import code, dotted_name, keyword, literal_value
from pandas_doctor.static.rules.base import Rule
from pandas_doctor.static.translate import Untranslatable, translate

NOTE = (
    "Not applied automatically: apply(axis=1) hands your function each row as a Series, "
    "which turns ints into floats when the frame also has float columns, so the rewrite can "
    "change dtypes. `pandas-doctor run` checks it on your data."
)


def rowwise_apply(call: cst.Call) -> bool:
    if not (isinstance(call.func, cst.Attribute) and call.func.attr.value == "apply"):
        return False
    axis = keyword(call, "axis")
    if axis is None:
        return False
    return literal_value(axis.value) in (1, "columns")


def _single_param(params: cst.Parameters) -> cst.Param | None:
    """The only parameter, if the signature is exactly `(x)`."""
    if len(params.params) != 1 or params.kwonly_params or params.posonly_params:
        return None
    if isinstance(params.star_arg, (cst.Param, cst.ParamStar)) or params.star_kwarg is not None:
        return None
    only = params.params[0]
    return only if only.default is None else None


def row_function(call: cst.Call, module: cst.Module) -> tuple[str, cst.BaseExpression] | None:
    """The row parameter and the returned expression of the applied function, if simple."""
    func_args = [a for a in call.args if a.keyword is None]
    func_kw = keyword(call, "func")
    func = func_args[0].value if func_args else (func_kw.value if func_kw else None)
    if isinstance(func, cst.Lambda):
        param = _single_param(func.params)
        return (param.name.value, func.body) if param is not None else None
    if isinstance(func, cst.Name):
        for stmt in module.body:
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value == func.value:
                return _simple_def(stmt)
    return None


def _simple_def(fn: cst.FunctionDef) -> tuple[str, cst.BaseExpression] | None:
    param = _single_param(fn.params)
    if fn.decorators or fn.asynchronous or param is None:
        return None
    if not isinstance(fn.body, cst.IndentedBlock):
        return None
    stmts = list(fn.body.body)
    # skip a docstring
    if stmts and isinstance(stmts[0], cst.SimpleStatementLine) and len(stmts[0].body) == 1:
        first = stmts[0].body[0]
        if isinstance(first, cst.Expr) and isinstance(
            first.value, (cst.SimpleString, cst.ConcatenatedString)
        ):
            stmts = stmts[1:]
    if len(stmts) != 1 or not isinstance(stmts[0], cst.SimpleStatementLine):
        return None
    ret = stmts[0].body[0] if len(stmts[0].body) == 1 else None
    if isinstance(ret, cst.Return) and ret.value is not None:
        return param.name.value, ret.value
    return None


class _Swap(cst.CSTTransformer):
    def __init__(self, target: cst.CSTNode, new: cst.BaseExpression) -> None:
        super().__init__()
        self.target = target
        self.new = new

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:  # type: ignore[override]
        return self.new if original_node is self.target else updated_node


def vectorised_apply(call: cst.Call, module: cst.Module) -> str | None:
    """Column-wise code that replaces the `apply` call, or None."""
    if any(a.keyword is not None and a.keyword.value not in {"axis", "func"} for a in call.args):
        return None  # raw=, result_type=, args= ... change what the function sees
    assert isinstance(call.func, cst.Attribute)
    frame_node = call.func.value
    if dotted_name(frame_node) is None:
        return None
    found = row_function(call, module)
    if found is None:
        return None
    row, expr = found
    frame = code(frame_node)
    try:
        result = translate(expr, row, frame)
    except Untranslatable:
        return None
    if result.is_array:
        return f"pd.Series({result.code}, index={frame}.index)"
    return result.code


class RowwiseApply(Rule):
    code = "PDD002"
    name = "rowwise-apply"
    summary = "DataFrame.apply(..., axis=1); use column operations"

    def visit_Call(self, node: cst.Call) -> None:
        if not rowwise_apply(node):
            return
        assert isinstance(node.func, cst.Attribute)
        frame = code(node.func.value)
        replacement = vectorised_apply(node, self.ctx.module)
        suggestion = None
        if replacement is not None:
            suggestion = self._in_statement(node, replacement)
        self.report(
            node,
            f"`{frame}.apply(..., axis=1)` calls Python once per row",
            suggestion=suggestion,
            note=NOTE,
        )

    def _in_statement(self, node: cst.Call, replacement: str) -> str:
        """Show the whole statement with the call swapped, which reads better than a fragment."""
        stmt = next(
            (a for a in self.ancestors(node) if isinstance(a, cst.SimpleStatementLine)), None
        )
        if stmt is None:
            return replacement
        swapped = stmt.visit(_Swap(node, cst.parse_expression(replacement)))
        assert isinstance(swapped, cst.SimpleStatementLine)
        return code(swapped.with_changes(leading_lines=[]))
