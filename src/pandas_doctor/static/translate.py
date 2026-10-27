"""Turn an expression over one row into the same expression over whole columns.

`row["price"] * row["qty"]` becomes `df["price"] * df["qty"]`. The translator refuses
anything it can't map one to one: calls to your own functions, `in` checks, `and`/`or`
on non-boolean values, names that change inside the loop, and so on. A refusal just
means no suggestion; it never produces a guess.

The translation is not claimed to be identical. Row-wise code sees upcast values (an
int column turns into float when the frame also has float columns), so the static
checker only suggests it, and `pandas-doctor run` checks it on the real data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import libcst as cst

from pandas_doctor.static.analysis import code, dotted_name


class Untranslatable(Exception):
    """The expression has no safe column-wise equivalent."""


@dataclass(frozen=True)
class Translation:
    code: str
    vector: bool  # refers to at least one column
    is_array: bool  # result is a numpy array (np.where), not a Series
    uses_numpy: bool


@dataclass(frozen=True)
class _T:
    node: cst.BaseExpression
    vector: bool = False
    array: bool = False
    boolean: bool = False


UFUNCS = frozenset(
    {
        "sqrt",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "floor",
        "ceil",
        "trunc",
        "fabs",
        "abs",
        "absolute",
        "sign",
        "square",
    }
)
_MATH_TO_NUMPY = {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}
STR_METHODS = frozenset(
    {
        "lower",
        "upper",
        "strip",
        "lstrip",
        "rstrip",
        "title",
        "capitalize",
        "casefold",
        "swapcase",
        "zfill",
        "startswith",
        "endswith",
        "replace",
        "isdigit",
        "isalpha",
        "isalnum",
        "isnumeric",
        "isdecimal",
        "isspace",
        "islower",
        "isupper",
        "istitle",
    }
)
_BOOL_STR_METHODS = frozenset(
    m for m in STR_METHODS if m.startswith(("is", "startswith", "endswith"))
)
DT_ATTRS = frozenset(
    {
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
    }
)
_SAFE_SCALAR_CALLS = frozenset({"abs", "round", "min", "max", "float", "int", "str", "len", "bool"})
_ARITH = (
    cst.Add,
    cst.Subtract,
    cst.Multiply,
    cst.Divide,
    cst.FloorDivide,
    cst.Modulo,
    cst.Power,
    cst.BitAnd,
    cst.BitOr,
    cst.BitXor,
)
_COMPARE = (
    cst.LessThan,
    cst.GreaterThan,
    cst.LessThanEqual,
    cst.GreaterThanEqual,
    cst.Equal,
    cst.NotEqual,
)


@cache
def _series_attrs() -> frozenset[str]:
    import pandas as pd

    return frozenset(dir(pd.Series))


def _atomic(node: cst.BaseExpression) -> bool:
    if node.lpar:
        return True
    return isinstance(
        node,
        (
            cst.Name,
            cst.Attribute,
            cst.Subscript,
            cst.Call,
            cst.Integer,
            cst.Float,
            cst.SimpleString,
        ),
    )


def _paren(node: cst.BaseExpression) -> cst.BaseExpression:
    if _atomic(node):
        return node
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _expr(text: str) -> cst.BaseExpression:
    return cst.parse_expression(text)


class _Translator:
    def __init__(
        self,
        row: str,
        column: Callable[[str], str],
        blocked: frozenset[str],
        series: str | None = None,
    ) -> None:
        self.row = row
        self.column = column
        self.blocked = blocked
        # elementwise mode: `row` is one element of this Series, not a row of a frame
        self.series = series
        self.uses_numpy = False

    # Each method returns the rewritten node plus what kind of value it is.

    def tx(self, node: cst.BaseExpression) -> _T:
        if isinstance(node, cst.Name):
            if node.value == self.row:
                if self.series is not None:
                    return _T(
                        _paren(_expr(self.series)).with_changes(lpar=node.lpar, rpar=node.rpar),
                        vector=True,
                    )
                raise Untranslatable(f"`{self.row}` is used as a whole row")
            if node.value in self.blocked:
                raise Untranslatable(f"`{node.value}` changes inside the loop")
            return _T(node, boolean=node.value in {"True", "False"})
        if isinstance(
            node, (cst.Integer, cst.Float, cst.Imaginary, cst.SimpleString, cst.ConcatenatedString)
        ):
            return _T(node)
        if isinstance(node, cst.Subscript):
            return self._subscript(node)
        if isinstance(node, cst.Attribute):
            return self._attribute(node)
        if isinstance(node, cst.BinaryOperation):
            return self._binary(node)
        if isinstance(node, cst.UnaryOperation):
            return self._unary(node)
        if isinstance(node, cst.Comparison):
            return self._comparison(node)
        if isinstance(node, cst.BooleanOperation):
            return self._boolean(node)
        if isinstance(node, cst.IfExp):
            return self._ifexp(node)
        if isinstance(node, cst.Call):
            return self._call(node)
        raise Untranslatable(f"unsupported expression: {code(node)}")

    def _column_node(self, literal: str, original: cst.BaseExpression) -> cst.BaseExpression:
        new = _expr(self.column(literal))
        return new.with_changes(lpar=original.lpar, rpar=original.rpar)

    def _subscript(self, node: cst.Subscript) -> _T:
        if self.series is not None and self._mentions_row(node):
            raise Untranslatable("indexing into each element")
        if isinstance(node.value, cst.Name) and node.value.value == self.row:
            if len(node.slice) != 1 or not isinstance(node.slice[0].slice, cst.Index):
                raise Untranslatable("row indexed with a slice")
            key = node.slice[0].slice.value
            if not isinstance(key, (cst.SimpleString, cst.ConcatenatedString)):
                raise Untranslatable("row indexed by something other than a column name")
            return _T(self._column_node(code(key), node), vector=True)
        if self._mentions_row(node):
            raise Untranslatable(f"unsupported indexing: {code(node)}")
        self._check_blocked(node)
        return _T(node)

    def _attribute(self, node: cst.Attribute) -> _T:
        attr = node.attr.value
        if (
            self.series is None
            and isinstance(node.value, cst.Name)
            and node.value.value == self.row
        ):
            if attr in _series_attrs() or attr.startswith("_"):
                raise Untranslatable(f"`{self.row}.{attr}` is a Series attribute, not a column")
            return _T(self._column_node(f'"{attr}"', node), vector=True)
        base = self.tx(node.value)
        if not base.vector:
            return _T(node)
        if base.array:
            raise Untranslatable("attribute on an array result")
        if attr in DT_ATTRS:
            return _T(
                _expr(f"{code(_paren(base.node))}.dt.{attr}").with_changes(
                    lpar=node.lpar, rpar=node.rpar
                ),
                vector=True,
            )
        raise Untranslatable(f"unsupported attribute `.{attr}` on a column value")

    def _binary(self, node: cst.BinaryOperation) -> _T:
        if not isinstance(node.operator, _ARITH):
            raise Untranslatable("unsupported operator")
        left, right = self.tx(node.left), self.tx(node.right)
        vector = left.vector or right.vector
        array = vector and all(t.array for t in (left, right) if t.vector)
        boolean = (
            isinstance(node.operator, (cst.BitAnd, cst.BitOr, cst.BitXor))
            and left.boolean
            and right.boolean
        )
        return _T(node.with_changes(left=left.node, right=right.node), vector, array, boolean)

    def _unary(self, node: cst.UnaryOperation) -> _T:
        inner = self.tx(node.expression)
        if isinstance(node.operator, cst.Not):
            if not inner.vector:
                return _T(node.with_changes(expression=inner.node), boolean=True)
            if not inner.boolean:
                raise Untranslatable("`not` on a non-boolean column value")
            new = cst.UnaryOperation(operator=cst.BitInvert(), expression=_paren(inner.node))
            return _T(new.with_changes(lpar=node.lpar, rpar=node.rpar), True, inner.array, True)
        return _T(node.with_changes(expression=inner.node), inner.vector, inner.array, False)

    def _comparison(self, node: cst.Comparison) -> _T:
        if not all(isinstance(c.operator, _COMPARE) for c in node.comparisons):
            raise Untranslatable("`in`/`is` checks have no direct column form")
        left = self.tx(node.left)
        parts = [self.tx(c.comparator) for c in node.comparisons]
        vector = left.vector or any(p.vector for p in parts)
        array = vector and all(t.array for t in (left, *parts) if t.vector)
        if len(parts) == 1 or not vector:
            new = node.with_changes(
                left=left.node,
                comparisons=[
                    c.with_changes(comparator=p.node)
                    for c, p in zip(node.comparisons, parts, strict=True)
                ],
            )
            return _T(new, vector, array, True)
        # a < b < c  ->  (a < b) & (b < c)
        pieces: list[cst.BaseExpression] = []
        lhs = left.node
        for comp, part in zip(node.comparisons, parts, strict=True):
            pieces.append(
                cst.Comparison(
                    left=lhs, comparisons=[comp.with_changes(comparator=part.node)]
                ).with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
            )
            lhs = part.node
        out = pieces[0]
        for piece in pieces[1:]:
            out = cst.BinaryOperation(left=out, operator=cst.BitAnd(), right=piece)
        return _T(_paren(out) if node.lpar else out, vector, array, True)

    def _boolean(self, node: cst.BooleanOperation) -> _T:
        left, right = self.tx(node.left), self.tx(node.right)
        if not (left.vector or right.vector):
            return _T(
                node.with_changes(left=left.node, right=right.node),
                boolean=left.boolean and right.boolean,
            )
        if not (left.boolean and right.boolean):
            raise Untranslatable("`and`/`or` on non-boolean values returns an operand, not a bool")
        op: cst.BaseBinaryOp = cst.BitAnd() if isinstance(node.operator, cst.And) else cst.BitOr()
        new = cst.BinaryOperation(left=_paren(left.node), operator=op, right=_paren(right.node))
        array = all(t.array for t in (left, right) if t.vector)
        return _T(new.with_changes(lpar=node.lpar, rpar=node.rpar), True, array, True)

    def _ifexp(self, node: cst.IfExp) -> _T:
        test, body, orelse = self.tx(node.test), self.tx(node.body), self.tx(node.orelse)
        if not test.vector:
            if body.vector != orelse.vector:
                raise Untranslatable("branches mix a column and a single value")
            return _T(
                node.with_changes(test=test.node, body=body.node, orelse=orelse.node),
                body.vector,
                body.array and orelse.array,
            )
        if not test.boolean:
            raise Untranslatable("condition is not a boolean expression")
        self.uses_numpy = True
        new = _expr(f"np.where({code(test.node)}, {code(body.node)}, {code(orelse.node)})")
        return _T(new, vector=True, array=True, boolean=body.boolean and orelse.boolean)

    def _call(self, node: cst.Call) -> _T:
        if any(a.star or a.keyword is not None for a in node.args):
            raise Untranslatable("call with keyword or star arguments")
        func = node.func
        name = dotted_name(func)
        if isinstance(func, cst.Attribute) and (
            name is None or not name.startswith(("np.", "numpy.", "math.", "pd.", "pandas."))
        ):
            return self._method(node, func)
        args = [self.tx(a.value) for a in node.args]
        vector = any(a.vector for a in args)
        if not vector:
            if name in _SAFE_SCALAR_CALLS or (
                name and name.split(".")[0] in {"math", "np", "numpy"}
            ):
                return _T(
                    node.with_changes(
                        args=[
                            a.with_changes(value=t.node)
                            for a, t in zip(node.args, args, strict=True)
                        ]
                    )
                )
            raise Untranslatable(
                f"call to `{name or code(func)}` would run once instead of per row"
            )
        arg_code = [code(t.node) for t in args]
        if name in {"abs", "round"}:
            new: cst.BaseExpression = node.with_changes(
                args=[a.with_changes(value=t.node) for a, t in zip(node.args, args, strict=True)]
            )
            return _T(new, True, all(t.array for t in args if t.vector))
        if name in {"min", "max"} and len(args) == 2:
            self.uses_numpy = True
            fn = "np.minimum" if name == "min" else "np.maximum"
            return _T(
                _expr(f"{fn}({arg_code[0]}, {arg_code[1]})"),
                True,
                all(t.array for t in args if t.vector),
            )
        if name in {"float", "int", "str", "bool"} and len(args) == 1:
            return _T(
                _expr(f"{code(_paren(args[0].node))}.astype({name})"),
                True,
                args[0].array,
                name == "bool",
            )
        if name == "len" and len(args) == 1 and not args[0].array:
            return _T(_expr(f"{code(_paren(args[0].node))}.str.len()"), True)
        if (
            name in {"pd.isna", "pd.isnull", "pandas.isna", "pandas.isnull"}
            and len(args) == 1
            and not args[0].array
        ):
            return _T(_expr(f"{code(_paren(args[0].node))}.isna()"), True, False, True)
        if (
            name in {"pd.notna", "pd.notnull", "pandas.notna", "pandas.notnull"}
            and len(args) == 1
            and not args[0].array
        ):
            return _T(_expr(f"{code(_paren(args[0].node))}.notna()"), True, False, True)
        if name and "." in name:
            module, _, fn_name = name.rpartition(".")
            if module in {"math", "np", "numpy"}:
                fn_name = _MATH_TO_NUMPY.get(fn_name, fn_name)
                if fn_name in UFUNCS or fn_name == "isnan":
                    self.uses_numpy = True
                    joined = ", ".join(arg_code)
                    return _T(
                        _expr(f"np.{fn_name}({joined})"),
                        True,
                        all(t.array for t in args if t.vector),
                        fn_name == "isnan",
                    )
        raise Untranslatable(f"`{name or code(func)}` can't be applied to a whole column")

    def _method(self, node: cst.Call, func: cst.Attribute) -> _T:
        base = self.tx(func.value)
        method = func.attr.value
        if not base.vector:
            if self._mentions_row(node):
                raise Untranslatable(f"unsupported call: {code(node)}")
            raise Untranslatable(f"method call `.{method}()` would run once instead of per row")
        args = [self.tx(a.value) for a in node.args]
        if any(a.vector for a in args):
            raise Untranslatable(f"`.{method}()` with a per-row argument")
        if base.array:
            raise Untranslatable("method call on an array result")
        joined = ", ".join(code(a.node) for a in args)
        base_code = code(_paren(base.node))
        if method in STR_METHODS:
            if method == "replace" and len(args) != 2:
                raise Untranslatable("str.replace with a count")
            return _T(
                _expr(f"{base_code}.str.{method}({joined})"),
                True,
                False,
                method in _BOOL_STR_METHODS,
            )
        if method == "weekday" and not args:
            return _T(_expr(f"{base_code}.dt.weekday"), True)
        if method == "round":
            return _T(_expr(f"{base_code}.round({joined})"), True)
        raise Untranslatable(f"`.{method}()` has no checked column form")

    def _mentions_row(self, node: cst.CSTNode) -> bool:
        from pandas_doctor.static.analysis import mentions

        return mentions(node, self.row) > 0

    def _check_blocked(self, node: cst.CSTNode) -> None:
        from pandas_doctor.static.analysis import names_in

        for n in names_in(node):
            if n.value in self.blocked:
                raise Untranslatable(f"`{n.value}` changes inside the loop")


def translate(
    expr: cst.BaseExpression,
    row: str,
    frame: str,
    *,
    blocked: frozenset[str] = frozenset(),
    column: Callable[[str], str] | None = None,
    elementwise: bool = False,
) -> Translation:
    """Translate `expr` (which reads `row`) into column form over `frame`.

    `column` renders one column reference from its quoted name; the default is
    `frame[name]`. With `elementwise`, `row` stands for one element of the Series
    `frame` (as in `s.apply(lambda x: ...)`), and bare uses of it become the Series.
    Raises Untranslatable when there's no one-to-one equivalent.
    """
    render = column or (lambda literal: f"{frame}[{literal}]")
    tr = _Translator(row, render, blocked, series=frame if elementwise else None)
    result = tr.tx(expr)
    if not result.vector:
        raise Untranslatable("the expression doesn't use any column")
    return Translation(code(result.node), True, result.array, tr.uses_numpy)
