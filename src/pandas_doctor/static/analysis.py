"""Small syntactic helpers the rules share.

Everything here errs on the side of "found a reference": when a rule asks whether a name
is used somewhere, a false "yes" only costs a fix, while a false "no" could break code.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import libcst as cst
from libcst.metadata import QualifiedNameProvider

_RENDER = cst.Module(body=[])


def code(node: cst.CSTNode) -> str:
    """Source text of a node, without its surrounding whitespace."""
    return _RENDER.code_for_node(node).strip()


class _NameCollector(cst.CSTVisitor):
    """Every Name node in a subtree, except attribute names and keyword-argument names."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[cst.Name] = []

    def visit_Name(self, node: cst.Name) -> None:
        self.names.append(node)

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        node.value.visit(self)
        return False  # `x.attr`: only `x` refers to a variable

    def visit_Arg(self, node: cst.Arg) -> bool:
        node.value.visit(self)
        return False  # `f(key=value)`: `key` is not a variable


def names_in(node: cst.CSTNode) -> list[cst.Name]:
    collector = _NameCollector()
    node.visit(collector)
    return collector.names


def mentions(node: cst.CSTNode, name: str) -> int:
    """How many times `name` appears as a variable anywhere inside `node`."""
    return sum(1 for n in names_in(node) if n.value == name)


def dotted_name(node: cst.BaseExpression) -> str | None:
    """`pd.concat` -> "pd.concat"; anything that isn't a plain dotted name -> None."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr.value}" if base else None
    return None


def resolves_to(visitor: cst.CSTVisitor, node: cst.BaseExpression, target: str) -> bool:
    """Whether `node` refers to `target` (e.g. "pandas.concat").

    Uses libcst's import tracking, and falls back to the usual alias spellings when the
    import isn't in this file (common in notebooks split across cells).
    """
    try:
        qualified = visitor.get_metadata(QualifiedNameProvider, node, set())
    except KeyError:
        qualified = set()
    if qualified:
        return any(q.name == target for q in qualified)
    spelled = dotted_name(node)
    if spelled is None:
        return False
    module, _, attr = target.rpartition(".")
    aliases = {"pandas": ("pd", "pandas"), "numpy": ("np", "numpy")}.get(module, (module,))
    return spelled in {f"{alias}.{attr}" for alias in aliases}


def keyword(call: cst.Call, name: str) -> cst.Arg | None:
    for arg in call.args:
        if arg.keyword is not None and arg.keyword.value == name:
            return arg
    return None


def literal_value(node: cst.BaseExpression) -> object:
    """The Python value of a simple literal, or a sentinel when it isn't one."""
    if isinstance(node, cst.Integer):
        return int(node.value.replace("_", ""), 0)
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value
    if isinstance(node, cst.Name) and node.value in {"True", "False", "None"}:
        return {"True": True, "False": False, "None": None}[node.value]
    return _NOT_LITERAL


_NOT_LITERAL = object()


def is_literal(node: cst.BaseExpression) -> bool:
    return literal_value(node) is not _NOT_LITERAL


def target_names(target: cst.BaseExpression) -> list[str] | None:
    """Names bound by a `for` target like `i, row`; None if it isn't plain names."""
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)):
        out: list[str] = []
        for el in target.elements:
            if not isinstance(el, cst.Element) or not isinstance(el.value, cst.Name):
                return None
            out.append(el.value.value)
        return out
    return None


def body_statements(body: cst.BaseSuite) -> list[cst.BaseStatement]:
    """Statements of a block, treating `for x in y: stmt` one-liners like a block."""
    if isinstance(body, cst.IndentedBlock):
        return list(body.body)
    if isinstance(body, cst.SimpleStatementSuite):
        return [cst.SimpleStatementLine(body=body.body)]
    return []


def small_statements(stmt: cst.BaseStatement) -> list[cst.BaseSmallStatement]:
    if isinstance(stmt, cst.SimpleStatementLine):
        return list(stmt.body)
    return []


def walk(node: cst.CSTNode) -> Iterator[cst.CSTNode]:
    """Every node under `node`, depth first, `node` included."""
    yield node
    for child in node.children:
        yield from walk(child)


def contains(node: cst.CSTNode, kinds: tuple[type[cst.CSTNode], ...]) -> bool:
    return any(isinstance(n, kinds) for n in walk(node))


def all_names(module: cst.Module) -> set[str]:
    return {n.value for n in walk(module) if isinstance(n, cst.Name)}


def unique_name(base: str, taken: Iterable[str]) -> str:
    taken = set(taken)
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"
