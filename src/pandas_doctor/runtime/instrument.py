"""Wrap the sites in a script so the runtime can watch them.

This works on Python's `ast` (not libcst) because the result is compiled and run, and
`ast` keeps the original line numbers: tracebacks from the user's code still point at
their file and lines.

Each site statement S becomes

    __pdd_t_<id> = __pdd__.enter(<id>, locals(), globals())
    S
    __pdd__.exit(<id>, __pdd_t_<id>, locals(), globals())

and, for concat loops, the growing assignment `X = pd.concat([X, part], ...)` becomes
`X = __pdd__.concat_step(<id>, lambda objs: pd.concat(objs, ...), X, part)`, which runs
the same concat but keeps a reference to each part. Top-level statements also get a
`__pdd__.mark(<line>)` before them so the report can say where the time went.
"""

from __future__ import annotations

import ast
from typing import Any

from pandas_doctor.runtime.sites import Site


def _stmts(text: str, like: ast.AST) -> list[ast.stmt]:
    nodes = ast.parse(text).body
    for top in nodes:
        for n in ast.walk(top):
            if hasattr(n, "lineno"):
                ast.copy_location(n, like)
                n.end_lineno = getattr(like, "end_lineno", None)  # type: ignore[attr-defined]
                n.end_col_offset = getattr(like, "end_col_offset", None)  # type: ignore[attr-defined]
    return nodes


class _Instrumenter(ast.NodeTransformer):
    def __init__(self, source: str, sites: list[Site]) -> None:
        self.source = source
        self.by_pos = {(s.line, s.col): s for s in sites}
        self.concats = {
            (s.concat.assign_line, s.concat.assign_col): s for s in sites if s.concat is not None
        }
        self.wrapped: set[int] = set()

    def visit(self, node: ast.AST) -> Any:
        new = super().visit(node)
        if isinstance(node, ast.Assign):
            site = self.concats.get((node.lineno, node.col_offset))
            if site is not None:
                return self._concat_step(node, site)
        if isinstance(node, ast.stmt):
            site = self.by_pos.get((node.lineno, node.col_offset))
            if site is not None and site.id not in self.wrapped:
                self.wrapped.add(site.id)
                token = f"__pdd_t_{site.id}"
                before = _stmts(f"{token} = __pdd__.enter({site.id}, locals(), globals())", node)
                after = _stmts(f"__pdd__.exit({site.id}, {token}, locals(), globals())", node)
                return [*before, new, *after]
        return new

    def _concat_step(self, node: ast.Assign, site: Site) -> ast.stmt:
        assert site.concat is not None
        value = node.value
        assert isinstance(value, ast.Call) and value.args
        objs = value.args[0]
        assert isinstance(objs, (ast.List, ast.Tuple)) and len(objs.elts) == 2
        part = ast.get_source_segment(self.source, objs.elts[1])
        if part is None:
            return node
        info = site.concat
        text = (
            f"{info.target} = __pdd__.concat_step({site.id}, "
            f"lambda __pdd_objs: {info.callee}(__pdd_objs{info.kwargs}), {info.target}, ({part}))"
        )
        return _stmts(text, node)[0]


def _with_marks(tree: ast.Module) -> None:
    body: list[ast.stmt] = []
    for i, stmt in enumerate(tree.body):
        is_docstring = (
            i == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        is_future = isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
        if not (is_docstring or is_future):
            body.extend(_stmts(f"__pdd__.mark({stmt.lineno})", stmt))
        body.append(stmt)
    if tree.body:
        body.extend(_stmts("__pdd__.mark(None)", tree.body[-1]))
    tree.body = body


def instrument(source: str, sites: list[Site], *, marks: bool = True) -> ast.Module:
    """The instrumented module, ready for `compile(tree, filename, "exec")`."""
    tree = ast.parse(source)
    tree = _Instrumenter(source, sites).visit(tree)
    assert isinstance(tree, ast.Module)
    if marks:
        _with_marks(tree)
    ast.fix_missing_locations(tree)
    return tree
