"""Turn verified rewrites into a patch for the user's source."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider

from pandas_doctor.runtime.doctor import SiteResult
from pandas_doctor.runtime.sites import SiteMap
from pandas_doctor.static.analysis import walk


class _Replace(cst.CSTTransformer):
    def __init__(
        self, replacements: dict[int, cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]]
    ) -> None:
        super().__init__()
        self.replacements = replacements

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode):  # type: ignore[no-untyped-def]
        return self.replacements.get(id(original_node), updated_node)


def _is_binding(name: cst.Name, parents: Mapping[cst.CSTNode, cst.CSTNode]) -> bool:
    """Whether this occurrence (re)binds the name rather than reading it."""
    child: cst.CSTNode = name
    parent = parents.get(child)
    while isinstance(parent, (cst.Tuple, cst.List, cst.Element, cst.StarredElement)):
        child, parent = parent, parents.get(parent)
    if isinstance(parent, cst.AssignTarget):
        return True
    if isinstance(parent, (cst.For, cst.CompFor)) and parent.target is child:
        return True
    if isinstance(parent, cst.AnnAssign) and parent.target is child and parent.value is not None:
        return True
    return (
        isinstance(parent, (cst.AsName, cst.NamedExpr))
        and getattr(parent, "name", getattr(parent, "target", None)) is child
    )


def _loop_vars_used_after(module: cst.Module, node: cst.CSTNode, names: list[str]) -> str | None:
    """A loop variable the code reads after the loop before rebinding it.

    The rewrite removes the loop, so those variables would no longer be set.
    """
    names = [n for n in names if n != "_"]
    if not names:
        return None
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    positions = wrapper.resolve(PositionProvider)
    parents = wrapper.resolve(ParentNodeProvider)
    end = positions[node].end
    later = sorted(
        (
            (positions[n].start.line, positions[n].start.column, n)
            for n in walk(module)
            if isinstance(n, cst.Name) and n.value in names and n in positions
        ),
        key=lambda item: (item[0], item[1]),
    )
    pending = set(names)
    for line, col, n in later:
        if (line, col) <= (end.line, end.column) or n.value not in pending:
            continue
        if _is_binding(n, parents):
            pending.discard(n.value)  # rebound before any read: the old value is dead
        else:
            return n.value
    return None


def _needs_numpy_import(module: cst.Module) -> bool:
    for stmt in module.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if isinstance(small, cst.Import):
                    for alias in small.names:
                        bound = alias.asname.name if alias.asname else alias.name
                        if isinstance(bound, cst.Name) and bound.value == "np":
                            return False
    return True


def _add_numpy_import(module: cst.Module) -> cst.Module:
    body = list(module.body)
    insert_at = 0
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine) and all(
            isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
        ):
            insert_at = i + 1
        elif (
            i == 0
            and isinstance(stmt, cst.SimpleStatementLine)
            and isinstance(stmt.body[0], cst.Expr)
        ):
            insert_at = 1  # after a docstring
    body.insert(insert_at, cst.parse_statement("import numpy as np\n"))
    return module.with_changes(body=body)


def apply_rewrites(sitemap: SiteMap, results: Iterable[SiteResult]) -> tuple[str, list[str]]:
    """New source with every verified rewrite applied, and notes on the ones skipped."""
    module = sitemap.module
    replacements: dict[int, cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]] = {}
    skipped: list[str] = []
    numpy_needed = False
    for res in results:
        site = res.site
        if res.chosen is None:
            continue
        if not res.worth_applying:
            skipped.append(f"line {site.line}: identical, but not faster on this data")
            continue
        if site.concat is not None:
            if site.fix is not None and isinstance(site.fix.replacement, cst.FlattenSentinel):
                replacements[id(site.node)] = site.fix.replacement
            continue
        used = _loop_vars_used_after(module, site.node, site.loop_vars)
        if used:
            skipped.append(
                f"line {site.line}: `{used}` is used after the loop, so the loop can't be removed"
            )
            continue
        new_body = cst.parse_module(
            res.chosen.candidate.code + "\n", config=module.config_for_parsing
        ).body
        if not new_body:
            continue
        first = new_body[0].with_changes(leading_lines=site.node.leading_lines)  # type: ignore[attr-defined]
        replacements[id(site.node)] = cst.FlattenSentinel([first, *new_body[1:]])
        numpy_needed |= res.chosen.candidate.uses_numpy
    if not replacements:
        return module.code, skipped
    new_module = module.visit(_Replace(replacements))
    assert isinstance(new_module, cst.Module)
    if numpy_needed and _needs_numpy_import(new_module):
        new_module = _add_numpy_import(new_module)
    code = new_module.code
    compile(code, "<rewritten>", "exec")
    return code, skipped
