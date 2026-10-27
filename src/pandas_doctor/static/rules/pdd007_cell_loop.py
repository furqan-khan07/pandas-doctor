"""PDD007: writing one cell at a time in a loop (`df.at[i, "c"] = ...`, `.loc`, `.iat`, `.iloc`).

Each cell write goes through pandas' indexing machinery, so a loop over the index is
thousands of times slower than assigning the whole column. For the common shape

    for i in df.index:
        df.at[i, "c"] = df.at[i, "a"] * 2

the rule prints `df["c"] = df["a"] * 2`. Loops over `iterrows()` are PDD001's job.
Not applied automatically: the column's resulting dtype can differ (a column created
cell by cell starts as float NaN). `pandas-doctor run` checks it on your data.
"""

from __future__ import annotations

import libcst as cst

from pandas_doctor.static.analysis import (
    body_statements,
    code,
    mentions,
    small_statements,
    target_names,
    walk,
)
from pandas_doctor.static.rules.base import Rule
from pandas_doctor.static.rules.pdd001_iterrows import assigned_names, iterrows_frame
from pandas_doctor.static.translate import Untranslatable, translate

NOTE = (
    "Not applied automatically: a column built cell by cell can end up with a different dtype. "
    "`pandas-doctor run` checks it on your data."
)
_CELL_ACCESSORS = {"at", "loc", "iat", "iloc"}
ROW = "__pdd_row"


def _cell(node: cst.BaseExpression, var: str) -> tuple[str, str, str] | None:
    """`df.at[i, "c"]` -> ("df", "at", '"c"') when the row part is exactly the loop variable."""
    if not (
        isinstance(node, cst.Subscript)
        and isinstance(node.value, cst.Attribute)
        and node.value.attr.value in _CELL_ACCESSORS
        and isinstance(node.value.value, cst.Name)
        and len(node.slice) == 2
    ):
        return None
    first, second = (el.slice for el in node.slice)
    if not (
        isinstance(first, cst.Index)
        and isinstance(first.value, cst.Name)
        and first.value.value == var
    ):
        return None
    if not (isinstance(second, cst.Index) and isinstance(second.value, cst.SimpleString)):
        return None
    return node.value.value.value, node.value.attr.value, code(second.value)


class _ToRow(cst.CSTTransformer):
    """Reads `df.at[i, "a"]` become `__pdd_row["a"]` so the row translator can take over."""

    def __init__(self, frame: str, var: str) -> None:
        super().__init__()
        self.frame = frame
        self.var = var

    def leave_Subscript(
        self, original_node: cst.Subscript, updated_node: cst.Subscript
    ) -> cst.BaseExpression:
        cell = _cell(original_node, self.var)
        if cell is not None and cell[0] == self.frame and cell[1] in {"at", "loc"}:
            return cst.parse_expression(f"{ROW}[{cell[2]}]")
        return updated_node


def _index_loop_var(node: cst.For) -> tuple[str, str] | None:
    """`for i in df.index:` -> ("i", "df")."""
    names = target_names(node.target)
    if names is None or len(names) != 1:
        return None
    it = node.iter
    if (
        isinstance(it, cst.Attribute)
        and it.attr.value == "index"
        and isinstance(it.value, cst.Name)
    ):
        return names[0], it.value.value
    return None


def cell_loop_rewrite(node: cst.For) -> tuple[str, list[str]] | None:
    """Column-wise statement for `for i in df.index: df.at[i, "c"] = <expr of df.at[i, ...]>`."""
    found = _index_loop_var(node)
    stmts = body_statements(node.body)
    if found is None or len(stmts) != 1 or node.orelse is not None:
        return None
    var, frame = found
    smalls = small_statements(stmts[0])
    if len(smalls) != 1 or not isinstance(smalls[0], cst.Assign) or len(smalls[0].targets) != 1:
        return None
    assign = smalls[0]
    cell = _cell(assign.targets[0].target, var)
    if cell is None or cell[0] != frame or cell[1] not in {"at", "loc"}:
        return None
    value = assign.value.visit(_ToRow(frame, var))
    assert isinstance(value, cst.BaseExpression)
    blocked = frozenset(assigned_names(node.body) | {var})
    if mentions(value, var):
        return None
    try:
        t = translate(value, ROW, frame, blocked=blocked)
    except Untranslatable:
        return None
    return f"{frame}[{cell[2]}] = {t.code}", [frame]


def _writes_cell(node: cst.BaseExpression, var: str) -> bool:
    """`X.at/.loc/.iat/.iloc[<uses var>, <anything>] = ...` (detection only)."""
    if not (
        isinstance(node, cst.Subscript)
        and isinstance(node.value, cst.Attribute)
        and node.value.attr.value in _CELL_ACCESSORS
        and len(node.slice) == 2
    ):
        return False
    first = node.slice[0].slice
    return isinstance(first, cst.Index) and mentions(first.value, var) > 0


class CellLoop(Rule):
    code = "PDD007"
    name = "cell-by-cell-loop"
    summary = "df.at/.loc/.iat/.iloc[i, col] = ... inside a loop; assign the whole column"

    def visit_For(self, node: cst.For) -> None:
        if iterrows_frame(node.iter) is not None:
            return  # PDD001 covers iterrows loops
        names = target_names(node.target) or []
        if len(names) != 1:
            return
        var = names[0]
        for n in walk(node.body):
            if isinstance(n, cst.AssignTarget) and _writes_cell(n.target, var):
                found = cell_loop_rewrite(node)
                self.report(
                    node,
                    "writes one cell per iteration through `.at`/`.loc`/`.iat`/`.iloc`",
                    suggestion=found[0] if found else None,
                    note=NOTE,
                )
                return
