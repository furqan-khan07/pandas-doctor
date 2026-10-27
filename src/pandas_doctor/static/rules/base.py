"""What a rule is, and the context it runs in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import libcst as cst
from libcst.metadata import (
    ParentNodeProvider,
    PositionProvider,
    ProviderT,
    QualifiedNameProvider,
    ScopeProvider,
)

from pandas_doctor.findings import Finding, Fix, Severity


@dataclass
class FileContext:
    path: str
    module: cst.Module  # the tree the rule visits; fixes must target nodes of this tree


class Rule(cst.CSTVisitor):
    """One check. Subclasses visit the tree and call `report()`."""

    code: ClassVar[str]
    name: ClassVar[str]
    summary: ClassVar[str]
    severity: ClassVar[Severity] = Severity.WARNING
    # "safe" or "unsafe" when some findings come with a static fix (see findings.Fix).
    fix_safety: ClassVar[str | None] = None

    METADATA_DEPENDENCIES: ClassVar[tuple[ProviderT, ...]] = (
        PositionProvider,
        QualifiedNameProvider,
        ScopeProvider,
        ParentNodeProvider,
    )

    def __init__(self, ctx: FileContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.findings: list[Finding] = []

    def parent(self, node: cst.CSTNode) -> cst.CSTNode | None:
        try:
            return self.get_metadata(ParentNodeProvider, node)
        except KeyError:
            return None

    def ancestors(self, node: cst.CSTNode) -> list[cst.CSTNode]:
        out: list[cst.CSTNode] = []
        current = self.parent(node)
        while current is not None:
            out.append(current)
            current = self.parent(current)
        return out

    def report(
        self,
        node: cst.CSTNode,
        message: str,
        *,
        suggestion: str | None = None,
        note: str | None = None,
        fix: Fix | None = None,
        fix_skipped: str | None = None,
    ) -> None:
        pos = self.get_metadata(PositionProvider, node)
        self.findings.append(
            Finding(
                code=self.code,
                message=message,
                path=self.ctx.path,
                line=pos.start.line,
                column=pos.start.column + 1,
                end_line=pos.end.line,
                end_column=pos.end.column + 1,
                severity=self.severity,
                suggestion=suggestion,
                note=note,
                fix=fix,
                fix_skipped=fix_skipped,
            )
        )
