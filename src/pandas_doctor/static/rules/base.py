"""What a rule is, and the context it runs in."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import libcst as cst
from libcst.metadata import (
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    ProviderT,
    QualifiedNameProvider,
)

from pandas_doctor.findings import Finding, Fix, Severity


@dataclass
class FileContext:
    """One file being checked. Metadata is computed lazily, only if a rule asks for it:
    most files have nothing to report and never need positions or parent links."""

    path: str
    module: cst.Module  # the tree the rule visits; fixes must target nodes of this tree
    wrapper: MetadataWrapper | None = None
    _cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    def resolve(self, provider: ProviderT) -> Mapping[cst.CSTNode, Any]:
        if provider not in self._cache:
            if self.wrapper is None:
                self.wrapper = MetadataWrapper(self.module, unsafe_skip_copy=True)
            self._cache[provider] = self.wrapper.resolve(provider)
        return self._cache[provider]  # type: ignore[no-any-return]

    def position(self, node: cst.CSTNode) -> CodeRange:
        return self.resolve(PositionProvider)[node]  # type: ignore[no-any-return]

    def qualified_names(self) -> Mapping[cst.CSTNode, Any]:
        return self.resolve(QualifiedNameProvider)


class Rule(cst.BatchableCSTVisitor):
    """One check. Subclasses visit the tree and call `report()`.

    All rules run in a single batched traversal. `triggers` lists substrings the source
    must contain for the rule to be worth running (empty: always run), which lets the
    engine skip most files without parsing them.
    """

    code: ClassVar[str]
    name: ClassVar[str]
    summary: ClassVar[str]
    severity: ClassVar[Severity] = Severity.WARNING
    # "safe" or "unsafe" when some findings come with a static fix (see findings.Fix).
    fix_safety: ClassVar[str | None] = None

    triggers: ClassVar[tuple[str, ...]] = ()

    def __init__(self, ctx: FileContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.findings: list[Finding] = []

    def parent(self, node: cst.CSTNode) -> cst.CSTNode | None:
        return self.ctx.resolve(ParentNodeProvider).get(node)

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
        pos = self.ctx.position(node)
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
