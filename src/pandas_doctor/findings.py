"""The data types every rule and reporter share."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TypeAlias

import libcst as cst

# A fix swaps one node for a new node, for several statements, or for nothing.
Replacement: TypeAlias = cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Fix:
    """A rewrite the static checker can apply without running the code.

    `safe` fixes keep the result identical for every input, so `check --fix` applies
    them. Unsafe fixes are identical only under a condition the checker can't see
    from the code (stated in `caveat`); they need `--unsafe-fixes`, and `pandas-doctor
    run` can check them on real data instead.
    `target` is a node of the parsed module; the engine replaces it by identity.
    """

    target: cst.CSTNode
    replacement: Replacement
    description: str
    safe: bool = True
    caveat: str | None = None


@dataclass
class Finding:
    """One problem found in one file."""

    code: str
    message: str
    path: str
    line: int
    column: int  # 1-based, like most editors
    end_line: int
    end_column: int
    severity: Severity
    suggestion: str | None = None
    note: str | None = None
    fix: Fix | None = field(default=None, repr=False, compare=False)
    # Why the finding has no automatic fix, when it could plausibly have one.
    fix_skipped: str | None = None
    # Set by the notebook reader: which cell the finding is in.
    cell: int | None = None

    @property
    def fixable(self) -> bool:
        return self.fix is not None

    def location(self) -> str:
        if self.cell is not None:
            return f"{self.path}:cell {self.cell}:{self.line}:{self.column}"
        return f"{self.path}:{self.line}:{self.column}"

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "cell": self.cell,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "severity": self.severity.value,
            "suggestion": self.suggestion,
            "note": self.note,
            "fixable": self.fixable,
            "fix_skipped": self.fix_skipped,
        }
