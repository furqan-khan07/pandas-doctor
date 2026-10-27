"""The three answers an equivalence check can give.

These words appear in reports, so they have to stay honest:

* ``Identical``: every value, dtype, index and name matched exactly.
* ``WithinTolerance``: same structure, but some float values differ by at most ``rtol``
  (typically last-bit rounding from summing in a different order). The largest
  difference is always shown.
* ``Different``: anything else, with the first place it differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Identical:
    def describe(self) -> str:
        return "identical"


@dataclass(frozen=True)
class WithinTolerance:
    max_abs_diff: float
    max_rel_diff: float
    rtol: float
    where: str | None = None

    def describe(self) -> str:
        text = (
            f"equal within rtol={self.rtol:g} (largest relative difference {self.max_rel_diff:.1e})"
        )
        return f"{text} in {self.where}" if self.where else text


@dataclass(frozen=True)
class Different:
    reason: str
    where: str | None = None

    def describe(self) -> str:
        return f"different: {self.reason}" + (f" (in {self.where})" if self.where else "")


Verdict: TypeAlias = Identical | WithinTolerance | Different


def worst(a: Verdict, b: Verdict) -> Verdict:
    """Combine two verdicts: Different beats WithinTolerance beats Identical."""
    if isinstance(a, Different):
        return a
    if isinstance(b, Different):
        return b
    if isinstance(a, WithinTolerance) and isinstance(b, WithinTolerance):
        keep = a if a.max_rel_diff >= b.max_rel_diff else b
        return WithinTolerance(
            max_abs_diff=max(a.max_abs_diff, b.max_abs_diff),
            max_rel_diff=max(a.max_rel_diff, b.max_rel_diff),
            rtol=a.rtol,
            where=keep.where,
        )
    if isinstance(a, WithinTolerance):
        return a
    return b
