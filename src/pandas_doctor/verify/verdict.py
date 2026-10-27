"""The four answers an equivalence check can give.

These words appear in reports, so they have to stay honest:

* ``Identical``: every value, dtype, index, name and Python type matched exactly.
* ``ScalarTypesDiffer``: every value matched exactly, but some scalars have a different
  Python type of the same kind, e.g. ``numpy.float64`` where the original had ``float``.
  Never called identical; the types are always shown.
* ``WithinTolerance``: same structure, but some float values differ by at most ``rtol``
  (typically last-bit rounding from summing in a different order). The largest
  difference is always shown, and so are any scalar types that differ as well.
* ``Different``: anything else, with the first place it differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeAlias

# (original's type, rewrite's type), each pair listed once
TypePairs: TypeAlias = tuple[tuple[str, str], ...]


def _pairs_text(pairs: TypePairs) -> str:
    return "; ".join(f"{a} vs {b}" for a, b in pairs)


def _merge_pairs(a: TypePairs, b: TypePairs) -> TypePairs:
    return a + tuple(p for p in b if p not in a)


@dataclass(frozen=True)
class Identical:
    kind: ClassVar[str] = "identical"

    def describe(self) -> str:
        return "identical"

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "description": self.describe()}


@dataclass(frozen=True)
class ScalarTypesDiffer:
    types: TypePairs
    where: str | None = None
    kind: ClassVar[str] = "scalar_types"

    def describe(self) -> str:
        text = f"equal values, different scalar types ({_pairs_text(self.types)})"
        return f"{text} in {self.where}" if self.where else text

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "description": self.describe(),
            "scalar_types": [list(p) for p in self.types],
            "where": self.where,
        }


@dataclass(frozen=True)
class WithinTolerance:
    max_abs_diff: float
    max_rel_diff: float
    rtol: float
    where: str | None = None
    scalar_types: TypePairs = ()
    kind: ClassVar[str] = "within_tolerance"

    def describe(self) -> str:
        text = (
            f"equal within rtol={self.rtol:g} (largest relative difference {self.max_rel_diff:.1e})"
        )
        if self.where:
            text += f" in {self.where}"
        if self.scalar_types:
            text += f", and different scalar types ({_pairs_text(self.scalar_types)})"
        return text

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "description": self.describe(),
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "rtol": self.rtol,
            "scalar_types": [list(p) for p in self.scalar_types],
            "where": self.where,
        }


@dataclass(frozen=True)
class Different:
    reason: str
    where: str | None = None
    kind: ClassVar[str] = "different"

    def describe(self) -> str:
        return f"different: {self.reason}" + (f" (in {self.where})" if self.where else "")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "description": self.describe(),
            "reason": self.reason,
            "where": self.where,
        }


Verdict: TypeAlias = Identical | ScalarTypesDiffer | WithinTolerance | Different


def _scalar_types(v: Verdict) -> TypePairs:
    if isinstance(v, ScalarTypesDiffer):
        return v.types
    if isinstance(v, WithinTolerance):
        return v.scalar_types
    return ()


def worst(a: Verdict, b: Verdict) -> Verdict:
    """Combine two verdicts: Different beats WithinTolerance beats ScalarTypesDiffer beats
    Identical. Scalar type differences are kept when tolerance differences join them."""
    if isinstance(a, Different):
        return a
    if isinstance(b, Different):
        return b
    types = _merge_pairs(_scalar_types(a), _scalar_types(b))
    if isinstance(a, WithinTolerance) and isinstance(b, WithinTolerance):
        keep = a if a.max_rel_diff >= b.max_rel_diff else b
        return WithinTolerance(
            max_abs_diff=max(a.max_abs_diff, b.max_abs_diff),
            max_rel_diff=max(a.max_rel_diff, b.max_rel_diff),
            rtol=a.rtol,
            where=keep.where,
            scalar_types=types,
        )
    for v in (a, b):
        if isinstance(v, WithinTolerance):
            return WithinTolerance(
                v.max_abs_diff, v.max_rel_diff, v.rtol, v.where, scalar_types=types
            )
    for v in (a, b):
        if isinstance(v, ScalarTypesDiffer):
            return ScalarTypesDiffer(types, v.where)
    return a


def accepted(verdict: Verdict, *, tolerance: bool = False, strict_types: bool = False) -> bool:
    """Whether a rewrite with this verdict may be recommended.

    Identical always is. Equal values with different scalar types are too, unless
    `strict_types`; the report says so either way. Float differences within the
    tolerance need `tolerance`.
    """
    if isinstance(verdict, Identical):
        return True
    if isinstance(verdict, ScalarTypesDiffer):
        return not strict_types
    if isinstance(verdict, WithinTolerance):
        return tolerance and not (strict_types and verdict.scalar_types)
    return False


def preference(verdict: Verdict) -> int:
    """Lower is better, for picking between equally plain rewrites."""
    return {"identical": 0, "scalar_types": 1, "within_tolerance": 2}.get(verdict.kind, 3)
