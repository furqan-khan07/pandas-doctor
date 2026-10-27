"""Equivalence checking and timing: the proof behind every claim pandas-doctor prints."""

from pandas_doctor.verify.compare import DEFAULT_RTOL, compare
from pandas_doctor.verify.verdict import (
    Different,
    Identical,
    ScalarTypesDiffer,
    Verdict,
    WithinTolerance,
)

__all__ = [
    "DEFAULT_RTOL",
    "Different",
    "Identical",
    "ScalarTypesDiffer",
    "Verdict",
    "WithinTolerance",
    "compare",
]
