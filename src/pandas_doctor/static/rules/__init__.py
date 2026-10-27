"""The rule registry."""

from __future__ import annotations

from pandas_doctor.static.rules.base import FileContext, Rule
from pandas_doctor.static.rules.pdd001_iterrows import IterrowsLoop
from pandas_doctor.static.rules.pdd002_apply_rowwise import RowwiseApply
from pandas_doctor.static.rules.pdd003_concat_in_loop import ConcatInLoop

ALL_RULES: tuple[type[Rule], ...] = (
    IterrowsLoop,
    RowwiseApply,
    ConcatInLoop,
)

BY_CODE: dict[str, type[Rule]] = {rule.code: rule for rule in ALL_RULES}

__all__ = ["ALL_RULES", "BY_CODE", "FileContext", "Rule"]
