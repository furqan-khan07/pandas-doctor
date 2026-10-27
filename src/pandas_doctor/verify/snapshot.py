"""Copies of the inputs a block reads, so the original and the rewrite start from the same state."""

from __future__ import annotations

import copy
import types
from typing import Any

import numpy as np
import pandas as pd

# Things that are passed through by reference: they aren't data the block can change.
_SHARED = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)


class NotCopyable(Exception):
    pass


def snapshot(value: Any) -> Any:
    """A deep copy that the rewrite can modify without touching the user's objects."""
    if isinstance(value, _SHARED) or value is None:
        return value
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        return value.copy(deep=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (int, float, complex, str, bytes, bool, np.generic)):
        return value
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise NotCopyable(f"can't copy a {type(value).__name__}: {exc}") from exc


def snapshot_all(values: dict[str, Any]) -> dict[str, Any]:
    return {name: snapshot(v) for name, v in values.items()}


def nbytes(value: Any) -> int:
    """Rough memory size of the data in a value (shallow for pandas objects)."""
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=False).sum())
    if isinstance(value, (pd.Series, pd.Index)):
        return int(value.memory_usage(deep=False))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple, dict, set)):
        return 64 * len(value)
    return 64
