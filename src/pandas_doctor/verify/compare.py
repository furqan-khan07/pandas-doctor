"""Strict equivalence check for the values pandas code produces.

The same function backs the test suite, `pandas-doctor verify` and `pandas-doctor run`,
so whatever the tests prove is what the product claims.

Strict means: same Python type (``numpy.float64`` is not ``float``), same dtype, same
index and columns (labels, dtype and names), same Series names, missing values in the
same places, and exactly equal values. The one leniency is pandas' own: a RangeIndex and
an int64 Index with the same labels are treated as the same.

Two softer verdicts exist, and neither is ever called identical. ``ScalarTypesDiffer``
is for a Python scalar and a numpy scalar of the same kind holding exactly the same
value (``1.5`` vs ``numpy.float64(1.5)``). ``WithinTolerance`` is for floats, and only
when everything except the float values already matched.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd

from pandas_doctor.verify.verdict import (
    Different,
    Identical,
    ScalarTypesDiffer,
    Verdict,
    WithinTolerance,
    worst,
)

DEFAULT_RTOL = 1e-9

_IDENTICAL = Identical()


def compare(expected: Any, actual: Any, *, rtol: float = DEFAULT_RTOL) -> Verdict:
    """Compare what the original code produced (`expected`) with the rewrite's output."""
    return _compare(expected, actual, rtol, "")


def _type_name(obj: object) -> str:
    t = type(obj)
    return t.__name__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"


def _where(where: str) -> str | None:
    return where or None


# A Python scalar type and the numpy scalars that hold the same kind of value exactly.
_SCALAR_KINDS: tuple[tuple[type, type], ...] = (
    (bool, np.bool_),
    (int, np.integer),
    (float, np.floating),
    (complex, np.complexfloating),
    (str, np.str_),
    (bytes, np.bytes_),
)


def _same_kind_scalars(a: Any, b: Any) -> bool:
    """One is a builtin scalar, the other a numpy scalar of the same kind that converts to
    it exactly (so numpy.longdouble doesn't count)."""
    for builtin, numpy_type in _SCALAR_KINDS:
        for x, y in ((a, b), (b, a)):
            if type(x) is builtin and isinstance(y, numpy_type):
                return not isinstance(y, (np.floating, np.complexfloating)) or (
                    y.dtype.itemsize <= (8 if isinstance(y, np.floating) else 16)
                )
    return False


def _mixed_scalars(a: Any, b: Any, rtol: float, where: str) -> Verdict:
    if isinstance(a, (float, complex, np.floating, np.complexfloating)):
        values = _float_scalar(a, b, rtol, where)
    else:
        values = _plain(a, b, where)
    if isinstance(values, Different):
        return values
    return worst(values, ScalarTypesDiffer(((_type_name(a), _type_name(b)),), _where(where)))


def _compare(a: Any, b: Any, rtol: float, where: str) -> Verdict:
    if type(a) is not type(b):
        if _same_kind_scalars(a, b):
            return _mixed_scalars(a, b, rtol, where)
        return Different(f"type differs: {_type_name(a)} vs {_type_name(b)}", _where(where))
    if isinstance(a, pd.DataFrame):
        return _frame(a, b, rtol, where)
    if isinstance(a, pd.Series):
        return _series(a, b, rtol, where, check_index=True)
    if isinstance(a, pd.Index):
        return _index(a, b, where or "index")
    if isinstance(a, np.ndarray):
        return _ndarray(a, b, rtol, where)
    if isinstance(a, pd.api.extensions.ExtensionArray):
        return _series(pd.Series(a), pd.Series(b), rtol, where, check_index=False)
    if isinstance(a, (list, tuple)):
        return _sequence(a, b, rtol, where)
    if isinstance(a, dict):
        return _mapping(a, b, rtol, where)
    if isinstance(a, (float, complex, np.floating, np.complexfloating)):
        return _float_scalar(a, b, rtol, where)
    if a is pd.NA or a is pd.NaT or a is None:
        return _IDENTICAL  # singletons, and the types already matched
    if (
        isinstance(a, (pd.Timestamp, dt.datetime))
        and a.tzinfo != b.tzinfo
        and str(a.tzinfo) != str(b.tzinfo)
    ):
        return Different(f"timezone differs: {a.tzinfo} vs {b.tzinfo}", _where(where))
    return _plain(a, b, where)


def _plain(a: Any, b: Any, where: str) -> Verdict:
    try:
        equal = a == b
    except Exception as exc:  # comparing exotic objects can raise
        return Different(f"cannot compare {_type_name(a)} values ({exc})", _where(where))
    if isinstance(equal, (bool, np.bool_)):
        return _IDENTICAL if equal else Different(f"{a!r} vs {b!r}", _where(where))
    return Different(f"cannot compare {_type_name(a)} values", _where(where))


def _float_scalar(a: Any, b: Any, rtol: float, where: str) -> Verdict:
    if isinstance(a, (complex, np.complexfloating)):
        parts_a, parts_b = [a.real, a.imag], [b.real, b.imag]
    else:
        parts_a, parts_b = [a], [b]
    verdict: Verdict = _IDENTICAL
    for x, y in zip(parts_a, parts_b, strict=True):
        xf, yf = float(x), float(y)
        if math.isnan(xf) and math.isnan(yf):
            continue
        if xf == yf:
            continue
        if math.isnan(xf) or math.isnan(yf) or math.isinf(xf) or math.isinf(yf):
            return Different(f"{a!r} vs {b!r}", _where(where))
        abs_diff = abs(xf - yf)
        rel = abs_diff / max(abs(xf), abs(yf))
        if rel > rtol:
            return Different(f"{a!r} vs {b!r} (relative difference {rel:.1e})", _where(where))
        verdict = worst(verdict, WithinTolerance(abs_diff, rel, rtol, _where(where)))
    return verdict


def _sequence(a: list[Any] | tuple[Any, ...], b: Any, rtol: float, where: str) -> Verdict:
    if len(a) != len(b):
        return Different(f"length differs: {len(a)} vs {len(b)}", _where(where))
    verdict: Verdict = _IDENTICAL
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        verdict = worst(verdict, _compare(x, y, rtol, f"{where}[{i}]"))
        if isinstance(verdict, Different):
            return verdict
    return verdict


def _mapping(a: dict[Any, Any], b: dict[Any, Any], rtol: float, where: str) -> Verdict:
    keys_a, keys_b = list(a), list(b)
    if set(keys_a) != set(keys_b):
        missing = [k for k in keys_a if k not in b][:3]
        extra = [k for k in keys_b if k not in a][:3]
        return Different(f"keys differ: missing {missing}, extra {extra}", _where(where))
    if keys_a != keys_b:
        return Different("same keys in a different order", _where(where))
    verdict: Verdict = _IDENTICAL
    for key in keys_a:
        verdict = worst(verdict, _compare(a[key], b[key], rtol, f"{where}[{key!r}]"))
        if isinstance(verdict, Different):
            return verdict
    return verdict


def _names_equal(x: object, y: object) -> bool:
    if x is None or y is None:
        return x is y
    try:
        if pd.isna(x) and pd.isna(y):
            return type(x) is type(y)
    except (TypeError, ValueError):
        pass
    try:
        return bool(x == y) and type(x) is type(y)
    except Exception:
        return False


def _index_types_match(a: pd.Index, b: pd.Index) -> bool:
    """RangeIndex and an int64 Index holding the same labels count as the same, like
    pandas' own `check_index_type="equiv"`: they behave the same everywhere that matters."""
    if type(a) is type(b):
        return True
    int_like = (pd.RangeIndex, pd.Index)
    return (
        isinstance(a, int_like)
        and isinstance(b, int_like)
        and {type(a), type(b)} == {pd.RangeIndex, pd.Index}
        and a.dtype == b.dtype
    )


def _index(a: pd.Index, b: pd.Index, where: str) -> Verdict:
    if not _index_types_match(a, b):
        return Different(f"index type differs: {_type_name(a)} vs {_type_name(b)}", where)
    if a.dtype != b.dtype:
        return Different(f"index dtype differs: {a.dtype} vs {b.dtype}", where)
    if len(a) != len(b):
        return Different(f"index length differs: {len(a)} vs {len(b)}", where)
    if list(a.names) != list(b.names) and not all(
        _names_equal(x, y) for x, y in zip(a.names, b.names, strict=False)
    ):
        return Different(f"index names differ: {list(a.names)} vs {list(b.names)}", where)
    if not a.equals(b):
        for pos, (x, y) in enumerate(zip(a, b, strict=True)):
            if not _names_equal(x, y):
                return Different(f"labels differ at position {pos}: {x!r} vs {y!r}", where)
        return Different("labels differ", where)
    if isinstance(a, pd.DatetimeIndex) and a.freq != b.freq:
        return Different(f"index freq differs: {a.freq} vs {b.freq}", where)
    return _IDENTICAL


def _frame(a: pd.DataFrame, b: pd.DataFrame, rtol: float, where: str) -> Verdict:
    if a.shape != b.shape:
        return Different(f"shape differs: {a.shape} vs {b.shape}", _where(where))
    for verdict in (
        _index(a.columns, b.columns, f"{where}.columns".lstrip(".")),
        _index(a.index, b.index, f"{where}.index".lstrip(".")),
    ):
        if isinstance(verdict, Different):
            return verdict
    if a.flags.allows_duplicate_labels != b.flags.allows_duplicate_labels:
        return Different("allows_duplicate_labels flag differs", _where(where))
    total: Verdict = _IDENTICAL
    for pos, label in enumerate(a.columns):
        col_where = f"{where}[{label!r}]" if where else f"column {label!r}"
        verdict = _series(a.iloc[:, pos], b.iloc[:, pos], rtol, col_where, check_index=False)
        total = worst(total, verdict)
        if isinstance(total, Different):
            return total
    return total


def _is_float_dtype(dtype: Any) -> bool:
    return bool(pd.api.types.is_float_dtype(dtype) or pd.api.types.is_complex_dtype(dtype))


def _series(a: pd.Series, b: pd.Series, rtol: float, where: str, *, check_index: bool) -> Verdict:
    label = where or "series"
    if len(a) != len(b):
        return Different(f"length differs: {len(a)} vs {len(b)}", label)
    if check_index:
        verdict = _index(a.index, b.index, f"{label}.index")
        if isinstance(verdict, Different):
            return verdict
        if not _names_equal(a.name, b.name):
            return Different(f"name differs: {a.name!r} vs {b.name!r}", label)
    if a.dtype != b.dtype:
        return Different(f"dtype differs: {a.dtype} vs {b.dtype}", label)
    try:
        pd.testing.assert_series_equal(
            a, b, check_exact=True, check_names=False, check_index=False, check_freq=False
        )
        return _object_types(a, b, label) if a.dtype == object else _IDENTICAL
    except AssertionError:
        pass
    if _is_float_dtype(a.dtype):
        return _float_values(a, b, rtol, label)
    return _first_value_difference(a, b, label)


def _object_types(a: pd.Series, b: pd.Series, where: str) -> Verdict:
    """Equal object columns can still hold different Python types (1.0 vs numpy.float64)."""
    types_a, types_b = list(map(type, a.array)), list(map(type, b.array))
    if types_a == types_b:
        return _IDENTICAL
    pairs: dict[tuple[str, str], None] = {}
    for pos, (ta, tb) in enumerate(zip(types_a, types_b, strict=True)):
        if ta is tb:
            continue
        x, y = a.iloc[pos], b.iloc[pos]
        if not _same_kind_scalars(x, y):
            return Different(
                f"type differs at row {a.index[pos]!r}: {_type_name(x)} vs {_type_name(y)}", where
            )
        pairs.setdefault((_type_name(x), _type_name(y)))
    return ScalarTypesDiffer(tuple(pairs), where)


def _first_value_difference(a: pd.Series, b: pd.Series, where: str) -> Different:
    for pos in range(len(a)):
        x, y = a.iloc[pos], b.iloc[pos]
        both_missing = bool(pd.isna(x)) and bool(pd.isna(y)) if np.ndim(x) == 0 else False
        if both_missing:
            if type(x) is not type(y):
                return Different(
                    f"missing values differ at row {a.index[pos]!r}: {x!r} vs {y!r}", where
                )
            continue
        try:
            equal = bool(x == y)
        except Exception:
            equal = False
        if not equal:
            return Different(f"values differ at row {a.index[pos]!r}: {x!r} vs {y!r}", where)
    return Different("values differ", where)


def _float_values(a: pd.Series, b: pd.Series, rtol: float, where: str) -> Verdict:
    complex_ = pd.api.types.is_complex_dtype(a.dtype)
    target = np.complex128 if complex_ else np.float64
    xa = a.to_numpy(dtype=target, na_value=np.nan)
    xb = b.to_numpy(dtype=target, na_value=np.nan)
    if complex_:
        xa = np.concatenate([xa.real, xa.imag])
        xb = np.concatenate([xb.real, xb.imag])
    nan_a, nan_b = np.isnan(xa), np.isnan(xb)
    if not np.array_equal(nan_a, nan_b):
        pos = int(np.argmax(nan_a != nan_b)) % len(a)
        return Different(f"missing values in different rows, first at row {a.index[pos]!r}", where)
    ok = ~nan_a
    fa, fb = xa[ok], xb[ok]
    inf = np.isinf(fa) | np.isinf(fb)
    if inf.any() and not np.array_equal(fa[inf], fb[inf]):
        return Different("infinite values differ", where)
    fa, fb = fa[~inf], fb[~inf]
    diff = np.abs(fa - fb)
    scale = np.maximum(np.abs(fa), np.abs(fb))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(diff == 0, 0.0, diff / scale)
    max_rel = float(rel.max()) if rel.size else 0.0
    max_abs = float(diff.max()) if diff.size else 0.0
    if max_rel == 0.0:
        # Equal as floats but assert_series_equal still failed, so something else differs.
        return _first_value_difference(a, b, where)
    if max_rel > rtol:
        idx = np.flatnonzero(ok)[~inf][int(np.argmax(rel))] % len(a)
        return Different(
            f"values differ at row {a.index[idx]!r}: {a.iloc[idx]!r} vs {b.iloc[idx]!r} "
            f"(relative difference {max_rel:.1e})",
            where,
        )
    return WithinTolerance(max_abs, max_rel, rtol, where)


def _ndarray(a: np.ndarray, b: np.ndarray, rtol: float, where: str) -> Verdict:
    label = where or "array"
    if a.shape != b.shape:
        return Different(f"shape differs: {a.shape} vs {b.shape}", label)
    if a.dtype != b.dtype:
        return Different(f"dtype differs: {a.dtype} vs {b.dtype}", label)
    return _series(
        pd.Series(a.reshape(-1)), pd.Series(b.reshape(-1)), rtol, label, check_index=False
    )
