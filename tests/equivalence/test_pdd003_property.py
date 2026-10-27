"""Property tests for PDD003's rewrite: repeated concat vs one concat.

Hypothesis builds lists of frames that share columns and dtypes, with empty frames,
duplicate and non-default indexes, NaN/NA values, categoricals and tz-aware datetimes.
For each list we run the loop as written and the exact code `check --fix --unsafe-fixes`
produces, and require the comparator to say Identical.

When the parts have *different* columns the rewrite is not identical (see the regression
tests at the bottom). That's why the fix is marked unsafe and `run` checks it on real data.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pandas_doctor.static.engine import check_source
from pandas_doctor.verify import Different, Identical, compare
from tests.helpers import fixed

KINDS = ["int", "float", "Int64", "bool", "str", "category", "datetime"]
CATEGORIES = ["x", "y", "z"]


def _column(kind: str, n: int, draw: st.DrawFn) -> pd.Series:
    if kind == "int":
        return pd.Series(draw(st.lists(st.integers(-5, 5), min_size=n, max_size=n)), dtype="int64")
    if kind == "float":
        vals = draw(
            st.lists(st.one_of(st.floats(-1e3, 1e3), st.just(np.nan)), min_size=n, max_size=n)
        )
        return pd.Series(vals, dtype="float64")
    if kind == "Int64":
        vals = draw(st.lists(st.one_of(st.integers(-5, 5), st.none()), min_size=n, max_size=n))
        return pd.Series(vals, dtype="Int64")
    if kind == "bool":
        return pd.Series(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype="bool")
    if kind == "str":
        vals = draw(
            st.lists(st.one_of(st.sampled_from(["x", "y", ""]), st.none()), min_size=n, max_size=n)
        )
        return pd.Series(vals, dtype=object)
    if kind == "category":
        vals = draw(st.lists(st.sampled_from(CATEGORIES), min_size=n, max_size=n))
        return pd.Series(pd.Categorical(vals, categories=CATEGORIES))
    if kind == "datetime":
        vals = draw(st.lists(st.integers(0, 10**6), min_size=n, max_size=n))
        return pd.Series(pd.to_datetime(vals, unit="s", utc=True))
    raise AssertionError(kind)


@st.composite
def schemas(draw: st.DrawFn) -> list[tuple[str, str]]:
    cols = draw(st.lists(st.sampled_from(["a", "b", "c"]), unique=True, min_size=1, max_size=3))
    return [(c, draw(st.sampled_from(KINDS))) for c in cols]


@st.composite
def frame_with(draw: st.DrawFn, schema: list[tuple[str, str]]) -> pd.DataFrame:
    n = draw(st.integers(0, 4))
    data = {
        c: _column(kind, n, draw).to_numpy()
        if kind in {"int", "float", "bool"}
        else _column(kind, n, draw).array
        for c, kind in schema
    }
    index_kind = draw(st.sampled_from(["range", "ints", "strings"]))
    index: pd.Index | None = None
    if index_kind == "ints":
        index = pd.Index(draw(st.lists(st.integers(0, 3), min_size=n, max_size=n)), dtype="int64")
    elif index_kind == "strings":
        index = pd.Index(
            draw(st.lists(st.sampled_from(["p", "q"]), min_size=n, max_size=n)), dtype=object
        )
    return pd.DataFrame(data, index=index)


@st.composite
def same_schema_parts(draw: st.DrawFn) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    schema = draw(schemas())
    parts = draw(st.lists(frame_with(schema), max_size=5))
    start = draw(st.one_of(st.just(pd.DataFrame()), frame_with(schema)))
    # same columns and dtypes throughout; the index dtype may vary
    return start, parts


LOOP = """\
import pandas as pd
out = start
for p in parts:
    out = pd.concat([out, p]{kwargs})
"""


def _execute(code: str, start: pd.DataFrame, parts: list[pd.DataFrame]) -> tuple[str, Any]:
    ns: dict[str, Any] = {"start": start.copy(), "parts": [p.copy() for p in parts]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            exec(code, ns)
        except Exception as exc:  # both versions must fail the same way
            return "error", type(exc)
    return "ok", ns["out"]


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=same_schema_parts(),
    kwargs=st.sampled_from(
        ["", ", ignore_index=True", ", ignore_index=False", ", sort=False", ", axis=0"]
    ),
)
def test_same_columns_and_dtypes_give_identical_frame(
    data: tuple[pd.DataFrame, list[pd.DataFrame]], kwargs: str
) -> None:
    start, parts = data
    original = LOOP.format(kwargs=kwargs)
    rewritten = fixed(original)
    assert rewritten != original
    status_a, result_a = _execute(original, start, parts)
    status_b, result_b = _execute(rewritten, start, parts)
    assert status_a == status_b
    if status_a == "ok":
        assert compare(result_a, result_b) == Identical(), rewritten
    else:
        assert result_a == result_b


def test_fix_is_marked_unsafe_with_the_condition() -> None:
    finding = next(f for f in check_source(LOOP.format(kwargs="")).findings if f.code == "PDD003")
    assert finding.fix is not None and not finding.fix.safe
    assert finding.fix.caveat and "same columns and dtypes" in finding.fix.caveat
    # plain --fix leaves it alone
    assert fixed(LOOP.format(kwargs=""), unsafe=False) == LOOP.format(kwargs="")


# Found by an earlier, wider version of the property test: with parts that have different
# columns, filling gaps one concat at a time keeps NaT / <NA>, while one concat writes NaN.
COUNTEREXAMPLES = [
    [
        pd.DataFrame({"a": pd.to_datetime([], utc=True)}),
        pd.DataFrame(index=[0]),
        pd.DataFrame({"a": pd.Series([], dtype=object)}),
    ],
    [
        pd.DataFrame({"a": pd.array([], dtype="Int64")}),
        pd.DataFrame(index=[0]),
        pd.DataFrame({"a": pd.Series([], dtype=bool)}),
    ],
]


def test_different_columns_can_differ_and_the_comparator_says_so() -> None:
    original = LOOP.format(kwargs="")
    rewritten = fixed(original)
    verdicts = []
    for parts in COUNTEREXAMPLES:
        _, a = _execute(original, pd.DataFrame(), parts)
        _, b = _execute(rewritten, pd.DataFrame(), parts)
        verdicts.append(compare(a, b))
    # at least one of these differs on every supported pandas version we've seen;
    # whichever does, the comparator must not call it identical
    assert any(isinstance(v, Different) for v in verdicts), verdicts
