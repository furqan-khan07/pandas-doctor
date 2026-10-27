"""The comparator decides every "identical" pandas-doctor prints, so it gets the most tests.

Half of these check that real differences are caught: a comparator that says "identical"
too easily would make the whole tool dishonest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_doctor.verify import Different, Identical, WithinTolerance, compare


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": [1, 2, 3],
            "f": [0.5, np.nan, 2.5],
            "s": ["a", "b", None],
            "c": pd.Categorical(["x", "y", "x"]),
            "t": pd.to_datetime(["2024-01-01", "2024-03-10", None]).tz_localize("UTC"),
        },
        index=pd.Index([10, 20, 30], name="id"),
    )


class TestIdentical:
    def test_frame_copy(self) -> None:
        assert compare(frame(), frame().copy()) == Identical()

    def test_nan_in_same_places(self) -> None:
        a = pd.Series([1.0, np.nan, 3.0])
        assert compare(a, a.copy()) == Identical()

    @pytest.mark.parametrize(
        "value",
        [1, 1.5, "x", None, True, np.int64(3), np.float64(2.5), np.bool_(True), pd.NA, pd.NaT],
    )
    def test_scalars(self, value: object) -> None:
        assert compare(value, value) == Identical()

    def test_nested_containers(self) -> None:
        a = {"x": [np.float64(1.0), 2], "y": (pd.Series([1, 2]), "z")}
        b = {"x": [np.float64(1.0), 2], "y": (pd.Series([1, 2]), "z")}
        assert compare(a, b) == Identical()

    def test_arrays_with_nan(self) -> None:
        assert compare(np.array([1.0, np.nan]), np.array([1.0, np.nan])) == Identical()

    def test_object_series_with_none(self) -> None:
        a = pd.Series(["x", None], dtype=object)
        assert compare(a, a.copy()) == Identical()


class TestDifferent:
    def test_iterrows_style_upcast_is_caught(self) -> None:
        # what iterrows does to an int column when the frame also has floats
        df = frame()[["i", "f"]]
        upcast = df.astype({"i": "float64"})
        verdict = compare(df, upcast)
        assert isinstance(verdict, Different)
        assert "dtype differs: int64 vs float64" in verdict.reason

    def test_numpy_scalar_vs_python_scalar(self) -> None:
        verdict = compare([np.float64(1.0)], [1.0])
        assert isinstance(verdict, Different)
        assert "numpy.float64 vs float" in verdict.reason

    def test_int_vs_numpy_int(self) -> None:
        assert isinstance(compare(3, np.int64(3)), Different)

    def test_series_name(self) -> None:
        verdict = compare(pd.Series([1, 2], name="a"), pd.Series([1, 2]))
        assert isinstance(verdict, Different) and "name differs" in verdict.reason

    def test_index_labels(self) -> None:
        verdict = compare(pd.Series([1, 2], index=[0, 1]), pd.Series([1, 2], index=[0, 2]))
        assert isinstance(verdict, Different) and "labels differ" in verdict.reason

    def test_range_index_matches_equal_int_index(self) -> None:
        # pandas treats these as equivalent too (check_index_type="equiv")
        assert compare(pd.Series([1, 2]), pd.Series([1, 2], index=[0, 1])) == Identical()

    def test_index_dtype(self) -> None:
        a = pd.Series([1, 2], index=pd.Index([0, 1], dtype="int64"))
        b = pd.Series([1, 2], index=pd.Index([0, 1], dtype="int32"))
        assert isinstance(compare(a, b), Different)

    def test_index_name(self) -> None:
        assert isinstance(compare(frame(), frame().rename_axis("other")), Different)

    def test_column_order(self) -> None:
        df = frame()
        assert isinstance(compare(df, df[list(reversed(df.columns))]), Different)

    def test_missing_value_moved(self) -> None:
        verdict = compare(pd.Series([1.0, np.nan]), pd.Series([np.nan, 1.0]))
        assert isinstance(verdict, Different) and "missing values" in verdict.reason

    def test_nan_vs_value(self) -> None:
        assert isinstance(compare(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0])), Different)

    def test_categories_differ(self) -> None:
        a = pd.Series(pd.Categorical(["x"], categories=["x", "y"]))
        b = pd.Series(pd.Categorical(["x"], categories=["x"]))
        assert isinstance(compare(a, b), Different)

    def test_timezone(self) -> None:
        a = pd.Timestamp("2024-01-01", tz="UTC")
        b = pd.Timestamp("2024-01-01 01:00", tz="Europe/Paris")
        assert isinstance(compare(a, b), Different)

    def test_dict_key_order(self) -> None:
        assert isinstance(compare({"a": 1, "b": 2}, {"b": 2, "a": 1}), Different)

    def test_list_length(self) -> None:
        assert isinstance(compare([1, 2], [1, 2, 3]), Different)

    def test_frame_vs_series(self) -> None:
        df = frame()
        assert isinstance(compare(df[["i"]], df["i"]), Different)

    def test_big_float_difference(self) -> None:
        verdict = compare(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.001]))
        assert isinstance(verdict, Different) and "relative difference" in verdict.reason

    def test_object_values(self) -> None:
        verdict = compare(pd.Series(["a", "b"]), pd.Series(["a", "c"]))
        assert isinstance(verdict, Different)


class TestWithinTolerance:
    def test_summation_order(self) -> None:
        values = np.random.default_rng(0).random(10_000)
        loop_total = 0.0
        for v in values:
            loop_total += v
        verdict = compare(np.float64(loop_total), values.sum())
        # sequential vs pairwise summation: equal up to rounding, never silently "identical"
        assert isinstance(verdict, (Identical, WithinTolerance))
        if isinstance(verdict, WithinTolerance):
            assert 0 < verdict.max_rel_diff < 1e-12

    def test_series_rounding(self) -> None:
        verdict = compare(pd.Series([0.1 + 0.2]), pd.Series([0.3]))
        assert isinstance(verdict, WithinTolerance)
        assert verdict.max_rel_diff < 1e-15

    def test_rtol_zero_makes_it_different(self) -> None:
        assert isinstance(compare(pd.Series([0.1 + 0.2]), pd.Series([0.3]), rtol=0), Different)

    def test_tolerance_needs_matching_structure(self) -> None:
        # a float rounding difference does not excuse a dtype difference
        a = pd.Series([0.1 + 0.2], dtype="float64")
        b = pd.Series([0.3], dtype="float32")
        assert isinstance(compare(a, b), Different)

    def test_inf_must_match_exactly(self) -> None:
        assert isinstance(compare(pd.Series([np.inf]), pd.Series([-np.inf])), Different)
