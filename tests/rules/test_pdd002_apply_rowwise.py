from __future__ import annotations

import pytest

from tests.helpers import findings


def suggestion(source: str) -> str | None:
    found = findings(source, "PDD002")
    assert len(found) == 1
    assert found[0].fix is None
    return found[0].suggestion


@pytest.mark.parametrize(
    "source",
    [
        "df.apply(f, axis=1)",
        "df.apply(f, axis='columns')",
        "df.apply(lambda r: r.a, axis=1, result_type='expand')",
    ],
)
def test_detected(source: str) -> None:
    assert len(findings(source, "PDD002")) == 1


@pytest.mark.parametrize(
    "source",
    ["df.apply(f)", "df.apply(f, axis=0)", "s.map(f)", "df.agg(f, axis=1)"],
)
def test_not_flagged(source: str) -> None:
    assert findings(source, "PDD002") == []


def test_lambda_arithmetic() -> None:
    assert (
        suggestion('df["t"] = df.apply(lambda r: r["price"] * r["qty"], axis=1)')
        == 'df["t"] = df["price"] * df["qty"]'
    )


def test_lambda_conditional_wraps_array_in_series() -> None:
    assert (
        suggestion("s = df.apply(lambda r: 'hi' if r['a'] > 1 else 'lo', axis=1)")
        == "s = pd.Series(np.where(df['a'] > 1, 'hi', 'lo'), index=df.index)"
    )


def test_named_function() -> None:
    assert (
        suggestion(
            """
            def margin(row):
                \"\"\"Profit per unit.\"\"\"
                return row["price"] - row["cost"]

            df["m"] = df.apply(margin, axis=1)
            """
        )
        == 'df["m"] = df["price"] - df["cost"]'
    )


@pytest.mark.parametrize(
    "source",
    [
        "df.apply(lambda r: helper(r['a']), axis=1)",
        "df.apply(lambda r: r['a'], axis=1, raw=True)",
        "df.apply(lambda r, k=2: r['a'] * k, axis=1)",
        "df[df.a > 0].apply(lambda r: r['a'], axis=1)",
        "df.apply(lambda r: r.sum(), axis=1)",
        """
        def f(row):
            x = row["a"]
            return x
        df.apply(f, axis=1)
        """,
    ],
)
def test_no_suggestion(source: str) -> None:
    assert suggestion(source) is None
