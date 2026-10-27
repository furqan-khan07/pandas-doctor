from __future__ import annotations

import pytest

from tests.helpers import findings


def one(source: str, code: str) -> str | None:
    found = findings(source, code)
    assert len(found) == 1, found
    assert found[0].fix is None
    return found[0].suggestion


class TestPDD004:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('df["n"].apply(lambda x: x.strip().lower())', 'df["n"].str.strip().str.lower()'),
            ("df.name.apply(lambda x: len(x))", "df.name.str.len()"),
            ('df["d"].apply(lambda t: t.year)', 'df["d"].dt.year'),
            ("df.price.map(lambda v: v * 2)", "df.price * 2"),
            ("s.apply(lambda x: x.upper())", "s.str.upper()"),
            (
                "s.map(lambda x: 'a' if x > 0 else 'b')",
                "pd.Series(np.where(s > 0, 'a', 'b'), index=s.index)",
            ),
        ],
    )
    def test_suggestions(self, source: str, expected: str) -> None:
        assert one(source, "PDD004") == expected

    @pytest.mark.parametrize(
        "source",
        [
            "df.apply(lambda col: col * 2)",  # DataFrame.apply passes whole columns
            "df.apply(lambda col: col.replace(0, 1))",  # Series has .replace too
            "s.apply(lambda x: len(x))",  # could be a frame: len(column)
            "df.groupby('k').apply(lambda g: len(g))",  # groups, not elements
            "df['a'].rolling(3).apply(lambda w: w.sum())",
            "df['a'].apply(lambda x: helper(x))",  # nothing to vectorise
            "df['a'].apply(lambda x, k=1: x + k)",
            "df['a'].apply(str.lower)",
            "df['a'].str.apply(lambda x: x)",
        ],
    )
    def test_not_flagged(self, source: str) -> None:
        assert findings(source, "PDD004") == []


class TestPDD005:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('df.groupby("k").apply(lambda g: g["x"].sum())', 'df.groupby("k")["x"].sum()'),
            (
                'df.groupby("k").apply(lambda g: g["x"].sum() / g["y"].sum())',
                'df.groupby("k")["x"].sum() / df.groupby("k")["y"].sum()',
            ),
            (
                'df.groupby(["a", "b"]).apply(lambda g: g.x.mean())',
                'df.groupby(["a", "b"])["x"].mean()',
            ),
        ],
    )
    def test_suggestions(self, source: str, expected: str) -> None:
        assert one(source, "PDD005") == expected

    @pytest.mark.parametrize(
        "source",
        [
            'df.groupby("k").apply(lambda g: g.sort_values("x").head(1))',
            'df.groupby("k").apply(lambda g: g["x"].sum() + len(other))',
            'df.groupby("k").apply(lambda g: g["x"].quantile(0.9))',
            'df.groupby("k").apply(f)',
            'df.apply(lambda g: g["x"].sum())',
        ],
    )
    def test_not_flagged(self, source: str) -> None:
        assert findings(source, "PDD005") == []


class TestPDD007:
    def test_index_loop_with_at(self) -> None:
        assert (
            one(
                """
                for i in df.index:
                    df.at[i, "c"] = df.at[i, "a"] * 2 + df.loc[i, "b"]
                """,
                "PDD007",
            )
            == 'df["c"] = df["a"] * 2 + df["b"]'
        )

    def test_positional_writes_are_flagged_without_suggestion(self) -> None:
        assert one("for i in range(len(df)):\n    df.iloc[i, 2] = 0", "PDD007") is None

    def test_other_frame_no_suggestion(self) -> None:
        assert one("for i in df.index:\n    df.loc[i, 'c'] = other.loc[i, 'a']", "PDD007") is None

    @pytest.mark.parametrize(
        "source",
        [
            "for i, r in df.iterrows():\n    df.at[i, 'c'] = 1",  # that's PDD001
            "for i in df.index:\n    x = df.at[i, 'a']",  # reads only
            "for k in keys:\n    df.at[0, k] = 1",  # loop var isn't the row
        ],
    )
    def test_not_flagged(self, source: str) -> None:
        assert findings(source, "PDD007") == []
