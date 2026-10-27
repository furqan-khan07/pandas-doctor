from __future__ import annotations

import pytest

from pandas_doctor.findings import Severity
from tests.helpers import findings


def suggestion(source: str) -> str | None:
    found = findings(source, "PDD001")
    assert len(found) == 1
    assert found[0].fix is None  # never auto-fixed
    return found[0].suggestion


class TestDetection:
    def test_for_loop(self) -> None:
        found = findings(
            """
            for i, row in df.iterrows():
                print(row)
            """,
            "PDD001",
        )
        assert [(f.line, f.code) for f in found] == [(2, "PDD001")]
        assert "df.iterrows()" in found[0].message

    def test_list_comprehension(self) -> None:
        assert len(findings("x = [r['a'] for _, r in df.iterrows()]", "PDD001")) == 1

    def test_attribute_frame(self) -> None:
        assert len(findings("for _, r in self.data.iterrows(): pass", "PDD001")) == 1

    @pytest.mark.parametrize(
        "source",
        [
            "for x in df.itertuples(): pass",
            "for x in rows(): pass",
            "for k, v in d.items(): pass",
            "it = df.iterrows()",  # no loop, nothing to vectorise here
        ],
    )
    def test_not_flagged(self, source: str) -> None:
        assert findings(source, "PDD001") == []


class TestSuggestions:
    def test_column_assignment(self) -> None:
        assert (
            suggestion(
                """
                for i, row in df.iterrows():
                    df.at[i, "revenue"] = row["price"] * row["qty"]
                """
            )
            == 'df["revenue"] = df["price"] * df["qty"]'
        )

    def test_loc_assignment_and_attribute_access(self) -> None:
        assert (
            suggestion(
                """
                for idx, r in df.iterrows():
                    df.loc[idx, "x"] = r.a + 1
                """
            )
            == 'df["x"] = df["a"] + 1'
        )

    def test_running_total(self) -> None:
        assert (
            suggestion(
                """
                for _, row in df.iterrows():
                    total += row["a"] * 2
                """
            )
            == 'total += (df["a"] * 2).sum()'
        )

    def test_conditional_count(self) -> None:
        assert (
            suggestion(
                """
                for _, row in df.iterrows():
                    if row["a"] > 0:
                        n += 1
                """
            )
            == 'n += int((df["a"] > 0).sum())'
        )

    def test_list_append(self) -> None:
        assert (
            suggestion(
                """
                for _, row in df.iterrows():
                    out.append(row["name"].strip().lower())
                """
            )
            == 'out.extend(df["name"].str.strip().str.lower().tolist())'
        )

    def test_filtered_append(self) -> None:
        assert (
            suggestion(
                """
                for _, row in df.iterrows():
                    if row["a"] > 1 and row["b"] < 2:
                        out.append(row["a"])
                """
            )
            == 'out.extend(df["a"][(df["a"] > 1) & (df["b"] < 2)].tolist())'
        )

    def test_if_else_cell(self) -> None:
        assert (
            suggestion(
                """
                for i, row in df.iterrows():
                    if row["qty"] > 3:
                        df.at[i, "size"] = "big"
                    else:
                        df.at[i, "size"] = "small"
                """
            )
            == 'df["size"] = np.where(df["qty"] > 3, "big", "small")'
        )

    def test_comprehension(self) -> None:
        found = findings("names = [r['n'].upper() for _, r in df.iterrows()]", "PDD001")
        assert found[0].suggestion == "df['n'].str.upper().tolist()"

    @pytest.mark.parametrize(
        "source",
        [
            # calls a user function per row
            """
            for i, row in df.iterrows():
                df.at[i, "x"] = score(row)
            """,
            # uses the index label
            """
            for i, row in df.iterrows():
                out.append(row["a"] * i)
            """,
            # more than one statement
            """
            for i, row in df.iterrows():
                v = row["a"]
                out.append(v)
            """,
            # writes to a different frame
            """
            for i, row in df.iterrows():
                other.at[i, "x"] = row["a"]
            """,
            # `in` has no direct column form
            """
            for _, row in df.iterrows():
                if row["a"] in (1, 2):
                    n += 1
            """,
            # frame is an expression; evaluating it twice could differ
            """
            for _, row in load().iterrows():
                out.append(row["a"])
            """,
            # a name that changes in the loop
            """
            for _, row in df.iterrows():
                total += row["a"] * weight
                weight = weight * 2
            """,
        ],
    )
    def test_no_suggestion_when_not_one_to_one(self, source: str) -> None:
        assert suggestion(source) is None

    def test_note_explains_why_not_automatic(self) -> None:
        found = findings("for i, r in df.iterrows():\n    df.at[i, 'b'] = r['a'] + 1", "PDD001")
        assert found[0].note and "ints into floats" in found[0].note


class TestSeverity:
    """Loops with a vectorised rewrite are warnings; loops without one are info."""

    @pytest.mark.parametrize(
        "source",
        [
            "for i, r in df.iterrows():\n    df.at[i, 'b'] = r['a'] + 1",
            "x = [r['a'] * 2 for _, r in df.iterrows()]",
        ],
    )
    def test_shape_match_is_a_warning(self, source: str) -> None:
        (found,) = findings(source, "PDD001")
        assert found.suggestion is not None
        assert found.severity is Severity.WARNING

    @pytest.mark.parametrize(
        "source",
        [
            # per-row side effects, the usual case in real notebooks
            """
            for _, row in df.iterrows():
                folium.Marker([row["lat"], row["lon"]]).add_to(m)
            """,
            """
            for i, row in df.iterrows():
                print(i, row["name"])
            """,
            "x = [score(r) for _, r in df.iterrows()]",
        ],
    )
    def test_no_shape_is_info(self, source: str) -> None:
        (found,) = findings(source, "PDD001")
        assert found.suggestion is None
        assert found.severity is Severity.INFO
        assert found.note and "doesn't fail" in found.note
        assert found.to_dict()["severity"] == "info"
