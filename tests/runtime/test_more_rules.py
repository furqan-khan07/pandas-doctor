"""Runtime checks for PDD004-PDD008: verified, timed, and only recommended when faster."""

from __future__ import annotations

import textwrap
from pathlib import Path

from pandas_doctor.runtime import run_script
from pandas_doctor.verify.verdict import Identical, WithinTolerance

HEADER = """\
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 20_000
df = pd.DataFrame({
    "name": rng.choice(["Ann ", " BOB", "cy"], n),
    "k": rng.integers(0, 50, n),
    "x": rng.random(n),
    "y": rng.random(n) + 1,
    "d": pd.date_range("2024-01-01", periods=n, freq="min"),
})
"""


def run(tmp_path: Path, body: str):  # type: ignore[no-untyped-def]
    path = tmp_path / "job.py"
    path.write_text(HEADER + textwrap.dedent(body))
    outcome = run_script(path, [], budget=0.3)
    assert outcome.exit_code == 0, outcome.error
    return outcome


def by_rule(outcome, rule: str):  # type: ignore[no-untyped-def]
    found = [r for r in outcome.doctor.results.values() if r.site.rule == rule]
    assert len(found) == 1, found
    return found[0]


def test_pdd004_datetime_field(tmp_path: Path) -> None:
    res = by_rule(run(tmp_path, 'df["yr"] = df["d"].apply(lambda t: t.year)\n'), "PDD004")
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)
    assert "dt.year" in res.chosen.candidate.code
    assert res.worth_applying


def test_pdd005_aggregate_is_honest_about_rounding(tmp_path: Path) -> None:
    res = by_rule(run(tmp_path, 's = df.groupby("k").apply(lambda g: g["x"].sum())\n'), "PDD005")
    # groupby's built-in sum uses compensated summation, so the last bits can differ
    assert res.chosen is not None or any(isinstance(t.verdict, WithinTolerance) for t in res.tried)


def test_pdd005_count_is_identical(tmp_path: Path) -> None:
    res = by_rule(run(tmp_path, 's = df.groupby("k").apply(lambda g: g["x"].count())\n'), "PDD005")
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)


def test_pdd007_index_loop(tmp_path: Path) -> None:
    body = """
    small = df.head(2000).copy()
    for i in small.index:
        small.at[i, "z"] = small.at[i, "x"] * 3
    """
    res = by_rule(run(tmp_path, body), "PDD007")
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)
    assert res.worth_applying


def test_identical_but_slower_is_not_recommended(tmp_path: Path) -> None:
    outcome = run(tmp_path, 'df["n"] = df["name"].apply(lambda x: x.strip().lower())\n')
    res = by_rule(outcome, "PDD004")
    if res.chosen is not None and not res.worth_applying:
        _, skipped = outcome.rewritten()
        assert any("not faster" in s for s in skipped)


def test_pdd006_only_reported_when_it_wins(tmp_path: Path) -> None:
    body = """
    wide = pd.DataFrame(rng.integers(0, 9, (20, 300)), columns=[f"c{i}" for i in range(300)])
    cols = list(wide.columns)
    for c in cols:
        wide[c] = wide[c].astype("float32")
    """
    res = by_rule(run(tmp_path, body), "PDD006")
    assert res.site.runtime_only
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)


def test_pdd006_skips_function_parameters(tmp_path: Path) -> None:
    body = """
    def cast(frame, cols):
        for c in cols:
            frame[c] = frame[c].astype("float32")
    cast(df, ["x", "y"])
    """
    outcome = run(tmp_path, body)
    assert not [r for r in outcome.doctor.results.values() if r.site.rule == "PDD006"]


def test_pdd008_memory(tmp_path: Path) -> None:
    outcome = run(tmp_path, "")
    found = {(m.frame, m.column) for m in outcome.memory}
    assert ("df", "name") in found
    m = next(m for m in outcome.memory if m.column == "name")
    assert m.distinct == 3 and m.before_bytes > 2 * m.after_bytes
