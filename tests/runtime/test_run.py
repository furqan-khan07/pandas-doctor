"""End-to-end tests for `pandas-doctor run`: real scripts, real data, real comparisons."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pandas_doctor.cli import main
from pandas_doctor.runtime import run_script
from pandas_doctor.verify.verdict import Different, Identical, WithinTolerance

HEADER = """\
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "price": rng.uniform(1, 100, 400).round(2),
    "qty": rng.integers(1, 9, 400),
    "name": rng.choice(["Ann ", " bob", "Cy"], 400),
})
"""


def script(tmp_path: Path, body: str, name: str = "job.py") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + textwrap.dedent(body))
    return path


def only_result(path: Path, **kw):  # type: ignore[no-untyped-def]
    outcome = run_script(path, [], budget=0.2, **kw)
    assert outcome.exit_code == 0, outcome.error
    results = list(outcome.doctor.results.values())
    assert len(results) == 1, results
    return outcome, results[0]


def test_cell_write_verified_identical(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        for i, row in df.iterrows():
            df.at[i, "total"] = row["price"] * row["qty"]
        """,
    )
    _, res = only_result(path)
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)
    assert res.original is not None and res.chosen.timing is not None
    assert res.chosen.timing.seconds < res.original.seconds


def test_list_append_numeric_frame_needs_numpy_scalars(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        num = df[["price", "qty"]]
        out = []
        for _, row in num.iterrows():
            out.append(row["price"] * 2)
        """,
    )
    _, res = only_result(path)
    # all-numeric rows hold numpy.float64; `.tolist()` gives Python floats: caught, not hidden
    assert isinstance(res.tried[0].verdict, Different)
    assert "numpy.float64 vs float" in res.tried[0].verdict.reason
    assert res.chosen is not None and "np.asarray" in res.chosen.candidate.code


def test_list_append_mixed_frame_plain_is_identical(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        out = []
        for _, row in df.iterrows():
            out.append(row["price"] * 2)
        """,
    )
    _, res = only_result(path)
    # with a string column, iterrows rows are object dtype and hold Python floats
    assert res.chosen is res.tried[0] and isinstance(res.chosen.verdict, Identical)


def test_running_total_exact_or_tolerance(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        total = 0.0
        for _, row in df.iterrows():
            total += row["price"] * row["qty"]
        """,
    )
    _, res = only_result(path)
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)


def test_tolerance_accepted_only_when_asked(tmp_path: Path) -> None:
    body = """
        vals = df["price"].to_numpy() / 7
        s = pd.Series(vals)
        frame = pd.DataFrame({"v": vals})
        total = 0.0
        for _, row in frame.iterrows():
            total += row["v"]
    """
    path = script(tmp_path, body)
    _, strict = only_result(path)
    tried = [type(r.verdict) for r in strict.tried]
    assert WithinTolerance in tried or strict.chosen is not None
    _, loose = only_result(path, rtol=1e-9, accept_tolerance=True)
    assert loose.chosen is not None


def test_apply_rowwise(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        df["size"] = df.apply(lambda r: "big" if r["qty"] > 4 else "small", axis=1)
        """,
    )
    _, res = only_result(path)
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)


def test_apply_upcast_needs_faithful_rewrite(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        s = df[["price", "qty"]].apply(lambda r: r["qty"] * 2, axis=1)
        """,
    )
    outcome = run_script(path, [], budget=0.2)
    # the frame slice isn't a plain name, so this site isn't checked at all
    assert not outcome.doctor.results

    path = script(
        tmp_path,
        """
        sub = df[["price", "qty"]]
        s = sub.apply(lambda r: r["qty"] * 2, axis=1)
        """,
        "job2.py",
    )
    _, res = only_result(path)
    # qty is int, but apply sees float rows: the plain rewrite gives int64
    assert isinstance(res.tried[0].verdict, Different)
    assert res.chosen is not None and "astype" in res.chosen.candidate.code


def test_concat_loop(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        out = pd.DataFrame()
        for k in range(30):
            part = df.iloc[[k]].copy()
            out = pd.concat([out, part], ignore_index=True)
        """,
    )
    _, res = only_result(path)
    assert res.concat_iterations == 30
    assert res.chosen is not None and isinstance(res.chosen.verdict, Identical)


def test_concat_with_different_columns_is_not_claimed(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        parts = [
            pd.DataFrame({"a": pd.to_datetime([], utc=True)}),
            pd.DataFrame(index=[0]),
            pd.DataFrame({"a": pd.Series([], dtype=object)}),
        ]
        out = pd.DataFrame()
        for p in parts:
            out = pd.concat([out, p])
        """,
    )
    _, res = only_result(path)
    assert res.chosen is None
    assert isinstance(res.tried[0].verdict, Different)


def test_function_site_checked_once_timed_always(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        def add_total(frame):
            for i, row in frame.iterrows():
                frame.at[i, "t"] = row["price"] + 1
            return frame

        for _ in range(3):
            add_total(df.copy())
        """,
    )
    _, res = only_result(path)
    assert res.calls == 3
    assert res.chosen is not None


def test_user_program_output_is_unchanged(tmp_path: Path) -> None:
    body = """
        out = []
        for _, row in df.iterrows():
            out.append(row["name"].strip().lower())
        print(sorted(set(out)))
    """
    path = script(tmp_path, body)
    plain = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=True)
    doctored = subprocess.run(
        [sys.executable, "-m", "pandas_doctor", "run", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert doctored.stdout == plain.stdout


def test_script_error_keeps_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = script(tmp_path, "raise SystemExit(3)\n")
    assert main(["run", str(path)]) == 3
    path = script(tmp_path, "1 / 0\n", "boom.py")
    assert main(["run", str(path)]) == 1
    assert "ZeroDivisionError" in capsys.readouterr().err


def test_fix_rewrites_and_result_is_the_same(tmp_path: Path) -> None:
    body = """
        for i, row in df.iterrows():
            df.at[i, "total"] = row["price"] * row["qty"]
        size = df.apply(lambda r: "big" if r["qty"] > 4 else "small", axis=1)
        out = pd.DataFrame()
        for k in range(5):
            out = pd.concat([out, df.iloc[[k]].copy()])
        print(df["total"].sum(), size.value_counts().to_dict(), out.shape)
    """
    path = script(tmp_path, body)
    before = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=True
    ).stdout
    assert main(["run", "--fix", "--budget", "0.2", str(path)]) == 0
    text = path.read_text()
    assert "iterrows" not in text and "axis=1" not in text and "out_parts" in text
    after = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=True
    ).stdout
    assert after == before


def test_loop_variable_used_later_blocks_the_rewrite(tmp_path: Path) -> None:
    path = script(
        tmp_path,
        """
        for i, row in df.iterrows():
            df.at[i, "t"] = row["price"] * 2
        print(i)
        """,
    )
    outcome, res = only_result(path)
    assert res.chosen is not None
    new_source, skipped = outcome.rewritten()
    assert new_source == outcome.source
    assert skipped and "`i` is used after the loop" in skipped[0]


def test_json_report(tmp_path: Path) -> None:
    import json

    path = script(
        tmp_path,
        """
        for i, row in df.iterrows():
            df.at[i, "t"] = row["price"] * 2
        """,
    )
    out = tmp_path / "report.json"
    assert main(["run", "--json", str(out), "--budget", "0.2", str(path)]) == 0
    data = json.loads(out.read_text())
    assert data["scope"] == "verified on this run's data only"
    assert data["sites"][0]["verified"] is True
    assert data["sites"][0]["verdict"] == "identical"
