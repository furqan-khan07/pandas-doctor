from __future__ import annotations

import pytest

IPython = pytest.importorskip("IPython")


@pytest.fixture(scope="module")
def ip():  # type: ignore[no-untyped-def]
    from IPython.testing.globalipapp import start_ipython

    shell = start_ipython()
    shell.run_cell("%load_ext pandas_doctor")
    shell.run_cell(
        "import numpy as np, pandas as pd\n"
        "df = pd.DataFrame({'a': np.arange(500, dtype=float), 'b': np.arange(500) % 7})"
    )
    return shell


def test_cell_runs_in_notebook_namespace(ip, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    ip.run_cell(
        "%%doctor --budget 0.1\nfor i, r in df.iterrows():\n    df.at[i, 'c'] = r['a'] + r['b']"
    )
    out = capsys.readouterr().out
    assert "identical" in out and "verified on this data" in out
    assert "c" in ip.user_ns["df"].columns
    assert not [k for k in ip.user_ns if k.startswith("__pdd")]


def test_rewrite_flag_offers_new_cell(
    ip, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    offered: list[str] = []
    monkeypatch.setattr(ip, "set_next_input", lambda text, replace=False: offered.append(text))
    ip.run_cell(
        "%%doctor --rewrite --budget 0.1\nfor i, r in df.iterrows():\n    df.at[i, 'd'] = r['a'] * 2"
    )
    assert offered and 'df["d"] = df["a"] * 2' in offered[0].replace("'", '"')
    out = capsys.readouterr().out
    assert "in a new cell below" in out and "Use `%%doctor --rewrite`" not in out
