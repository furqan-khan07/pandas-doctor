from __future__ import annotations

import json

import pytest

from pandas_doctor.cli import main
from pandas_doctor.verify.api import default_outputs, verify_code
from pandas_doctor.verify.purity import Impure, check_pure, check_setup
from pandas_doctor.verify.verdict import Different, Identical, WithinTolerance

SETUP = """
import numpy as np
import pandas as pd
rng = np.random.default_rng(1)
df = pd.DataFrame({"x": rng.random(500), "k": rng.integers(0, 5, 500)})
"""


def test_identical_rewrite() -> None:
    r = verify_code(
        "df['y'] = df.apply(lambda r: r['x'] * 2, axis=1)",
        "df['y'] = df['x'] * 2",
        setup=SETUP,
        budget=0.2,
    )
    assert r.verdict == Identical()
    assert r.speedup is not None and r.speedup > 1
    assert "verified on this data" in r.describe()


def test_wrong_rewrite_is_caught() -> None:
    r = verify_code("df['y'] = df['x'] * 2", "df['y'] = df['x'] * 3", setup=SETUP, budget=0.1)
    assert isinstance(r.verdict, Different)


def test_float_rounding_reported_as_tolerance() -> None:
    r = verify_code(
        "t = 0.0\nfor v in df['x']:\n    t += v",
        "t = df['x'].sum()",
        setup=SETUP,
        budget=0.1,
    )
    assert isinstance(r.verdict, (WithinTolerance, Different))  # float vs numpy.float64 too


def test_missing_output() -> None:
    r = verify_code("a = df['x'].sum()", "b = df['x'].sum()", setup=SETUP, budget=0.1)
    assert isinstance(r.verdict, Different) and "doesn't define `a`" in r.verdict.reason


def test_refuses_io_and_randomness() -> None:
    for code in [
        "df.to_csv('x.csv')",
        "import os\nos.remove('x')",
        "x = np.random.rand(3)",
        "open('f').read()",
        "pd.set_option('display.width', 10)",
    ]:
        r = verify_code(code, "x = 1", setup=SETUP)
        assert r.refused is not None, code


def test_setup_may_read_and_use_seeded_random() -> None:
    check_setup("import pandas as pd\ndf = pd.read_csv('a.csv')\nimport random\nrandom.seed(1)")
    with pytest.raises(Impure):
        check_setup("import subprocess")
    with pytest.raises(Impure):
        check_pure("df = pd.read_csv('a.csv')")


def test_default_outputs() -> None:
    src = "out = []\nfor i, r in df.iterrows():\n    out.append(r['x'])\n    df.loc[i, 'y'] = 1\ntotal += 1"
    assert set(default_outputs(src)) == {"out", "df", "total"}


def test_cli_verify(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "setup.py").write_text(SETUP)
    (tmp_path / "orig.py").write_text("s = df.groupby('k')['x'].apply(lambda g: g.sum())")
    (tmp_path / "cand.py").write_text("s = df.groupby('k')['x'].sum()")
    code = main(
        [
            "verify",
            "--setup",
            str(tmp_path / "setup.py"),
            "--original",
            str(tmp_path / "orig.py"),
            "--candidate",
            str(tmp_path / "cand.py"),
            "--budget",
            "0.1",
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["scope"] == "verified on the given inputs only"
    assert code in (0, 1)
    assert data["verdict"] in {"identical", "scalar_types", "within_tolerance", "different"}
    assert data["details"]["kind"] == data["verdict"]


SCALAR_ORIGINAL = "out = []\nfor _, r in df.iterrows():\n    out.append(r['a'] * 2)"
SCALAR_CANDIDATE = "out = (df['a'] * 2).tolist()"
SCALAR_SETUP = "import pandas as pd\ndf = pd.DataFrame({'a': [1.0, 2.0, 3.0]})"


def test_verify_scalar_types() -> None:
    r = verify_code(SCALAR_ORIGINAL, SCALAR_CANDIDATE, setup=SCALAR_SETUP, budget=0.1)
    assert r.verdict.kind == "scalar_types"
    assert "equal values, different scalar types (numpy.float64 vs float)" in r.describe()
    data = r.to_dict()
    assert data["verdict"] == "scalar_types"
    assert data["details"]["scalar_types"] == [["numpy.float64", "float"]]


@pytest.mark.parametrize(("extra", "expected"), [([], 0), (["--strict-types"], 1)])
def test_cli_verify_scalar_types_exit_code(
    capsys: pytest.CaptureFixture[str], extra: list[str], expected: int
) -> None:
    code = main(
        [
            "verify",
            "--setup-code",
            SCALAR_SETUP,
            "--original-code",
            SCALAR_ORIGINAL,
            "--candidate-code",
            SCALAR_CANDIDATE,
            "--budget",
            "0.1",
            *extra,
        ]
    )
    assert code == expected
    assert "different scalar types" in capsys.readouterr().out
