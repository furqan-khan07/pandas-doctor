from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from pandas_doctor.cli import main

SLOW = textwrap.dedent(
    """\
    import pandas as pd

    df = pd.DataFrame({"a": [1.0, 2.0]})
    for i, row in df.iterrows():
        df.at[i, "b"] = row["a"] * 2

    out = pd.DataFrame()
    for p in [df, df]:
        out = pd.concat([out, p])
    """
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "slow.py").write_text(SLOW)
    (tmp_path / "clean.py").write_text("import pandas as pd\nx = 1\n")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "ignored.py").write_text(SLOW)
    return tmp_path


def test_check_reports_and_exits_1(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(project)]) == 1
    out = capsys.readouterr().out
    assert "PDD001" in out and "PDD003" in out
    assert ".venv" not in out
    assert "Found 2 issues (1 fixable with --unsafe-fixes)." in out


def test_clean_file_exits_0(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(project / "clean.py")]) == 0
    assert "No slow pandas patterns found." in capsys.readouterr().out


def test_exit_zero(project: Path) -> None:
    assert main(["check", str(project), "--exit-zero"]) == 0


def test_select_and_ignore(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(project), "--select", "pdd003"])
    out = capsys.readouterr().out
    assert "PDD003" in out and "PDD001" not in out
    main(["check", str(project), "--ignore", "PDD003"])
    out = capsys.readouterr().out
    assert "PDD001" in out and "PDD003" not in out


def test_config_in_pyproject(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (project / "pyproject.toml").write_text('[tool.pandas-doctor]\nignore = ["PDD001"]\n')
    main(["check", str(project)])
    assert "PDD001" not in capsys.readouterr().out


def test_plain_fix_does_not_apply_unsafe_fix(project: Path) -> None:
    main(["check", str(project / "slow.py"), "--fix"])
    assert (project / "slow.py").read_text() == SLOW


def test_unsafe_fix_writes_file(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(project / "slow.py"), "--fix", "--unsafe-fixes"]) == 1
    text = (project / "slow.py").read_text()
    assert "out_parts.append(p)" in text
    assert "fixed 1" in capsys.readouterr().out


def test_diff_changes_nothing(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(project / "slow.py"), "--diff", "--unsafe-fixes"])
    out = capsys.readouterr().out
    assert "+    out_parts.append(p)" in out
    assert (project / "slow.py").read_text() == SLOW


def test_json(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(project), "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    codes = sorted(f["code"] for f in data["findings"])
    assert codes == ["PDD001", "PDD003"]
    pdd001 = next(f for f in data["findings"] if f["code"] == "PDD001")
    assert pdd001["suggestion"] == 'df["b"] = df["a"] * 2'
    assert pdd001["line"] == 4


def test_parse_error_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "broken.py").write_text("def (:\n")
    assert main(["check", str(tmp_path)]) == 2
    assert "could not parse" in capsys.readouterr().out


def test_missing_path(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "does/not/exist"]) == 2


def test_rules_lists_everything(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    for code in ("PDD001", "PDD002", "PDD003"):
        assert code in out
