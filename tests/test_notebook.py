from __future__ import annotations

import json
from pathlib import Path

import pytest

from pandas_doctor.cli import main
from pandas_doctor.notebook import read_cells, sanitize


def make_notebook(path: Path, cells: list[tuple[str, str]]) -> Path:
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, src in cells
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return path


def test_sanitize_keeps_line_numbers() -> None:
    src = "%matplotlib inline\nimport pandas as pd\n!pip list\ndf?\nx = 1\n"
    out = sanitize(src)
    assert out is not None
    assert out.splitlines() == ["pass", "import pandas as pd", "pass", "pass", "x = 1"]


def test_sanitize_skips_other_languages() -> None:
    assert sanitize("%%bash\nls\n") is None
    assert sanitize("%%time\nx = 1\n") == "pass\nx = 1\n"


def test_cells_numbered_like_jupyter(tmp_path: Path) -> None:
    nb = make_notebook(
        tmp_path / "a.ipynb",
        [("markdown", "# title"), ("code", "import pandas as pd"), ("code", "%%sql\nselect 1")],
    )
    _, cells = read_cells(nb)
    assert [c.index for c in cells] == [2]


def test_check_notebook(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb = make_notebook(
        tmp_path / "eda.ipynb",
        [
            ("markdown", "# features"),
            ("code", "import pandas as pd\n"),
            ("code", "for i, row in df.iterrows():\n    df.at[i, 'y'] = row['x'] + 1\n"),
        ],
    )
    assert main(["check", str(nb)]) == 1
    out = capsys.readouterr().out
    assert "eda.ipynb:cell 3:1:1: PDD001" in out
    assert "df['y'] = df['x'] + 1" in out


def test_fix_writes_back_clean_cells_only(tmp_path: Path) -> None:
    loop = "import pandas as pd\nfor p in parts:\n    out = pd.concat([out, p])\n"
    nb = make_notebook(
        tmp_path / "b.ipynb",
        [("code", loop), ("code", "%time x = 1\n" + loop)],
    )
    main(["check", str(nb), "--fix", "--unsafe-fixes"])
    data = json.loads(nb.read_text())
    first = "".join(data["cells"][0]["source"])
    second = "".join(data["cells"][1]["source"])
    assert "out_parts.append(p)" in first
    assert second == "%time x = 1\n" + loop  # has IPython syntax: untouched


def test_empty_and_non_python_notebooks_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "empty.ipynb").write_text("")
    r_nb = make_notebook(tmp_path / "r.ipynb", [("code", "for (i in 1:10) print(i)")])
    data = json.loads(r_nb.read_text())
    data["metadata"] = {"kernelspec": {"language": "R", "name": "ir"}}
    r_nb.write_text(json.dumps(data))
    assert main(["check", str(tmp_path)]) == 0
