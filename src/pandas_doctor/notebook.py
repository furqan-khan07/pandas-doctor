"""Read and write Jupyter notebooks so the static rules can check their code cells.

IPython syntax (`%magic`, `!shell`, `df?`) isn't Python, so those lines are swapped for
`pass` before parsing, keeping line numbers. Cells that are entirely another language
(`%%bash`, `%%sql`, ...) are skipped. Fixes are only written back to cells that had no
IPython syntax, so the cell text round-trips exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Cell magics whose body is still Python.
_PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun", "doctor", "debug"}
_IPY_LINE = re.compile(r"^(\s*)([%!]|\?)|\?\s*$")


@dataclass
class Cell:
    index: int  # 1-based position among all cells, as Jupyter shows it
    source: str  # Python-only text, same line count as the original
    original: str
    clean: bool  # True when `source == original`


def _source_text(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def sanitize(text: str) -> str | None:
    """Python-only version of a cell, or None if the cell isn't Python at all."""
    lines = text.splitlines(keepends=True)
    if lines and lines[0].lstrip().startswith("%%"):
        magic = lines[0].lstrip()[2:].split()[0] if lines[0].strip() != "%%" else ""
        if magic not in _PYTHON_CELL_MAGICS:
            return None
        lines[0] = "pass\n"
    out = []
    for line in lines:
        m = _IPY_LINE.search(line)
        if m and not line.lstrip().startswith("#"):
            indent = re.match(r"\s*", line).group(0)  # type: ignore[union-attr]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}pass{newline}")
        else:
            out.append(line)
    return "".join(out)


def _language(nb: dict[str, Any]) -> str | None:
    meta = nb.get("metadata") or {}
    kernel = meta.get("kernelspec") or {}
    info = meta.get("language_info") or {}
    lang = kernel.get("language") or info.get("name")
    return str(lang).lower() if lang else None


def read_cells(path: Path) -> tuple[dict[str, Any], list[Cell]]:
    """The notebook and its Python code cells. Empty files and notebooks for other
    kernels (R, Julia, ...) have no cells to check."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}, []
    nb = json.loads(text)
    if _language(nb) not in (None, "python", "python3", "ipython", "ipython3"):
        return nb, []
    cells: list[Cell] = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        original = _source_text(cell)
        clean_source = sanitize(original)
        if clean_source is None:
            continue
        cells.append(Cell(i, clean_source, original, clean_source == original))
    return nb, cells


def write_cells(path: Path, nb: dict[str, Any], new_sources: dict[int, str]) -> None:
    """Write fixed cell sources back, keeping the notebook's JSON layout."""
    raw = path.read_text(encoding="utf-8")
    indent = 1
    m = re.search(r'\n( +)"', raw)
    if m:
        indent = len(m.group(1))
    for i, cell in enumerate(nb.get("cells", []), start=1):
        if i in new_sources:
            text = new_sources[i]
            cell["source"] = (
                text.splitlines(keepends=True) if isinstance(cell.get("source"), list) else text
            )
    path.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
