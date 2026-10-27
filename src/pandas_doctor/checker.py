"""`pandas-doctor check`: run the static rules over files and notebooks, optionally fixing."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from pandas_doctor import notebook
from pandas_doctor.config import Config
from pandas_doctor.findings import Finding
from pandas_doctor.static.engine import FileResult, check_source, fix_source, iter_python_files


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fixed: int = 0
    diffs: list[str] = field(default_factory=list)
    files: int = 0


def _diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=path if path.startswith("/") else f"a/{path}",
            tofile=path if path.startswith("/") else f"b/{path}",
        )
    )


def _run(source: str, path: str, config: Config, fixing: bool, unsafe: bool) -> FileResult:
    if fixing:
        return fix_source(source, path, config, unsafe=unsafe)
    return check_source(source, path, config)


def check_paths(
    paths: list[Path],
    config: Config,
    *,
    fix: bool = False,
    diff: bool = False,
    unsafe: bool = False,
) -> CheckResult:
    """Check every file under `paths`. With `fix`, write fixes; with `diff`, only show them.

    Only safe fixes are used unless `unsafe` is set.
    """
    out = CheckResult()
    fixing = fix or diff
    for path in iter_python_files(paths, config):
        out.files += 1
        shown = str(path)
        if path.suffix == ".ipynb":
            _check_notebook(path, shown, config, fixing, fix, unsafe, out)
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out.errors.append(f"{shown}: could not read ({exc})")
            continue
        result = _run(source, shown, config, fixing, unsafe)
        if result.error:
            out.errors.append(f"{shown}: {result.error}")
            continue
        out.findings.extend(result.findings)
        if result.fixed_source is not None:
            out.fixed += result.fixes_applied
            if diff:
                out.diffs.append(_diff(shown, source, result.fixed_source))
            if fix:
                path.write_text(result.fixed_source, encoding="utf-8")
    return out


def _check_notebook(
    path: Path,
    shown: str,
    config: Config,
    fixing: bool,
    write: bool,
    unsafe: bool,
    out: CheckResult,
) -> None:
    try:
        nb, cells = notebook.read_cells(path)
    except (OSError, ValueError) as exc:
        out.errors.append(f"{shown}: could not read notebook ({exc})")
        return
    new_sources: dict[int, str] = {}
    for cell in cells:
        result = _run(cell.source, shown, config, fixing and cell.clean, unsafe)
        if result.error:
            out.errors.append(f"{shown} cell {cell.index}: {result.error}")
            continue
        for f in result.findings:
            f.cell = cell.index
            if f.fix is not None and not cell.clean:
                f.fix = None
                f.fix_skipped = "this cell has IPython syntax, so it isn't rewritten automatically"
        out.findings.extend(result.findings)
        if result.fixed_source is not None:
            out.fixed += result.fixes_applied
            new_sources[cell.index] = result.fixed_source
            out.diffs.append(
                _diff(f"{shown} (cell {cell.index})", cell.original, result.fixed_source)
            )
    if write and new_sources:
        notebook.write_cells(path, nb, new_sources)
