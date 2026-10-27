"""`pandas-doctor run` and the `%%doctor` notebook magic."""

from __future__ import annotations

import builtins
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pandas_doctor.runtime.doctor import Doctor, MemoryFinding, category_suggestions
from pandas_doctor.runtime.instrument import instrument
from pandas_doctor.runtime.rewrite import apply_rewrites
from pandas_doctor.runtime.sites import SiteMap, find_sites
from pandas_doctor.verify.compare import DEFAULT_RTOL

__all__ = ["RunOutcome", "run_cell", "run_script"]


@dataclass
class RunOutcome:
    doctor: Doctor
    sitemap: SiteMap
    source: str
    exit_code: int
    error: BaseException | None = None
    memory: list[MemoryFinding] = field(default_factory=list)

    def rewritten(self) -> tuple[str, list[str]]:
        return apply_rewrites(self.sitemap, self.doctor.results.values())


def run_script(
    path: Path,
    argv: list[str],
    *,
    rtol: float = DEFAULT_RTOL,
    accept_tolerance: bool = False,
    budget: float = 2.0,
) -> RunOutcome:
    """Run a script the way `python path argv...` would, with the doctor watching."""
    source = path.read_text(encoding="utf-8")
    sitemap = find_sites(source, str(path))
    doctor = Doctor(sitemap.sites, rtol=rtol, accept_tolerance=accept_tolerance, budget=budget)
    code = compile(instrument(source, sitemap.sites), str(path), "exec")
    globs: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(path.resolve()),
        "__builtins__": builtins,
        "__pdd__": doctor,
    }
    old_argv, old_path = sys.argv[:], sys.path[:]
    sys.argv = [str(path), *argv]
    sys.path.insert(0, str(path.resolve().parent))
    exit_code, error = 0, None
    try:
        exec(code, globs)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:
        traceback.print_exc()
        exit_code, error = 1, exc
    finally:
        doctor.mark(None)
        sys.argv, sys.path[:] = old_argv, old_path
    memory = category_suggestions(globs)
    return RunOutcome(doctor, sitemap, source, exit_code, error, memory)


def run_cell(
    source: str,
    namespace: dict[str, Any],
    *,
    rtol: float = DEFAULT_RTOL,
    accept_tolerance: bool = False,
    budget: float = 2.0,
) -> RunOutcome:
    """Run a notebook cell in the notebook's namespace with the doctor watching."""
    sitemap = find_sites(source, "<cell>")
    doctor = Doctor(sitemap.sites, rtol=rtol, accept_tolerance=accept_tolerance, budget=budget)
    code = compile(instrument(source, sitemap.sites), "<doctor cell>", "exec")
    namespace["__pdd__"] = doctor
    error = None
    import ast as _ast

    cell_names = sorted({n.id for n in _ast.walk(_ast.parse(source)) if isinstance(n, _ast.Name)})
    try:
        exec(code, namespace)
    except Exception as exc:
        error = exc
    finally:
        doctor.mark(None)
        for key in [k for k in namespace if k.startswith("__pdd")]:
            del namespace[key]
    memory = category_suggestions(namespace, cell_names)
    return RunOutcome(doctor, sitemap, source, 1 if error else 0, error, memory)
