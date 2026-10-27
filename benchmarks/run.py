"""Run every benchmark case and write the results as JSON (plus a markdown table).

    python benchmarks/run.py                 # all sizes
    python benchmarks/run.py --max-n 100000  # skip the 1M-row cases

Results land in benchmarks/results/<pandas version>-<platform>.json. The README's numbers
are copied from these files; the machine and library versions are recorded inside.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import CASES

from pandas_doctor import __version__ as pdd_version
from pandas_doctor.verify.api import verify_code
from pandas_doctor.verify.timing import format_seconds, format_speedup

RESULTS = Path(__file__).resolve().parent / "results"


def machine() -> dict[str, str]:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
    return {
        "cpu": cpu,
        "cores": str(os.cpu_count()),
        "os": platform.platform(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": _pyarrow(),
        "pandas_doctor": pdd_version,
    }


def _pyarrow() -> str:
    try:
        import pyarrow

        return str(pyarrow.__version__)
    except ImportError:
        return "not installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--only", help="run cases whose rule or name contains this")
    parser.add_argument("--budget", type=float, default=4.0)
    args = parser.parse_args()

    info = machine()
    print(
        f"# pandas-doctor benchmarks: {info['cpu']}, Python {info['python']}, pandas {info['pandas']}"
    )
    rows = []
    started = time.time()
    for case in CASES:
        if args.only and args.only not in case.rule and args.only not in case.name:
            continue
        for n in case.sizes:
            if args.max_n and n > args.max_n:
                continue
            result = verify_code(
                case.original,
                case.rewrite,
                setup=case.setup.format(n=n),
                outputs=list(case.outputs),
                budget=args.budget,
            )
            if result.refused or result.original is None or result.candidate is None:
                print(f"  {case.rule} {case.name} n={n}: not measured ({result.refused})")
                continue
            row = {
                "rule": case.rule,
                "case": case.name,
                "n": n,
                "original_seconds": result.original.seconds,
                "original_runs": result.original.runs,
                "rewrite_seconds": result.candidate.seconds,
                "rewrite_runs": result.candidate.runs,
                "speedup": result.speedup,
                "verdict": result.verdict.describe(),
            }
            rows.append(row)
            print(
                f"  {case.rule} {case.name:<45} n={n:>9,}  "
                f"{format_seconds(row['original_seconds']):>9} -> {format_seconds(row['rewrite_seconds']):>9}  "
                f"{format_speedup(row['original_seconds'], row['rewrite_seconds']):>6}  {row['verdict']}",
                flush=True,
            )
    RESULTS.mkdir(exist_ok=True)
    arrow = "" if info["pyarrow"] == "not installed" else "-pyarrow"
    name = f"pandas-{info['pandas']}{arrow}-{sys.platform}.json"
    out = RESULTS / name
    out.write_text(
        json.dumps(
            {"machine": info, "seconds": round(time.time() - started), "results": rows}, indent=2
        )
        + "\n"
    )
    print(f"wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
