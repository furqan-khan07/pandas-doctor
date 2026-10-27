"""The `%%doctor` cell magic.

    %load_ext pandas_doctor

    %%doctor
    for i, row in df.iterrows():
        df.at[i, "total"] = row["price"] * row["qty"]

The cell runs normally in the notebook's namespace. Afterwards the doctor reports each
slow pattern it checked: the rewrite, whether it gave the identical result on this data,
and both timings. `%%doctor --rewrite` also puts the verified rewrite of the cell into a
new cell below, for you to review and run.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Any

from pandas_doctor.runtime import run_cell
from pandas_doctor.runtime.report import write_memory, write_report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="%%doctor", add_help=False)
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--strict-types", action="store_true")
    p.add_argument("--budget", type=float, default=2.0)
    return p


def register(ipython: Any) -> None:
    def doctor(line: str, cell: str) -> None:
        try:
            args = _parser().parse_args(shlex.split(line))
        except SystemExit:
            print(
                "usage: %%doctor [--rewrite] [--rtol RTOL] [--strict-types] [--budget SECONDS]",
                file=sys.stderr,
            )
            return
        accept = args.rtol is not None
        outcome = run_cell(
            cell,
            ipython.user_ns,
            rtol=args.rtol if accept else 1e-9,
            accept_tolerance=accept,
            strict_types=args.strict_types,
            budget=args.budget,
        )
        write_report(
            outcome.doctor,
            "cell",
            cell.splitlines(),
            stream=sys.stdout,
            hint="Use `%%doctor --rewrite` to get the verified rewrite as a new cell.",
        )
        write_memory(outcome.memory, stream=sys.stdout)
        if args.rewrite:
            new_source, skipped = outcome.rewritten()
            for note in skipped:
                print(f"not rewritten: {note}")
            if new_source != cell:
                ipython.set_next_input(new_source, replace=False)
        if outcome.error is not None:
            raise outcome.error

    ipython.register_magic_function(doctor, magic_kind="cell", magic_name="doctor")
