"""Print the README's benchmark table from a results file.

python benchmarks/table.py benchmarks/results/pandas-3.1.0-darwin.json --n 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pandas_doctor.verify.timing import format_seconds, format_speedup


def verdict(text: str) -> str:
    if text == "identical":
        return "identical"
    if text.startswith("equal within"):
        rel = text.split("largest relative difference ")[1].split(")")[0]
        return f"equal to {rel} (float rounding)"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument(
        "--n", type=int, default=100_000, help="rows to quote (falls back to the largest run)"
    )
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text())
    m = data["machine"]
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in data["results"]:
        by_case.setdefault(str(row["case"]), []).append(row)
    print("| Rule | Pattern | Rows | Before | After | Speedup | Result |")
    print("|---|---|---:|---:|---:|---:|---|")
    for case, rows in by_case.items():
        pick = next((r for r in rows if r["n"] == args.n), max(rows, key=lambda r: int(r["n"])))  # type: ignore[arg-type]
        before, after = float(pick["original_seconds"]), float(pick["rewrite_seconds"])  # type: ignore[arg-type]
        ratio = (
            format_speedup(before, after)
            if after <= before
            else f"{format_speedup(after, before)} slower"
        )
        print(
            f"| {pick['rule']} | {case} | {int(pick['n']):,} | {format_seconds(before)} | "  # type: ignore[call-overload]
            f"{format_seconds(after)} | {ratio} | {verdict(str(pick['verdict']))} |"
        )
    print()
    print(
        f"_{m['cpu']}, Python {m['python']}, pandas {m['pandas']}, numpy {m['numpy']}, "
        f'pyarrow {m["pyarrow"]}. Each "before" ran on a fresh copy; times are medians '
        f"unless a run took longer than the timing budget._"
    )


if __name__ == "__main__":
    main()
