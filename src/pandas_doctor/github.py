"""Helpers for the GitHub Action: inline annotations and a markdown summary.

    pandas-doctor check . --format json --exit-zero > pdd.json
    python -m pandas_doctor.github pdd.json --annotate > summary.md

`--annotate` prints `::warning file=...` lines (`::notice` for info findings) so findings
show up on the PR diff; the markdown goes to the job summary and, when enabled, a sticky
PR comment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MARKER = "<!-- pandas-doctor -->"


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotation_lines(findings: list[dict[str, Any]]) -> list[str]:
    out = []
    for f in findings:
        message = f["message"]
        if f.get("suggestion"):
            message += f"\nSuggested rewrite:\n{f['suggestion']}"
        if f.get("cell") is not None:
            message = f"(cell {f['cell']}) " + message
        level = {"info": "notice", "error": "error"}.get(f.get("severity", ""), "warning")
        out.append(
            f"::{level} file={f['path']},line={f['line']},col={f['column']},"
            f"title={f['code']}::{_escape(message)}"
        )
    return out


def markdown(findings: list[dict[str, Any]], errors: list[str]) -> str:
    lines = [MARKER, "### pandas-doctor"]
    if not findings:
        lines.append("No slow pandas patterns found. :white_check_mark:")
    else:
        counts = Counter(f["code"] for f in findings)
        summary = ", ".join(f"{n} x {code}" for code, n in sorted(counts.items()))
        lines.append(
            f"Found **{len(findings)}** slow pandas pattern{'s' if len(findings) != 1 else ''} ({summary})."
        )
        lines.append("")
        lines.append("| Where | Rule | What |")
        lines.append("|---|---|---|")
        for f in findings[:50]:
            where = (
                f"`{f['path']}:{f['line']}`"
                if f.get("cell") is None
                else f"`{f['path']}` cell {f['cell']}, line {f['line']}"
            )
            lines.append(f"| {where} | {f['code']} | {f['message'].replace('|', '/')} |")
        if len(findings) > 50:
            lines.append(f"| ... | | {len(findings) - 50} more |")
        lines.append("")
        lines.append(
            "Suggested rewrites are not applied automatically. Run `pandas-doctor run your_script.py` "
            "to check them on real data: it reports whether the result is identical and how much faster it is."
        )
    if errors:
        lines.append("")
        lines.append(f"<details><summary>{len(errors)} file(s) couldn't be checked</summary>\n")
        lines.extend(f"- {e}" for e in errors[:20])
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pandas_doctor.github")
    parser.add_argument("report", help="JSON from `pandas-doctor check --format json`")
    parser.add_argument(
        "--annotate", action="store_true", help="print GitHub annotation commands to stderr"
    )
    args = parser.parse_args(argv)
    data = json.loads(Path(args.report).read_text())
    findings, errors = data.get("findings", []), data.get("errors", [])
    if args.annotate:
        for line in annotation_lines(findings):
            print(line, file=sys.stderr)
    sys.stdout.write(markdown(findings, errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
