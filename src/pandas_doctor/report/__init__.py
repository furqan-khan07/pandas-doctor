"""Turn findings into text for people or JSON for tools."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import TextIO

from pandas_doctor.findings import Finding


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def use_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def format_finding(f: Finding, style: _Style, *, show_fix_reason: bool = False) -> str:
    head = f"{style.bold(f.location())}: {style.red(f.code)} {f.message}"
    parts = [head]
    if f.fix is not None and f.fix.safe:
        parts.append(style.green(f"    fixable with --fix: {f.fix.description}"))
    elif f.fix is not None:
        parts.append(style.yellow(f"    fixable with --fix --unsafe-fixes: {f.fix.description}"))
        if f.fix.caveat:
            parts.append(style.dim(_indent(f.fix.caveat, "    ")))
    if f.suggestion:
        parts.append(style.cyan("    suggested rewrite:"))
        parts.append(style.cyan(_indent(f.suggestion, "        ")))
    if f.note:
        parts.append(style.dim(_indent(f.note, "    ")))
    if show_fix_reason and f.fix_skipped:
        parts.append(style.dim(f"    no automatic fix: {f.fix_skipped}"))
    return "\n".join(parts)


def write_text(
    findings: Sequence[Finding],
    errors: Sequence[str],
    *,
    stream: TextIO | None = None,
    fixed: int = 0,
    verbose: bool = False,
) -> None:
    stream = stream or sys.stdout
    style = _Style(use_color(stream))
    for f in findings:
        print(format_finding(f, style, show_fix_reason=verbose), file=stream)
        print(file=stream)
    for err in errors:
        print(style.yellow(f"warning: {err}"), file=stream)
    fixable = sum(1 for f in findings if f.fix is not None and f.fix.safe)
    unsafe = sum(1 for f in findings if f.fix is not None and not f.fix.safe)
    if not findings and not fixed:
        print(style.green("No slow pandas patterns found."), file=stream)
        return
    summary = f"Found {len(findings)} issue{'s' if len(findings) != 1 else ''}"
    extras = []
    if fixed:
        extras.append(f"fixed {fixed}")
    if fixable:
        extras.append(f"{fixable} fixable with --fix")
    if unsafe:
        extras.append(f"{unsafe} more with --unsafe-fixes")
    if extras:
        summary += " (" + ", ".join(extras) + ")"
    print(style.bold(summary + "."), file=stream)


def write_json(
    findings: Sequence[Finding], errors: Sequence[str], *, stream: TextIO | None = None
) -> None:
    stream = stream or sys.stdout
    json.dump(
        {"findings": [f.to_dict() for f in findings], "errors": list(errors)},
        stream,
        indent=2,
    )
    stream.write("\n")
