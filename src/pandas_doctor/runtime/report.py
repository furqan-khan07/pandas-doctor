"""The report `pandas-doctor run` prints after the script finishes."""

from __future__ import annotations

import sys
from typing import TextIO

from pandas_doctor.report import _Style, use_color
from pandas_doctor.runtime.doctor import Doctor, MemoryFinding, SiteResult, environment
from pandas_doctor.verify.timing import format_seconds, format_speedup
from pandas_doctor.verify.verdict import Different, Identical, WithinTolerance

TOP_LINES = 5


def _ratio(before: float, after: float) -> str:
    if after > 0 and before < after:
        return f"{format_speedup(after, before)} slower"
    return f"{format_speedup(before, after)} faster"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _verdict_text(res: SiteResult, style: _Style) -> str:
    chosen = res.chosen
    if chosen is not None:
        v = chosen.verdict
        if isinstance(v, Identical):
            return style.green("identical")
        return style.yellow(v.describe())
    near = next((r for r in res.tried if isinstance(r.verdict, WithinTolerance)), None)
    if near is not None:
        return style.yellow(near.verdict.describe() + "; not identical")
    if res.tried:
        first = res.tried[0].verdict
        assert isinstance(first, Different)
        return style.red(first.describe())
    return style.dim("not checked")


def format_site(res: SiteResult, path: str, n: int, style: _Style) -> str:
    site = res.site
    lines = [style.bold(f"{n}. {path}:{site.line}  {site.rule} {site.message}")]
    if res.calls == 0:
        lines.append(style.dim("   never ran"))
        return "\n".join(lines)
    if res.calls > 1:
        lines.append(
            f"   ran {res.calls} times, {format_seconds(res.total_time)} in total; checked on the first run"
        )
    result = res.chosen or next((r for r in res.tried if r.timing is not None), None)
    if res.original is not None and result is not None and result.timing is not None:
        before, after = res.original, result.timing
        label = "concat, repeated" if site.concat else "before"
        where = " during your run" if before.runs == 1 else ""
        width = 16 if site.concat else 6
        lines.append(
            f"   {label:<{width}} {format_seconds(before.seconds):>9}  ({before.how}{where})"
        )
        after_label = "concat, once" if site.concat else "after"
        lines.append(
            f"   {after_label:<{width}} {format_seconds(after.seconds):>9}  ({after.how})   "
            + style.bold(_ratio(before.seconds, after.seconds))
        )
        if site.concat:
            lines.append(
                style.dim(
                    f"   (the rest of the loop body is unchanged; {res.concat_iterations} iterations)"
                )
            )
    lines.append(
        f"   result {_verdict_text(res, style)}"
        + style.dim("  (verified on this data)" if res.chosen else "")
    )
    if res.chosen is not None and not res.worth_applying:
        lines.append(
            style.yellow("   not faster on this data, so pandas-doctor won't suggest changing it")
        )
    if res.chosen is not None and not site.concat:
        lines.append("   rewrite:")
        lines.append(style.cyan(_indent(res.chosen.candidate.code, "       ")))
        if res.tried and res.tried[0] is not res.chosen:
            first = res.tried[0]
            lines.append(
                style.dim(
                    f"   (the plainest rewrite was {first.verdict.describe()}; this one is {res.chosen.candidate.label})"
                )
            )
    elif res.tried and not site.concat:
        lines.append(style.dim("   closest rewrite tried:"))
        lines.append(style.dim(_indent(res.tried[0].candidate.code, "       ")))
    if res.note:
        lines.append(style.dim(f"   note: {res.note}"))
    return "\n".join(lines)


def write_report(
    doctor: Doctor,
    path: str,
    source_lines: list[str],
    *,
    stream: TextIO | None = None,
    hint: str = "Run again with --diff to see them as a patch, or --fix to apply them.",
) -> None:
    stream = stream or sys.stderr
    style = _Style(use_color(stream))
    print(file=stream)
    print(
        style.bold(f"pandas-doctor run: {path}") + style.dim(f"   ({environment()})"), file=stream
    )

    if doctor.line_times:
        print(file=stream)
        print(style.bold("Where the time went (top-level statements):"), file=stream)
        top = sorted(doctor.line_times.items(), key=lambda kv: kv[1], reverse=True)[:TOP_LINES]
        for line, seconds in top:
            text = source_lines[line - 1].strip() if 0 < line <= len(source_lines) else ""
            print(f"  line {line:<5} {format_seconds(seconds):>9}  {text[:70]}", file=stream)

    # runtime-only checks (PDD006) only appear when they verify and win
    results = [r for r in doctor.results.values() if not r.site.runtime_only or r.worth_applying]
    print(file=stream)
    if not results:
        print("No slow pandas patterns found that pandas-doctor can rewrite.", file=stream)
        return
    print(style.bold("Rewrites checked on this data:"), file=stream)
    for n, res in enumerate(sorted(results, key=lambda r: r.site.line), start=1):
        print(file=stream)
        print(format_site(res, path, n, style), file=stream)
    good = [r for r in results if r.worth_applying]
    identical = sum(1 for r in good if r.chosen and isinstance(r.chosen.verdict, Identical))
    tolerant = sum(1 for r in good if r.chosen and isinstance(r.chosen.verdict, WithinTolerance))
    print(file=stream)
    summary = f"{identical} of {len(results)} rewrites verified identical and faster"
    if tolerant:
        summary += f", {tolerant} equal within tolerance"
    print(style.bold(summary + "."), file=stream)
    if identical or tolerant:
        print(style.dim(hint), file=stream)


def _megabytes(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


def write_memory(findings: list[MemoryFinding], *, stream: TextIO | None = None) -> None:
    if not findings:
        return
    stream = stream or sys.stderr
    style = _Style(use_color(stream))
    print(file=stream)
    print(
        style.bold("Memory (PDD008): text columns that would be much smaller as `category`"),
        file=stream,
    )
    for m in findings:
        print(
            f"  {m.frame}[{m.column!r}]: {m.distinct:,} distinct values in {m.rows:,} rows; "
            f"{_megabytes(m.before_bytes)} -> {_megabytes(m.after_bytes)} "
            + style.bold(f"({m.before_bytes / m.after_bytes:.0f}x smaller, measured)"),
            file=stream,
        )
    print(
        style.dim(
            "  A suggestion only: category changes some behaviour (sorting follows category order, "
            "groupby lists unused categories unless observed=True)."
        ),
        file=stream,
    )


def report_dict(doctor: Doctor, path: str) -> dict[str, object]:
    sites = []
    for res in sorted(doctor.results.values(), key=lambda r: r.site.line):
        chosen = res.chosen
        sites.append(
            {
                "rule": res.site.rule,
                "line": res.site.line,
                "message": res.site.message,
                "calls": res.calls,
                "total_seconds": res.total_time,
                "original_time": res.original.to_dict() if res.original else None,
                "verdict": chosen.verdict.describe()
                if chosen
                else (res.tried[0].verdict.describe() if res.tried else None),
                "verified": chosen is not None,
                "faster": res.worth_applying,
                "speedup": res.speedup,
                "rewrite": chosen.candidate.code if chosen and not res.site.concat else None,
                "rewrite_time": chosen.timing.to_dict() if chosen and chosen.timing else None,
                "note": res.note,
            }
        )
    return {
        "path": path,
        "environment": environment(),
        "scope": "verified on this run's data only",
        "line_seconds": {str(k): v for k, v in sorted(doctor.line_times.items())},
        "sites": sites,
    }
