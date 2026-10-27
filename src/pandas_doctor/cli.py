"""Command-line entry point: `pandas-doctor <command>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pandas_doctor import __version__
from pandas_doctor.config import apply_cli, load_config
from pandas_doctor.findings import Severity

EXIT_CLEAN, EXIT_FINDINGS, EXIT_ERROR = 0, 1, 2


def _cmd_check(args: argparse.Namespace) -> int:
    from pandas_doctor.checker import check_paths
    from pandas_doctor.report import write_json, write_text

    paths = [Path(p) for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"pandas-doctor: no such file or directory: {', '.join(missing)}", file=sys.stderr)
        return EXIT_ERROR
    config = apply_cli(load_config(paths[0]), args.select, args.ignore)
    result = check_paths(paths, config, fix=args.fix, diff=args.diff, unsafe=args.unsafe_fixes)
    if args.diff:
        for d in result.diffs:
            sys.stdout.write(d)
        if result.diffs:
            print()
    if args.format == "json":
        write_json(result.findings, result.errors)
    elif not (args.diff and not result.findings):
        write_text(
            result.findings,
            result.errors,
            fixed=result.fixed if args.fix else 0,
            verbose=args.verbose,
        )
    if result.errors and not result.findings:
        return EXIT_ERROR if not args.exit_zero else EXIT_CLEAN
    # info findings (no rewrite to offer) are reported but never fail the check
    failing = [f for f in result.findings if f.severity is not Severity.INFO]
    if failing and not args.exit_zero:
        return EXIT_FINDINGS
    return EXIT_CLEAN


def _cmd_rules(args: argparse.Namespace) -> int:
    from pandas_doctor.static.rules import ALL_RULES

    for rule in ALL_RULES:
        fix = {"safe": "fix ", "unsafe": "fix*"}.get(rule.fix_safety or "", "    ")
        print(f"{rule.code}  {fix}  {rule.name:<18} {rule.summary}")
    print(
        f"PDD006  run   {'astype-loop':<18} per-column astype loop; reported by `run` only when one astype call is faster"
    )
    print(
        f"PDD008  run   {'category-memory':<18} text columns much smaller as category; `run` measures the saving"
    )
    print(
        "\nrun:  checked only by `pandas-doctor run` (needs the data).\n"
        "fix:  `check --fix` rewrites it; the result is provably the same.\n"
        "fix*: needs `--unsafe-fixes`; identical only under a stated condition, "
        "which `pandas-doctor run` checks on your data."
    )
    return EXIT_CLEAN


def _cmd_run(args: argparse.Namespace) -> int:
    import difflib
    import json

    from pandas_doctor.runtime import run_script
    from pandas_doctor.runtime.report import report_dict, write_memory, write_report

    script = Path(args.script)
    if not script.is_file():
        print(f"pandas-doctor: no such script: {script}", file=sys.stderr)
        return EXIT_ERROR
    accept = args.rtol is not None
    outcome = run_script(
        script,
        list(args.args),
        rtol=args.rtol if accept else 1e-9,
        accept_tolerance=accept,
        strict_types=args.strict_types,
        budget=args.budget,
    )
    hint = (
        ""
        if (args.fix or args.diff)
        else "Run again with --diff to see them as a patch, or --fix to apply them."
    )
    write_report(outcome.doctor, str(script), outcome.source.splitlines(), hint=hint)
    write_memory(outcome.memory)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report_dict(outcome.doctor, str(script)), indent=2) + "\n"
        )
    if args.diff or args.fix:
        new_source, skipped = outcome.rewritten()
        for note in skipped:
            print(f"not rewritten: {note}", file=sys.stderr)
        if new_source != outcome.source:
            if args.diff:
                sys.stdout.write(
                    "".join(
                        difflib.unified_diff(
                            outcome.source.splitlines(keepends=True),
                            new_source.splitlines(keepends=True),
                            fromfile=str(script) if script.is_absolute() else f"a/{script}",
                            tofile=str(script) if script.is_absolute() else f"b/{script}",
                        )
                    )
                )
            if args.fix:
                script.write_text(new_source, encoding="utf-8")
                print(f"rewrote {script}", file=sys.stderr)
    return outcome.exit_code


def _cmd_verify(args: argparse.Namespace) -> int:
    import json

    from pandas_doctor.verify.api import verify_code

    def text(inline: str | None, path: str | None) -> str:
        if inline is not None:
            return inline
        if path is not None:
            return Path(path).read_text(encoding="utf-8")
        return ""

    original = text(args.original_code, args.original)
    candidate = text(args.candidate_code, args.candidate)
    if not original or not candidate:
        print(
            "pandas-doctor verify: give the original and the candidate (files or --*-code)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    outputs = [o.strip() for o in args.outputs.split(",")] if args.outputs else None
    result = verify_code(
        original,
        candidate,
        setup=text(args.setup_code, args.setup),
        outputs=outputs,
        rtol=args.rtol,
        trust_setup=args.trust_setup,
        budget=args.budget,
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.describe())
    if result.refused:
        return EXIT_ERROR
    from pandas_doctor.verify.verdict import accepted

    ok = accepted(result.verdict, tolerance=args.accept_tolerance, strict_types=args.strict_types)
    return EXIT_CLEAN if ok else EXIT_FINDINGS


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from pandas_doctor.mcp_server import main as serve
    except ImportError:  # pragma: no cover - depends on the extra
        print(
            'pandas-doctor mcp needs the MCP extra: pip install "pandas-doctor[mcp]"',
            file=sys.stderr,
        )
        return EXIT_ERROR
    try:
        serve()
    except ImportError:
        print(
            'pandas-doctor mcp needs the MCP extra: pip install "pandas-doctor[mcp]"',
            file=sys.stderr,
        )
        return EXIT_ERROR
    return EXIT_CLEAN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pandas-doctor",
        description="Find slow pandas code, rewrite it, and check on your data that the result is identical.",
    )
    parser.add_argument("--version", action="version", version=f"pandas-doctor {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    check = sub.add_parser("check", help="find slow pandas patterns without running anything")
    check.add_argument(
        "paths", nargs="*", default=["."], help="files, folders or notebooks (default: .)"
    )
    check.add_argument("--fix", action="store_true", help="apply the fixes that are provably safe")
    check.add_argument(
        "--diff", action="store_true", help="show the safe fixes as a diff, change nothing"
    )
    check.add_argument(
        "--unsafe-fixes",
        action="store_true",
        help="also apply fixes that are identical only under a stated condition",
    )
    check.add_argument("--select", help="only these rule codes, comma-separated")
    check.add_argument("--ignore", help="skip these rule codes, comma-separated")
    check.add_argument("--format", choices=("text", "json"), default="text")
    check.add_argument(
        "--exit-zero",
        action="store_true",
        help="exit 0 even when warnings are found (info findings never fail the check)",
    )
    check.add_argument("-v", "--verbose", action="store_true", help="say why a fix wasn't offered")
    check.set_defaults(func=_cmd_check)

    run = sub.add_parser(
        "run",
        help="run a script, check rewrites of its slow pandas code on its data, and time them",
        description=(
            "Runs the script as `python script args...` would. For each slow pattern it finds, "
            "it copies the inputs, lets your code run, then runs rewrites on the copies and "
            "compares the results. Verified means: on this run's data."
        ),
    )
    run.add_argument("--diff", action="store_true", help="print verified rewrites as a patch")
    run.add_argument("--fix", action="store_true", help="write verified rewrites into the script")
    run.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="also accept rewrites whose floats match within this relative tolerance",
    )
    run.add_argument(
        "--strict-types",
        action="store_true",
        help="don't accept rewrites whose values match but whose scalar types differ "
        "(e.g. numpy.float64 vs float)",
    )
    run.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    run.add_argument(
        "--budget", type=float, default=2.0, help="seconds spent timing each rewrite (default 2)"
    )
    run.add_argument("script")
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the script")
    run.set_defaults(func=_cmd_run)

    verify = sub.add_parser(
        "verify",
        help="check that a rewrite gives the same result as the original, and time both",
    )
    verify.add_argument("--original", metavar="FILE")
    verify.add_argument("--candidate", metavar="FILE")
    verify.add_argument("--setup", metavar="FILE", help="code that builds the inputs")
    verify.add_argument("--original-code", metavar="CODE")
    verify.add_argument("--candidate-code", metavar="CODE")
    verify.add_argument("--setup-code", metavar="CODE")
    verify.add_argument(
        "--outputs",
        help="variables to compare, comma-separated (default: what the original assigns)",
    )
    verify.add_argument("--rtol", type=float, default=1e-9)
    verify.add_argument(
        "--accept-tolerance", action="store_true", help="exit 0 for 'equal within tolerance' too"
    )
    verify.add_argument(
        "--strict-types",
        action="store_true",
        help="exit 1 for 'equal values, different scalar types' too",
    )
    verify.add_argument(
        "--trust-setup", action="store_true", help="skip the safety check on setup code"
    )
    verify.add_argument("--budget", type=float, default=2.0)
    verify.add_argument("--format", choices=("text", "json"), default="text")
    verify.set_defaults(func=_cmd_verify)

    mcp = sub.add_parser("mcp", help="serve check/verify to AI assistants over MCP (stdio)")
    mcp.set_defaults(func=_cmd_mcp)

    rules = sub.add_parser("rules", help="list the rules")
    rules.set_defaults(func=_cmd_rules)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_CLEAN
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
