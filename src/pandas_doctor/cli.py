"""Command-line entry point: `pandas-doctor <command>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pandas_doctor import __version__
from pandas_doctor.config import apply_cli, load_config

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
    if result.findings and not args.exit_zero:
        return EXIT_FINDINGS
    return EXIT_CLEAN


def _cmd_rules(args: argparse.Namespace) -> int:
    from pandas_doctor.static.rules import ALL_RULES

    for rule in ALL_RULES:
        fix = {"safe": "fix ", "unsafe": "fix*"}.get(rule.fix_safety or "", "    ")
        print(f"{rule.code}  {fix}  {rule.name:<16} {rule.summary}")
    print(
        "\nfix:  `check --fix` rewrites it; the result is provably the same.\n"
        "fix*: needs `--unsafe-fixes`; identical only under a stated condition, "
        "which `pandas-doctor run` checks on your data."
    )
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
    check.add_argument("--exit-zero", action="store_true", help="exit 0 even when issues are found")
    check.add_argument("-v", "--verbose", action="store_true", help="say why a fix wasn't offered")
    check.set_defaults(func=_cmd_check)

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
