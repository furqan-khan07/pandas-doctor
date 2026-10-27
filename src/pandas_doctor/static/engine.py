"""Run the rules over source code, and apply the fixes that are safe."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst

from pandas_doctor.config import Config
from pandas_doctor.findings import Finding, Fix
from pandas_doctor.static.rules import ALL_RULES, FileContext, Rule

MAX_FIX_PASSES = 10


@dataclass
class FileResult:
    path: str
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None
    fixed_source: str | None = None  # set when fixes changed the file
    fixes_applied: int = 0


def _active_rules(source: str, config: Config) -> list[type[Rule]]:
    return [
        r
        for r in ALL_RULES
        if config.enabled(r.code) and (not r.triggers or any(t in source for t in r.triggers))
    ]


def check_source(source: str, path: str = "<string>", config: Config | None = None) -> FileResult:
    """Findings for one file's source. Parse errors come back in `error`, not as exceptions."""
    config = config or Config()
    if not _active_rules(source, config):
        return FileResult(path)  # nothing any rule looks for: skip parsing
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        return FileResult(path, error=f"could not parse: {exc.message} (line {exc.raw_line})")
    return FileResult(path, findings=_run_rules(module, path, config)[0])


def _run_rules(module: cst.Module, path: str, config: Config) -> tuple[list[Finding], FileContext]:
    ctx = FileContext(path=path, module=module)
    rules = [rule_cls(ctx) for rule_cls in _active_rules(module.code, config)]
    if rules:
        # no metadata is declared, so this is a plain traversal; rules resolve it lazily
        cst.visit_batched(module, rules)
    findings = [f for rule in rules for f in rule.findings]
    findings.sort(key=lambda f: (f.line, f.column, f.code))
    return findings, ctx


class _ApplyFixes(cst.CSTTransformer):
    def __init__(self, fixes: list[Fix]) -> None:
        super().__init__()
        self.by_target = {id(f.target): f for f in fixes}

    def on_leave(  # type: ignore[override]
        self, original_node: cst.CSTNode, updated_node: cst.CSTNode
    ) -> cst.CSTNode | cst.FlattenSentinel[cst.CSTNode] | cst.RemovalSentinel:
        fix = self.by_target.get(id(original_node))
        return fix.replacement if fix is not None else updated_node


def _non_overlapping(fixes: list[tuple[Fix, tuple[int, int, int, int]]]) -> list[Fix]:
    chosen: list[tuple[Fix, tuple[int, int, int, int]]] = []
    for fix, span in sorted(fixes, key=lambda item: item[1]):
        start = (span[0], span[1])
        if any(start < (c[2], c[3]) and (c[0], c[1]) < (span[2], span[3]) for _, c in chosen):
            continue  # overlaps a fix already chosen; the next pass picks it up
        chosen.append((fix, span))
    return [fix for fix, _ in chosen]


def fix_source(
    source: str, path: str = "<string>", config: Config | None = None, *, unsafe: bool = False
) -> FileResult:
    """Apply every safe fix (and unsafe ones if asked), re-checking after each pass."""
    config = config or Config()
    if not _active_rules(source, config):
        return FileResult(path)
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        return FileResult(path, error=f"could not parse: {exc.message} (line {exc.raw_line})")
    applied = 0
    for _ in range(MAX_FIX_PASSES):
        findings, ctx = _run_rules(module, path, config)
        candidates = []
        for f in findings:
            if f.fix is None or not (f.fix.safe or unsafe):
                continue
            pos = ctx.position(f.fix.target)
            candidates.append(
                (f.fix, (pos.start.line, pos.start.column, pos.end.line, pos.end.column))
            )
        if not candidates:
            break
        fixes = _non_overlapping(candidates)
        new_module = ctx.module.visit(_ApplyFixes(fixes))
        new_code = new_module.code
        compile(new_code, path, "exec")  # a fix must never produce code that doesn't parse
        module = cst.parse_module(new_code)
        applied += len(fixes)
    remaining, _ = _run_rules(module, path, config)
    result = FileResult(path, findings=remaining, fixes_applied=applied)
    if module.code != source:
        result.fixed_source = module.code
    return result


def iter_python_files(paths: Iterable[Path], config: Config) -> Iterator[Path]:
    """Python files and notebooks under `paths`, skipping excluded folders."""
    seen: set[Path] = set()
    for root in paths:
        if root.is_file():
            if root not in seen:
                seen.add(root)
                yield root
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix not in {".py", ".ipynb"} or not p.is_file():
                continue
            rel_parts = p.relative_to(root).parts
            if any(_excluded(part, config) for part in rel_parts[:-1]):
                continue
            if _excluded(p.name, config) or p in seen:
                continue
            seen.add(p)
            yield p


def _excluded(part: str, config: Config) -> bool:
    return any(fnmatch.fnmatch(part, pattern) for pattern in config.exclude)
