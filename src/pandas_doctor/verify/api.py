"""`verify`: check that a rewrite gives the same result as the original, and time both.

This is the primitive everything else builds on, and the one AI assistants call through
the MCP server: bring an original snippet and a candidate rewrite, plus code that builds
the inputs, and get back a verdict and two measured timings.

    >>> from pandas_doctor import verify_code
    >>> r = verify_code(
    ...     original="total = 0\\nfor _, row in df.iterrows():\\n    total += row['x']",
    ...     candidate="total = df['x'].sum()",
    ...     setup="import pandas as pd\\ndf = pd.DataFrame({'x': [1.0, 2.0]})",
    ... )
    >>> r.verdict.describe()
    'identical'

The verdict holds for these inputs only; that's what "verified on this data" means.
Verdicts: identical, equal values with different scalar types (e.g. numpy.float64 vs
float), equal within a float tolerance, or different.
"""

from __future__ import annotations

import ast
import contextlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pandas_doctor.verify.compare import DEFAULT_RTOL, compare
from pandas_doctor.verify.purity import Impure, check_pure, check_setup
from pandas_doctor.verify.snapshot import snapshot_all
from pandas_doctor.verify.timing import (
    DEFAULT_BUDGET,
    Timing,
    format_seconds,
    format_speedup,
    measure,
)
from pandas_doctor.verify.verdict import Different, Identical, Verdict, worst

MUTATING_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "update",
        "add",
        "pop",
        "remove",
        "clear",
        "setdefault",
        "discard",
        "sort",
    }
)


@dataclass(frozen=True)
class OutputCheck:
    name: str
    verdict: Verdict


@dataclass
class VerifyResult:
    verdict: Verdict
    outputs: list[OutputCheck] = field(default_factory=list)
    original: Timing | None = None
    candidate: Timing | None = None
    refused: str | None = None  # set when nothing was run

    @property
    def speedup(self) -> float | None:
        if self.original is None or self.candidate is None or self.candidate.seconds <= 0:
            return None
        return self.original.seconds / self.candidate.seconds

    def describe(self) -> str:
        if self.refused:
            return f"not verified: {self.refused}"
        lines = [f"result: {self.verdict.describe()} (verified on this data)"]
        lines.extend(f"  {c.name}: {c.verdict.describe()}" for c in self.outputs)
        if self.original and self.candidate:
            lines.append(
                f"time: {format_seconds(self.original.seconds)} ({self.original.how}) -> "
                f"{format_seconds(self.candidate.seconds)} ({self.candidate.how}), "
                f"{format_speedup(self.original.seconds, self.candidate.seconds)}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        v = self.verdict
        return {
            "verdict": v.kind,
            "description": v.describe(),
            "details": v.to_dict(),
            "outputs": {c.name: c.verdict.to_dict() for c in self.outputs},
            "original_time": self.original.to_dict() if self.original else None,
            "candidate_time": self.candidate.to_dict() if self.candidate else None,
            "speedup": self.speedup,
            "refused": self.refused,
            "scope": "verified on the given inputs only",
        }


def default_outputs(original: str) -> list[str]:
    """Variables the original assigns or modifies in place (loop variables excluded)."""
    tree = ast.parse(original)
    loop_vars: set[str] = set()
    found: list[str] = []

    def add(name: str) -> None:
        if name not in found:
            found.append(name)

    def root(node: ast.expr) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def bind(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for el in target.elts:
                bind(el)
        elif isinstance(target, ast.Starred):
            bind(target.value)
        else:
            name = root(target)
            if name:
                add(name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    loop_vars.add(n.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                bind(t)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            bind(node.target)
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                bind(t)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATING_METHODS
        ) or (
            isinstance(node, ast.Call)
            and any(
                kw.arg == "inplace"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            and isinstance(node.func, ast.Attribute)
        ):
            name = root(node.func.value)
            if name:
                add(name)
    return [n for n in found if n not in loop_vars]


def _exec(code: Any, ns: dict[str, Any]) -> None:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        exec(code, ns)


def verify_code(
    original: str,
    candidate: str,
    *,
    setup: str = "",
    inputs: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    rtol: float = DEFAULT_RTOL,
    trust_setup: bool = False,
    budget: float = DEFAULT_BUDGET,
) -> VerifyResult:
    """Run `original` and `candidate` on the same fresh inputs; compare and time them."""
    try:
        check_pure(original)
        check_pure(candidate)
        if setup and not trust_setup:
            check_setup(setup)
    except Impure as why:
        return VerifyResult(
            Different(str(why)), refused=f"the code {why}; only pure computation is re-run"
        )
    except SyntaxError as exc:
        return VerifyResult(Different("syntax error"), refused=f"syntax error: {exc}")

    base: dict[str, Any] = {"pd": pd, "np": np}
    if setup:
        try:
            _exec(compile(setup, "<setup>", "exec"), base)
        except Exception as exc:
            return VerifyResult(
                Different("setup failed"), refused=f"setup raised {type(exc).__name__}: {exc}"
            )
    if inputs:
        base.update(inputs)
    base = {k: v for k, v in base.items() if not k.startswith("__")}

    original_code = compile(original, "<original>", "exec")
    candidate_code = compile(candidate, "<candidate>", "exec")

    ns_original = snapshot_all(base)
    try:
        _exec(original_code, ns_original)
    except Exception as exc:
        return VerifyResult(
            Different("original failed"), refused=f"the original raised {type(exc).__name__}: {exc}"
        )
    ns_candidate = snapshot_all(base)
    try:
        _exec(candidate_code, ns_candidate)
    except Exception as exc:
        return VerifyResult(Different(f"the rewrite raised {type(exc).__name__}: {exc}"))

    names = outputs if outputs is not None else default_outputs(original)
    names = [n for n in names if n in ns_original]
    if not names:
        return VerifyResult(
            Different("nothing to compare"),
            refused="the original defines no outputs to compare; pass outputs explicitly",
        )
    checks: list[OutputCheck] = []
    verdict: Verdict = Identical()
    for name in names:
        if name not in ns_candidate:
            v: Verdict = Different(f"the rewrite doesn't define `{name}`")
        else:
            v = compare(ns_original[name], ns_candidate[name], rtol=rtol)
        checks.append(OutputCheck(name, v))
        verdict = worst(verdict, v)

    holder: dict[str, dict[str, Any]] = {}

    def fresh() -> None:
        holder["ns"] = snapshot_all(base)

    t_original = measure(lambda: _exec(original_code, holder["ns"]), fresh, budget=budget / 2)
    t_candidate = measure(lambda: _exec(candidate_code, holder["ns"]), fresh, budget=budget / 2)
    return VerifyResult(verdict, checks, t_original, t_candidate)
