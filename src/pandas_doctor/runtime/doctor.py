"""The runtime half: let the user's code run for real, and check rewrites next to it.

`Doctor` is installed as `__pdd__` in the instrumented module. At each site it copies the
inputs, times the original as it runs, then runs candidate rewrites on the copies and
compares their outputs with what the original produced. The user's program always runs
its own code; rewrites only ever see copies.

Verification happens on the first call of a site (later calls are only timed), and its
cost is tracked as overhead and left out of the per-line timings.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pandas_doctor.runtime.sites import Candidate, Site
from pandas_doctor.verify.compare import DEFAULT_RTOL, compare
from pandas_doctor.verify.snapshot import NotCopyable, nbytes, snapshot_all
from pandas_doctor.verify.timing import Timing, measure
from pandas_doctor.verify.verdict import Different, Identical, Verdict, WithinTolerance, worst

REPEAT_LIMIT = 1.0  # originals faster than this are re-run on copies for a median
MAX_COPY_BYTES = 2 * 1024**3
_MISSING = object()


@dataclass
class CandidateResult:
    candidate: Candidate
    verdict: Verdict
    timing: Timing | None = None


@dataclass
class SiteResult:
    site: Site
    calls: int = 0
    total_time: float = 0.0
    original: Timing | None = None
    tried: list[CandidateResult] = field(default_factory=list)
    chosen: CandidateResult | None = None
    note: str | None = None
    concat_iterations: int = 0

    @property
    def verified(self) -> bool:
        return self.chosen is not None


def _lookup(name: str, loc: dict[str, Any], glob: dict[str, Any]) -> Any:
    if name in loc:
        return loc[name]
    return glob.get(name, _MISSING)


@contextlib.contextmanager
def _quiet() -> Any:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


class Doctor:
    def __init__(
        self,
        sites: list[Site],
        *,
        rtol: float = DEFAULT_RTOL,
        accept_tolerance: bool = False,
        budget: float = 2.0,
    ) -> None:
        self.sites = {s.id: s for s in sites}
        self.results = {s.id: SiteResult(s) for s in sites}
        self.rtol = rtol
        self.accept_tolerance = accept_tolerance
        self.budget = budget
        self.line_times: dict[int, float] = {}
        self._mark: tuple[int, float, float] | None = None
        self.overhead = 0.0
        self._concat: dict[int, dict[str, Any]] = {}

    # -- per-line timing --------------------------------------------------------------

    def mark(self, lineno: int | None) -> None:
        now = time.perf_counter()
        if self._mark is not None:
            line, start, overhead_then = self._mark
            spent = now - start - (self.overhead - overhead_then)
            self.line_times[line] = self.line_times.get(line, 0.0) + max(spent, 0.0)
        self._mark = (lineno, now, self.overhead) if lineno is not None else None

    # -- sites ------------------------------------------------------------------------

    def enter(self, sid: int, loc: dict[str, Any], glob: dict[str, Any]) -> tuple[Any, float]:
        t0 = time.perf_counter()
        site, res = self.sites[sid], self.results[sid]
        res.calls += 1
        snap = None
        if res.calls == 1:
            if site.concat is not None:
                start = _lookup(site.concat.target, loc, glob)
                if start is not _MISSING:
                    self._concat[sid] = {"start": start, "parts": [], "time": 0.0, "fn": None}
            elif site.build is not None:
                snap = self._snapshot(site, res, loc, glob)
        self.overhead += time.perf_counter() - t0
        return snap, time.perf_counter()

    def _snapshot(
        self, site: Site, res: SiteResult, loc: dict[str, Any], glob: dict[str, Any]
    ) -> dict[str, Any] | None:
        values = {}
        for name in site.inputs:
            value = _lookup(name, loc, glob)
            if value is not _MISSING and not name.startswith("__pdd"):
                values[name] = value
        if sum(nbytes(v) for v in values.values()) > MAX_COPY_BYTES:
            res.note = "inputs too large to copy (over 2 GB), so the rewrite wasn't checked"
            return None
        try:
            return snapshot_all(values)
        except NotCopyable as exc:
            res.note = f"couldn't copy the inputs ({exc}), so the rewrite wasn't checked"
            return None

    def exit(
        self, sid: int, token: tuple[Any, float], loc: dict[str, Any], glob: dict[str, Any]
    ) -> None:
        snap, t_start = token
        elapsed = time.perf_counter() - t_start
        t0 = time.perf_counter()
        site, res = self.sites[sid], self.results[sid]
        res.total_time += elapsed
        try:
            if site.concat is not None and sid in self._concat:
                self._verify_concat(site, res, elapsed, loc, glob)
            elif snap is not None:
                self._verify(site, res, snap, elapsed, loc, glob)
        except Exception as exc:  # the doctor must never break the user's run
            res.note = f"checking failed: {type(exc).__name__}: {exc}"
        self.overhead += time.perf_counter() - t0

    def concat_step(self, sid: int, fn: Any, current: Any, part: Any) -> Any:
        state = self._concat.get(sid)
        start = time.perf_counter()
        result = fn([current, part])
        if state is not None:
            state["time"] += time.perf_counter() - start
            state["parts"].append(part)
            state["fn"] = fn
        return result

    # -- checking -----------------------------------------------------------------------

    def _namespace(
        self, glob: dict[str, Any], loc: dict[str, Any], snap: dict[str, Any]
    ) -> dict[str, Any]:
        ns = dict(glob)
        if loc is not glob:
            ns.update({k: v for k, v in loc.items() if not k.startswith("__pdd")})
        ns.update(snapshot_all(snap))
        ns.setdefault("pd", pd)
        ns.setdefault("np", np)
        return ns

    def _verify(
        self,
        site: Site,
        res: SiteResult,
        snap: dict[str, Any],
        elapsed: float,
        loc: dict[str, Any],
        glob: dict[str, Any],
    ) -> None:
        after = {name: _lookup(name, loc, glob) for name in site.outputs}
        after = {k: v for k, v in after.items() if v is not _MISSING}
        assert site.build is not None
        candidates = site.build(snap, after)
        if not candidates:
            res.note = "no rewrite applies to this data"
            return
        holder: dict[str, dict[str, Any]] = {}

        def fresh() -> None:
            holder["ns"] = self._namespace(glob, loc, snap)

        if elapsed < REPEAT_LIMIT and site.repeatable:
            original_code = compile(site.source, "<original>", "exec")

            def run_original() -> None:
                with _quiet():
                    exec(original_code, holder["ns"])

            res.original = measure(run_original, fresh, budget=self.budget / 2)
        else:
            res.original = Timing(elapsed, 1)

        for cand in candidates:
            compiled = compile(cand.code, "<rewrite>", "exec")
            ns = self._namespace(glob, loc, snap)
            try:
                with _quiet():
                    exec(compiled, ns)
            except Exception as exc:
                res.tried.append(
                    CandidateResult(
                        cand, Different(f"the rewrite raised {type(exc).__name__}: {exc}")
                    )
                )
                continue
            verdict: Verdict = Identical()
            for name, expected in after.items():
                got = ns.get(name, _MISSING)
                v = (
                    Different(f"the rewrite doesn't define `{name}`")
                    if got is _MISSING
                    else compare(expected, got, rtol=self.rtol)
                )
                verdict = worst(verdict, v)
            result = CandidateResult(cand, verdict)
            res.tried.append(result)
            if self._acceptable(verdict):

                def run_candidate(code: Any = compiled) -> None:
                    with _quiet():
                        exec(code, holder["ns"])

                result.timing = measure(run_candidate, fresh, budget=self.budget / 2)
                res.chosen = result
                break
        if res.chosen is None:
            near = next((r for r in res.tried if isinstance(r.verdict, WithinTolerance)), None)
            if near is not None:
                near_code = compile(near.candidate.code, "<rewrite>", "exec")

                def run_near() -> None:
                    with _quiet():
                        exec(near_code, holder["ns"])

                near.timing = measure(run_near, fresh, budget=self.budget / 2)

    def _acceptable(self, verdict: Verdict) -> bool:
        return isinstance(verdict, Identical) or (
            self.accept_tolerance and isinstance(verdict, WithinTolerance)
        )

    def _verify_concat(
        self, site: Site, res: SiteResult, elapsed: float, loc: dict[str, Any], glob: dict[str, Any]
    ) -> None:
        state = self._concat.pop(site.id)
        parts = state["parts"]
        res.concat_iterations = len(parts)
        if not parts or state["fn"] is None:
            res.note = "the loop didn't run, so there was nothing to check"
            return
        assert site.concat is not None
        final = _lookup(site.concat.target, loc, glob)
        fn, start = state["fn"], state["start"]
        with _quiet():
            once = fn([start, *parts])
        verdict = compare(final, once, rtol=self.rtol)
        res.original = Timing(state["time"], 1)
        cand = Candidate(f"one concat over {len(parts)} parts", "collect, then concat once", False)
        result = CandidateResult(cand, verdict)
        res.tried.append(result)

        def run_once() -> None:
            with _quiet():
                fn([start, *parts])

        result.timing = measure(run_once, None, budget=self.budget / 2)
        if self._acceptable(verdict):
            res.chosen = result


def environment() -> str:
    return f"pandas {pd.__version__}, numpy {np.__version__}, Python {sys.version.split()[0]}"
