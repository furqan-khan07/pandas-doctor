"""Timing that never claims more precision than it has.

Every number pandas-doctor prints comes from here, together with how it was measured:
"timed once" for a single run, otherwise the median of N runs. Setup work (copying
inputs so each run starts fresh) is never inside the timed region.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_BUDGET = 2.0  # seconds spent repeating one measurement
MAX_RUNS = 25


@dataclass(frozen=True)
class Timing:
    seconds: float  # median
    runs: int

    @property
    def how(self) -> str:
        return "timed once" if self.runs == 1 else f"median of {self.runs}"

    def to_dict(self) -> dict[str, object]:
        return {"seconds": self.seconds, "runs": self.runs}


def measure(
    fn: Callable[[], object],
    setup: Callable[[], None] | None = None,
    *,
    budget: float = DEFAULT_BUDGET,
    min_runs: int = 3,
    max_runs: int = MAX_RUNS,
) -> Timing:
    """Run `fn` repeatedly (calling `setup` before each run, untimed); return the median.

    Stops after `max_runs`, or once `budget` seconds of timed work have accumulated and
    at least `min_runs` ran. A single run longer than the budget is reported as timed once.
    """
    samples: list[float] = []
    spent = 0.0
    while len(samples) < max_runs:
        if setup is not None:
            setup()
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        spent += elapsed
        if elapsed >= budget:
            break  # one long run says enough
        if spent >= budget and len(samples) >= min_runs:
            break
    return Timing(statistics.median(samples), len(samples))


def format_seconds(seconds: float) -> str:
    if seconds >= 100:
        return f"{seconds:.0f} s"
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms" if seconds < 0.1 else f"{seconds * 1e3:.0f} ms"
    return f"{seconds * 1e6:.0f} µs"


def format_speedup(before: float, after: float) -> str:
    if after <= 0:
        return "n/a"
    ratio = before / after
    if ratio >= 10:
        return f"{ratio:.0f}x"
    return f"{ratio:.1f}x"
