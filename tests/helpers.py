"""Shared helpers: check snippets and run before/after code on the same inputs."""

from __future__ import annotations

import copy
import textwrap
import warnings
from typing import Any

import numpy as np
import pandas as pd

from pandas_doctor.findings import Finding
from pandas_doctor.static.engine import check_source, fix_source


def findings(source: str, code: str | None = None) -> list[Finding]:
    result = check_source(textwrap.dedent(source), "snippet.py")
    assert result.error is None, result.error
    return [f for f in result.findings if code is None or f.code == code]


def fixed(source: str, *, unsafe: bool = True) -> str:
    """The source after `check --fix` (unchanged if nothing is fixable)."""
    source = textwrap.dedent(source)
    result = fix_source(source, "snippet.py", unsafe=unsafe)
    assert result.error is None, result.error
    return result.fixed_source if result.fixed_source is not None else source


def run(source: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute `source` with deep copies of `inputs`; return the namespace afterwards."""
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    namespace.update(copy.deepcopy(inputs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(textwrap.dedent(source), "<snippet>", "exec"), namespace)
    return namespace
