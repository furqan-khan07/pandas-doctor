"""Settings from `[tool.pandas-doctor]` in pyproject.toml, merged with command-line flags."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXCLUDE = (
    ".git",
    ".hg",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".nox",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    "site-packages",
    ".ipynb_checkpoints",
)


@dataclass
class Config:
    select: set[str] | None = None  # None means every rule
    ignore: set[str] = field(default_factory=set)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))

    def enabled(self, code: str) -> bool:
        if code in self.ignore:
            return False
        return self.select is None or code in self.select


def _codes(value: object) -> set[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        raise ValueError(f"expected a list of rule codes, got {value!r}")
    return {str(v).strip().upper() for v in value if str(v).strip()}


def find_pyproject(start: Path) -> Path | None:
    """Walk up from `start` to the nearest pyproject.toml."""
    here = start.resolve()
    if here.is_file():
        here = here.parent
    for folder in (here, *here.parents):
        candidate = folder / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def load_config(start: Path) -> Config:
    config = Config()
    pyproject = find_pyproject(start)
    if pyproject is None:
        return config
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("tool", {}).get("pandas-doctor", {})
    if "select" in section:
        config.select = _codes(section["select"])
    if "ignore" in section:
        config.ignore = _codes(section["ignore"])
    if "exclude" in section:
        config.exclude = [*DEFAULT_EXCLUDE, *[str(p) for p in section["exclude"]]]
    return config


def apply_cli(config: Config, select: str | None, ignore: str | None) -> Config:
    if select:
        config.select = _codes(select)
    if ignore:
        config.ignore |= _codes(ignore)
    return config
