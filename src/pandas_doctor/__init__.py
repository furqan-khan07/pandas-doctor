"""pandas-doctor: find slow pandas code, rewrite it, and check on your data that the result is identical."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "compare", "load_ipython_extension", "verify_code"]


def __getattr__(name: str) -> Any:
    # imported lazily so `pandas-doctor check` doesn't pay for importing pandas
    if name == "verify_code":
        from pandas_doctor.verify.api import verify_code

        return verify_code
    if name == "compare":
        from pandas_doctor.verify.compare import compare

        return compare
    raise AttributeError(name)


def load_ipython_extension(ipython: Any) -> None:
    """`%load_ext pandas_doctor` registers the `%%doctor` cell magic."""
    from pandas_doctor.magic import register

    register(ipython)
