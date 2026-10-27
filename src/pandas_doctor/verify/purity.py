"""Refuse to re-run code that could touch anything outside its own variables.

pandas-doctor runs an original block more than once (for timing) and runs rewrites next
to it. That's only fine for code that computes on data: no files, network, processes,
randomness, global state or imports beyond pandas/numpy/math. This check is
deliberately strict and syntactic; anything it can't vouch for is refused, and the
caller falls back to timing the real run once.
"""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "math"})
SETUP_IMPORTS = ALLOWED_IMPORTS | frozenset(
    {
        "datetime",
        "decimal",
        "fractions",
        "statistics",
        "string",
        "itertools",
        "functools",
        "collections",
        "random",
        "re",
    }
)
_BLOCKED_CALLS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "setattr",
        "delattr",
        "help",
    }
)
_BLOCKED_ROOTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pathlib",
        "urllib",
        "http",
        "requests",
        "pickle",
        "importlib",
        "ctypes",
        "threading",
        "multiprocessing",
        "asyncio",
        "signal",
        "tempfile",
        "glob",
        "io",
        "builtins",
    }
)
_IO_METHOD_PREFIXES = ("to_", "read_", "write", "save", "plot", "show", "dump")
_IO_METHOD_ALLOWED = frozenset(
    {
        "to_numpy",
        "to_list",
        "to_dict",
        "to_frame",
        "to_datetime",
        "to_timedelta",
        "to_numeric",
        "to_period",
        "to_timestamp",
        "to_records",
        "to_string",
        "to_series",
        "to_flat_index",
        "to_offset",
        "to_pydatetime",
        "to_julian_date",
    }
)


class Impure(Exception):
    """Why the code can't be re-run safely."""


def _root(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _Checker(ast.NodeVisitor):
    def __init__(self, imports: frozenset[str], allow_reads: bool, allow_random: bool) -> None:
        self.imports = imports
        self.allow_reads = allow_reads
        self.allow_random = allow_random

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] not in self.imports:
                raise Impure(f"imports `{alias.name}`")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or (node.module or "").split(".")[0] not in self.imports:
            raise Impure(f"imports from `{node.module}`")

    def visit_Global(self, node: ast.Global) -> None:
        raise Impure("uses `global`")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise Impure("uses `nonlocal`")

    def visit_Yield(self, node: ast.Yield) -> None:
        raise Impure("contains `yield`")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        raise Impure("contains `yield`")

    def visit_Await(self, node: ast.Await) -> None:
        raise Impure("contains `await`")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_ROOTS:
            raise Impure(f"uses `{node.id}`")
        if node.id == "random" and not self.allow_random:
            raise Impure("uses randomness")
        if node.id.startswith("__") and node.id.endswith("__") and node.id != "__name__":
            raise Impure(f"uses `{node.id}`")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise Impure(f"uses `.{node.attr}`")
        if node.attr == "random" and _root(node) in {"np", "numpy"} and not self.allow_random:
            raise Impure("uses numpy randomness")
        if isinstance(node.ctx, ast.Store) and _root(node) in {
            "pd",
            "np",
            "pandas",
            "numpy",
            "math",
        }:
            raise Impure("changes library settings")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
            raise Impure(f"calls `{func.id}()`")
        if isinstance(func, ast.Attribute):
            name = func.attr
            is_io = name.startswith(_IO_METHOD_PREFIXES) and name not in _IO_METHOD_ALLOWED
            if is_io and not (self.allow_reads and name.startswith("read_")):
                raise Impure(f"calls `.{name}()`, which can read or write outside the program")
            if name in {"set_option", "reset_option", "seterr"} or (
                name == "seed" and not self.allow_random
            ):
                raise Impure(f"calls `.{name}()`, which changes global state")
        self.generic_visit(node)


def check_pure(source: str) -> None:
    """Raise Impure unless `source` only computes on its own variables."""
    _Checker(ALLOWED_IMPORTS, allow_reads=False, allow_random=False).visit(ast.parse(source))


def check_setup(source: str) -> None:
    """Setup code may also read files and use (seeded) randomness to build inputs."""
    _Checker(SETUP_IMPORTS, allow_reads=True, allow_random=True).visit(ast.parse(source))


def is_pure(source: str) -> bool:
    try:
        check_pure(source)
    except (Impure, SyntaxError):
        return False
    return True
