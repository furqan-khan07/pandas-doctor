"""MCP server: lets AI assistants check pandas code and prove their rewrites.

    pip install "pandas-doctor[mcp]"
    pandas-doctor mcp            # speaks MCP over stdio

The point is that an assistant shouldn't claim a pandas rewrite is equivalent or faster
without evidence. `verify_rewrite` runs the original and the rewrite on the same inputs,
compares the results strictly, and returns measured timings. It only runs code that
passes the purity check (no files, network, subprocesses or randomness in the
original/rewrite), the same rule `pandas-doctor verify` uses.
"""

from __future__ import annotations

from typing import Any

INSTRUCTIONS = (
    "pandas-doctor finds slow pandas patterns and proves rewrites. Before telling a user that a "
    "pandas rewrite is equivalent or faster, call verify_rewrite with the original code, the "
    "rewrite, and setup code that builds representative inputs. Report its verdict and timings "
    "as returned; they hold for the given inputs only."
)


def check_code(code: str, filename: str = "snippet.py") -> dict[str, Any]:
    """Find slow pandas patterns in Python source; returns findings with suggested rewrites."""
    from pandas_doctor.static.engine import check_source

    result = check_source(code, filename)
    return {
        "findings": [f.to_dict() for f in result.findings],
        "error": result.error,
        "note": "Suggested rewrites are not verified; use verify_rewrite on representative data.",
    }


def verify_rewrite(
    original: str,
    candidate: str,
    setup: str = "",
    outputs: list[str] | None = None,
    rtol: float = 1e-9,
) -> dict[str, Any]:
    """Run `original` and `candidate` on the same inputs (built by `setup`), compare the
    variables they produce, and time both. `verdict` is one of: "identical";
    "scalar_types" (every value equal, but some scalars have a different type, e.g.
    numpy.float64 vs float; the pairs are in details.scalar_types, and it is not
    identical); "within_tolerance" (floats differ by at most rtol; the largest difference
    is reported); or "different"."""
    from pandas_doctor.verify.api import verify_code

    result = verify_code(original, candidate, setup=setup, outputs=outputs, rtol=rtol, budget=2.0)
    return result.to_dict()


def list_rules() -> list[dict[str, str]]:
    """The patterns pandas-doctor knows, with their codes."""
    from pandas_doctor.static.rules import ALL_RULES

    rules = [
        {
            "code": r.code,
            "name": r.name,
            "summary": r.summary,
            "fix": r.fix_safety or "suggestion only",
        }
        for r in ALL_RULES
    ]
    rules.append(
        {
            "code": "PDD006",
            "name": "astype-loop",
            "summary": "per-column astype loop",
            "fix": "run only",
        }
    )
    rules.append(
        {
            "code": "PDD008",
            "name": "category-memory",
            "summary": "text columns smaller as category",
            "fix": "run only",
        }
    )
    return rules


def build_server() -> Any:
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server  # type: ignore[no-redef,attr-defined]

    server = Server("pandas-doctor", instructions=INSTRUCTIONS)
    server.tool()(check_code)
    server.tool()(verify_rewrite)
    server.tool()(list_rules)
    return server


def main() -> None:
    build_server().run()
