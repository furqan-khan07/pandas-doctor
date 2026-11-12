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

from typing import Annotated, Any

from pydantic import Field

INSTRUCTIONS = (
    "pandas-doctor finds slow pandas patterns and proves rewrites. Before telling a user that a "
    "pandas rewrite is equivalent or faster, call verify_rewrite with the original code, the "
    "rewrite, and setup code that builds representative inputs. Report its verdict and timings "
    "as returned; they hold for the given inputs only."
)


def check_code(
    code: Annotated[
        str, Field(description="Python source to scan: a script, module or notebook cell.")
    ],
    filename: Annotated[
        str, Field(description="Name shown in findings; use the real path if there is one.")
    ] = "snippet.py",
) -> dict[str, Any]:
    """Scan Python source for slow pandas patterns (iterrows loops, row-wise apply, concat in a
    loop, cell-by-cell writes, groupby-apply aggregates) without running it.

    Use this first, to find what is worth rewriting. Each finding has a rule code (see
    list_rules), a line, a severity, a message and often a suggested rewrite. Suggestions are
    not verified: before telling the user a rewrite is equivalent or faster, call
    verify_rewrite on representative inputs. Read-only: the code is parsed, never executed.
    """
    from pandas_doctor.static.engine import check_source

    result = check_source(code, filename)
    return {
        "findings": [f.to_dict() for f in result.findings],
        "error": result.error,
        "note": "Suggested rewrites are not verified; use verify_rewrite on representative data.",
    }


def verify_rewrite(
    original: Annotated[
        str,
        Field(
            description="The pandas code to replace, e.g. an iterrows loop. It can use the "
            "variables setup creates."
        ),
    ],
    candidate: Annotated[
        str, Field(description="The rewrite that should produce the same variables as original.")
    ],
    setup: Annotated[
        str,
        Field(
            description="Code run once to build representative inputs, e.g. a DataFrame "
            "with realistic size and dtypes. It may read files and use seeded randomness. "
            "Leave empty if original builds its own inputs."
        ),
    ] = "",
    outputs: Annotated[
        list[str] | None,
        Field(
            description="Names of the variables to compare. Default: the variables "
            "original assigns."
        ),
    ] = None,
    rtol: Annotated[
        float,
        Field(
            description="Relative tolerance for floats. Results that differ by no more "
            "than this get within_tolerance instead of different.",
            ge=0,
        ),
    ] = 1e-9,
) -> dict[str, Any]:
    """Check that a pandas rewrite gives the same result as the original, and measure whether
    it is faster.

    Use this before telling the user a rewrite is equivalent or faster; use check_code to find
    candidates. Runs setup once, then original and candidate on separate copies of its
    variables, compares each output strictly (type, dtype, index, names, NaN positions, exact
    values) and times both (median of repeated runs, about 2 s budget).

    Side effects: executes the given code locally. original and candidate must be pure
    computation (no file, network, subprocess or global-state calls, no unseeded randomness);
    otherwise nothing runs and `refused` says why. The verdict holds for these inputs only.

    `verdict` is "identical"; "scalar_types" (every value equal, but some scalars differ in
    type, e.g. numpy.float64 vs float, listed in details.scalar_types; not identical);
    "within_tolerance" (floats differ by at most rtol, largest difference reported); or
    "different" (with where).
    """
    from pandas_doctor.verify.api import verify_code

    result = verify_code(original, candidate, setup=setup, outputs=outputs, rtol=rtol, budget=2.0)
    return result.to_dict()


def list_rules() -> list[dict[str, str]]:
    """List the slow pandas patterns pandas-doctor knows, with their rule codes.

    Use this to look up a code from a check_code finding (PDD001 ...) or to see which patterns
    are covered. `fix` says whether check_code can rewrite the pattern safely, only as a
    suggestion, or only at runtime (`pandas-doctor run`). Read-only, no arguments.
    """
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

    from mcp.types import ToolAnnotations

    # by alias: the field names are camelCase in mcp 1.x and snake_case in 2.x
    hints = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
    read_only = ToolAnnotations.model_validate(hints)
    runs_code = ToolAnnotations.model_validate({**hints, "destructiveHint": False})
    server = Server("pandas-doctor", instructions=INSTRUCTIONS)
    server.tool(title="Find slow pandas code", annotations=read_only)(check_code)
    server.tool(title="Verify a pandas rewrite", annotations=runs_code)(verify_rewrite)
    server.tool(title="List pandas-doctor rules", annotations=read_only)(list_rules)
    return server


def main() -> None:
    build_server().run()
