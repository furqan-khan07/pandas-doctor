# Changelog

## 0.1.2 (2026-11-12)

- MCP tools have titles, read-only/idempotent annotations, a description for every parameter, and
  descriptions that say when to use each tool and what `verify_rewrite` runs.
- The Claude Code plugin is a skill only; set up the MCP server with `claude mcp add` (see README).
- Dependabot and CodeQL.

## 0.1.1 (2026-11-12)

- Listed in the MCP Registry as `io.github.furqan-khan07/pandas-doctor` (`server.json`), with setup snippets for
  Claude Code, Cursor and other MCP clients in the README.
- A real-data example: `examples/taxi/features.py` on a month of NYC taxi trips.
- A Colab notebook to try it in the browser (`examples/try_it.ipynb`).
- `%%doctor --rewrite` no longer tells you to use `--rewrite`; it says the rewrite is in the new cell.
- `llms.txt` for AI tools that read it.

## 0.1.0 (2026-11-12)

First release.

- `pandas-doctor check`: static rules for `.py` files and notebooks, with text/JSON output,
  `--diff`, `--fix` and `--unsafe-fixes`, and config in `[tool.pandas-doctor]`. Findings are
  warnings or info; info (an `iterrows` loop with no vectorised rewrite) never fails the check.
- `pandas-doctor run script.py`: runs the script and checks each rewrite on its data
  (identical / equal values with different scalar types / equal within tolerance /
  different). It times both versions, applies only rewrites that are verified *and*
  faster, and has `--diff`, `--fix`, `--json`, `--rtol` and `--strict-types`.
- `pandas-doctor verify`: check any original/rewrite pair on given inputs.
- `%%doctor` notebook magic, with `--rewrite`.
- Rules:
  - PDD001: iterrows loops.
  - PDD002: row-wise apply.
  - PDD003: concat in a loop, an unsafe fix.
  - PDD004: element-wise apply/map.
  - PDD005: groupby-apply aggregates.
  - PDD007: cell-by-cell loops.
  - Runtime-only PDD006 (astype loops) and PDD008 (category memory).
- Integrations: pre-commit hook, GitHub Action, MCP server (`pandas-doctor mcp`) and a Claude Code
  plugin.
