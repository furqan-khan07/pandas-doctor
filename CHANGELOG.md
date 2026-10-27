# Changelog

## 0.1.0 (unreleased)

First release.

- `pandas-doctor check`: static rules for `.py` files and notebooks, with text/JSON output,
  `--diff`, `--fix` and `--unsafe-fixes`, and config in `[tool.pandas-doctor]`.
- `pandas-doctor run script.py`: runs the script and checks each rewrite on its data
  (identical / equal within tolerance / different). It times both versions, applies only
  rewrites that are identical *and* faster, and has `--diff`, `--fix` and `--json`.
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
