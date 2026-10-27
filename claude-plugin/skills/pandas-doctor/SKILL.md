---
name: pandas-doctor
description: Find slow pandas code (iterrows loops, row-wise apply, concat in a loop, cell-by-cell writes, groupby-apply aggregates) and prove that a faster rewrite gives the identical result before claiming it. Use when optimizing pandas code, reviewing pandas performance, or before telling the user a pandas rewrite is equivalent or faster.
---

# pandas-doctor

pandas-doctor checks pandas code for known slow patterns and, more importantly, **proves rewrites on real data**: it runs the original and the rewrite on the same inputs, compares the results strictly (dtypes, index, names, NaN positions, exact values), and times both.

## When rewriting pandas code for speed

1. Find the slow spots:

   ```bash
   uvx pandas-doctor check path/to/code.py --format json
   ```

   Each finding has a `code` (PDD001 ...), a `message` and often a `suggestion`. Suggestions are **not verified** yet.

2. Before claiming a rewrite is equivalent or faster, verify it on representative inputs:

   ```bash
   uvx pandas-doctor verify --format json \
     --setup-code 'import pandas as pd, numpy as np; rng = np.random.default_rng(0); df = pd.DataFrame({"price": rng.random(100_000), "qty": rng.integers(1, 9, 100_000)})' \
     --original-code 'for i, r in df.iterrows():
    df.at[i, "t"] = r["price"] * r["qty"]' \
     --candidate-code 'df["t"] = df["price"] * df["qty"]'
   ```

   If the MCP server is connected, call the `verify_rewrite` tool instead.

3. Report what it says, as it says it:
   - `Identical`: say "identical on this data", with the measured times.
   - `WithinTolerance`: say the floats match to the reported relative difference (usually summation order). Don't call it identical.
   - `Different`: don't apply the rewrite as an optimisation. Its `reason` says what changed, often a dtype (iterrows turns ints into floats) or numpy vs Python scalars.

4. If the user can run their script, `uvx pandas-doctor run script.py` checks every rewrite on the script's own data and prints a patch with `--diff`.

## Rules

- Never state a speedup that pandas-doctor didn't measure, and never call a rewrite "safe" or "identical" if it wasn't verified.
- A rewrite that is identical but not faster isn't an optimisation; leave the code alone.
- `verify` refuses code that touches files, the network, subprocesses or randomness. Build inputs in the setup code instead.
