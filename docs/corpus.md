# Corpus check

Before release, `pandas-doctor check` was run over 13 public repositories: notebooks, tutorials, and two large libraries. The goal was to find false positives and wrong suggestions. None of their code was executed. Only the static checker ran, which parses files and never imports or runs them.

| Repository | Commit | Files | Findings | Unparseable | Time |
|---|---|---:|---:|---:|---:|
| Kaggle/learntools | e9ba66d | 496 | 23 | 0 | 2.3 s |
| TomAugspurger/effective-pandas | bb3f490 | 10 | 2 | 0 | 0.2 s |
| ageron/handson-ml3 | ef1c9f5 | 29 | 0 | 0 | 0.7 s |
| donnemartin/data-science-ipython-notebooks | 5b3c00d | 168 | 3 | 19 | 0.7 s |
| guipsamora/pandas_exercises | daf0fd2 | 85 | 1 | 0 | 0.2 s |
| jakevdp/PythonDataScienceHandbook | d662314 | 146 | 2 | 2 | 0.3 s |
| jvns/pandas-cookbook | 69d3389 | 11 | 0 | 0 | 0.0 s |
| microsoft/Data-Science-For-Beginners | 4d2ac427 | 1,405 | 280 | 0 | 7.3 s |
| microsoft/ML-For-Beginners | de2d4e12 | 2,874 | 112 | 0 | 7.2 s |
| pandas-dev/pandas | cd83fbf | 1,547 | 133 | 0 | 75.9 s |
| statsmodels/statsmodels | d04ce4d | 1,336 | 15 | 1 | 32.9 s |
| wesm/pydata-book | a892791 | 16 | 1 | 0 | 0.3 s |
| ydataai/ydata-profiling | 98b1aba | 296 | 2 | 0 | 1.1 s |
| **Total** | | **8,419** | **574** | **22** | **129 s** |

The two Microsoft courses ship the same notebooks in many translations, so most of their findings are repeats. The 22 unparseable files are Python 2 code. The pandas repository is slow mostly because of a few test files over 10,000 lines.

## What was reviewed

- **Every suggested rewrite (19 unique).** Each was checked by hand against the code it replaces. All 19 compute the same values. Some may still differ in dtype or name, which is what `run` checks; every suggestion says so.
- **Every PDD003 (concat in a loop) and PDD007 (cell-by-cell loop) finding.**
- **A random sample of 25 of the 135 unique PDD001/PDD002 findings that have no suggestion.** All were real `iterrows` loops or `apply(axis=1)` calls.

`--fix --unsafe-fixes` would have changed **0 files**. The only automatic fix (PDD003) was refused in all 3 places, each for a stated reason: the loop variable is reused, or `axis=` isn't a literal. So there are no wrong automatic fixes.

## False positives found and fixed

The first pass found these. Each one is now a regression test.

1. **`.map(lambda ...)` on things that aren't a pandas column.** These got element-wise suggestions:
   - `Styler.map` in pandas' own style code
   - TensorFlow `dataset.map(lambda x: x * 2)`
   - `Index.map` (which led to `.dt` on a `DatetimeIndex`)
   - `DataFrame.map`
   - identity lambdas

   PDD004 now only trusts `.map` on an obvious column (`df["col"]`, `df.col`). It skips `.style`, `.index`, groupby-like variables and identity lambdas. (41 findings removed.)
2. **statsmodels' `res = res.append(data, refit=False)`.** This was reported as the removed `DataFrame.append`. PDD003 now only matches `.append` calls with DataFrame.append's arguments. (2 removed.)
3. **Cell writes in small fixed loops.** For example `for p in range(6): aic.iloc[p, q] = fit(...)`, where fitting a model is the cost, not the write. PDD007 now needs the loop to go over the written frame's rows: `df.index`, `range(len(df))`, `range(df.shape[0])`. (11 removed.)
4. **Empty notebooks and notebooks for other kernels (R)** were reported as unreadable. They're skipped now. (336 error lines removed.)

## Reproduce

```bash
git clone --depth 1 https://github.com/<repo>.git corpus/<name>   # for each repo above
pandas-doctor check corpus/<name> --format json --exit-zero
```
