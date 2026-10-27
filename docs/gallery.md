# Speedup gallery

These are slow lines pandas-doctor flagged in public notebooks during the [corpus check](corpus.md). Each one is re-run here as written, next to the rewrite pandas-doctor suggests, and compared with the same strict check `run` uses.

A few caveats:
- The data is synthetic and shaped like the notebook's (row counts and column types). That way the gallery never downloads anything, and anyone can reproduce it with `python benchmarks/gallery.py`.
- Most of these come from teaching notebooks. Several show the slow version on purpose before the fast one. They're here because they're the patterns people learn, not to criticise anyone's code.
- Numbers are from [`benchmarks/results/gallery-pandas-3.1.0.json`](../benchmarks/results/gallery-pandas-3.1.0.json) (Apple M4 Pro, Python 3.13.5, pandas 3.1.0, numpy 2.5.4, no pyarrow).

| Where | Pattern | Rows | Before | After | Speedup | Result |
|---|---|---:|---:|---:|---:|---|
| [Kaggle Learn pandas course, exercise 2](https://github.com/Kaggle/learntools/blob/e9ba66dc89/notebooks/pandas/raw/ex_2.ipynb) | row-wise apply (PDD002) | 129,971 | 478 ms | 91 µs | 5,261x | identical |
| [Kaggle Learn pandas course, tutorial 2](https://github.com/Kaggle/learntools/blob/e9ba66dc89/notebooks/pandas/raw/tut_2.ipynb) | map with a lambda (PDD004) | 129,971 | 11.1 ms | 67 µs | 167x | identical |
| [Python Data Science Handbook, seaborn chapter](https://github.com/donnemartin/data-science-ipython-notebooks/blob/5b3c00d462/matplotlib/04.14-Visualization-With-Seaborn.ipynb) | map with a lambda (PDD004) | 37,250 | 2.8 ms | 98 µs | 29x | identical |
| [pandas_exercises, time series](https://github.com/guipsamora/pandas_exercises/blob/daf0fd2c7c/09_Time_Series/Getting_Financial_Data/Exercises_solutions.ipynb) | map on timestamps (PDD004) | 1,300 | 386 µs | 89 µs | 4.4x | identical |
| [Data Science for Beginners, data preparation](https://github.com/microsoft/Data-Science-For-Beginners/blob/4d2ac427ad/2-Working-With-Data/08-data-preparation/notebook.ipynb) | apply with if/else (PDD004) | 100,000 | 6.1 ms | 190 µs | 32x | identical |
| [Data Science for Beginners, working with data](https://github.com/microsoft/Data-Science-For-Beginners/blob/4d2ac427ad/2-Working-With-Data/07-python/notebook.ipynb) | apply len() on strings (PDD004) | 100,000 | 17.4 ms | 2.7 ms | 6.4x | identical |
| [Effective pandas (Tom Augspurger), tidy data](https://github.com/TomAugspurger/effective-pandas/blob/bb3f49066b/modern_5_tidy.ipynb) | groupby.apply aggregating (PDD005) | 2,460 | 1.3 ms | 556 µs | 2.4x | identical |

## Something the gallery turned up

The first version of the wine-reviews data had a few prices of exactly 0. The original `reviews.apply(bargainness, axis='columns')` then crashed with `ZeroDivisionError`, while `reviews["points"] / reviews["price"]` quietly returned `inf`.

The cause: the frame has a text column, so each row `apply` builds is `object` dtype, and `row.points / row.price` is plain Python division. Same data, different behaviour. That's exactly the kind of thing a rewrite shouldn't change silently. `verify` and `run` refuse to compare when the original fails, so nothing was claimed. (The real dataset's cheapest wine is $4, so the gallery uses that floor.)
