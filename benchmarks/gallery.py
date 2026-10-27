"""Speedup gallery: slow lines found in public notebooks, re-run on synthetic data.

Each entry links the notebook line pandas-doctor flagged during the corpus check
(docs/corpus.md). The original line is reproduced as written. The data is synthetic,
shaped like the notebook's (row counts and column types), because the gallery must not
download anything. The rewrite is the one pandas-doctor suggests (with the dtype
adjustment `run` picks, where needed), measured with `verify_code`.

    python benchmarks/gallery.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import machine

from pandas_doctor.verify.api import verify_code
from pandas_doctor.verify.timing import format_seconds, format_speedup


@dataclass(frozen=True)
class Entry:
    where: str
    link: str
    pattern: str
    setup: str
    original: str
    rewrite: str
    outputs: tuple[str, ...]
    rows: int


WINE = """
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 129_971  # rows in the Kaggle wine reviews dataset
price = rng.gamma(2.0, 18.0, n).round(0) + 4  # the real data's cheapest wine is $4
price[rng.random(n) < 0.07] = np.nan
reviews = pd.DataFrame({"points": rng.integers(80, 101, n), "price": price,
                        "country": rng.choice(["US", "France", "Italy", "Spain"], n)})
review_points_mean = reviews.points.mean()
def bargainness(row):
    return row.points / row.price
"""

ENTRIES = [
    Entry(
        "Kaggle Learn pandas course, exercise 2",
        "https://github.com/Kaggle/learntools/blob/e9ba66dc89/notebooks/pandas/raw/ex_2.ipynb",
        "row-wise apply (PDD002)",
        WINE,
        "bargain_scores = reviews.apply(bargainness, axis='columns')",
        'bargain_scores = reviews["points"] / reviews["price"]',
        ("bargain_scores",),
        129_971,
    ),
    Entry(
        "Kaggle Learn pandas course, tutorial 2",
        "https://github.com/Kaggle/learntools/blob/e9ba66dc89/notebooks/pandas/raw/tut_2.ipynb",
        "map with a lambda (PDD004)",
        WINE,
        "out = reviews.points.map(lambda p: p - review_points_mean)",
        "out = reviews.points - review_points_mean",
        ("out",),
        129_971,
    ),
    Entry(
        "Python Data Science Handbook, seaborn chapter",
        "https://github.com/donnemartin/data-science-ipython-notebooks/blob/5b3c00d462/matplotlib/04.14-Visualization-With-Seaborn.ipynb",
        "map with a lambda (PDD004)",
        "import numpy as np, pandas as pd\nrng = np.random.default_rng(1)\ndata = pd.DataFrame({'age': rng.integers(18, 80, 37_250)})  # marathon finishers",
        "data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))",
        "data['age_dec'] = 10 * (data.age // 10)",
        ("data",),
        37_250,
    ),
    Entry(
        "pandas_exercises, time series",
        "https://github.com/guipsamora/pandas_exercises/blob/daf0fd2c7c/09_Time_Series/Getting_Financial_Data/Exercises_solutions.ipynb",
        "map on timestamps (PDD004)",
        "import pandas as pd\nvol_week = pd.DataFrame({'dt': pd.date_range('2000-01-02', periods=1300, freq='W')})",
        "vol_week['year'] = vol_week['dt'].map(lambda x: x.year)",
        "vol_week['year'] = vol_week['dt'].dt.year.astype('int64')",
        ("vol_week",),
        1_300,
    ),
    Entry(
        "Data Science for Beginners, data preparation",
        "https://github.com/microsoft/Data-Science-For-Beginners/blob/4d2ac427ad/2-Working-With-Data/08-data-preparation/notebook.ipynb",
        "apply with if/else (PDD004)",
        "import numpy as np, pandas as pd\nrng = np.random.default_rng(2)\nages = rng.integers(-5, 130, 100_000).astype(float)\ndirty_data = pd.DataFrame({'age': ages})",
        "dirty_data['age_clean'] = dirty_data['age'].apply(lambda x: np.nan if x < 0 or x > 120 else x)",
        "dirty_data['age_clean'] = pd.Series(np.where((dirty_data['age'] < 0) | (dirty_data['age'] > 120), np.nan, dirty_data['age']), index=dirty_data['age'].index)",
        ("dirty_data",),
        100_000,
    ),
    Entry(
        "Data Science for Beginners, working with data",
        "https://github.com/microsoft/Data-Science-For-Beginners/blob/4d2ac427ad/2-Working-With-Data/07-python/notebook.ipynb",
        "apply len() on strings (PDD004)",
        "import numpy as np, pandas as pd\nrng = np.random.default_rng(3)\ndf = pd.DataFrame({'B': rng.choice(['I', 'like', 'to', 'use', 'Python'], 100_000)})",
        "df['LenB'] = df['B'].apply(lambda x: len(x))",
        "df['LenB'] = df['B'].str.len()",
        ("df",),
        100_000,
    ),
    Entry(
        "Effective pandas (Tom Augspurger), tidy data",
        "https://github.com/TomAugspurger/effective-pandas/blob/bb3f49066b/modern_5_tidy.ipynb",
        "groupby.apply aggregating (PDD005)",
        "import numpy as np, pandas as pd\nrng = np.random.default_rng(4)\nteams = [f't{i}' for i in range(30)]\nidx = pd.MultiIndex.from_product([teams, range(82)], names=['team', 'game'])\nwins = pd.DataFrame({'n_wins': rng.integers(0, 2, len(idx)), 'n_games': 1}, index=idx)",
        "r = wins.groupby(level='team', as_index=True).apply(lambda x: x.n_wins.sum() / x.n_games.sum())",
        "r = wins.groupby(level='team', as_index=True)['n_wins'].sum() / wins.groupby(level='team', as_index=True)['n_games'].sum()",
        ("r",),
        2_460,
    ),
]


def main() -> None:
    info = machine()
    rows = []
    print("| Where | Pattern | Rows | Before | After | Speedup | Result |")
    print("|---|---|---:|---:|---:|---:|---|")
    for e in ENTRIES:
        r = verify_code(
            e.original,
            e.rewrite,
            setup=e.setup,
            outputs=list(e.outputs),
            budget=3.0,
            trust_setup=True,
        )
        if r.original is None or r.candidate is None:
            print(f"| [{e.where}]({e.link}) | {e.pattern} | not measured: {r.refused} |")
            continue
        b, a = r.original.seconds, r.candidate.seconds
        ratio = format_speedup(b, a) if a <= b else f"{format_speedup(a, b)} slower"
        rows.append(
            {
                "where": e.where,
                "link": e.link,
                "pattern": e.pattern,
                "rows": e.rows,
                "before": b,
                "after": a,
                "verdict": r.verdict.describe(),
            }
        )
        print(
            f"| [{e.where}]({e.link}) | {e.pattern} | {e.rows:,} | {format_seconds(b)} | {format_seconds(a)} | {ratio} | {r.verdict.describe()} |"
        )
    arrow = "" if info["pyarrow"] == "not installed" else "-pyarrow"
    out = (
        Path(__file__).resolve().parent / "results" / f"gallery-pandas-{info['pandas']}{arrow}.json"
    )
    out.write_text(json.dumps({"machine": info, "entries": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
