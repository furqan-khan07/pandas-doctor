"""Benchmark cases: each slow pattern next to the rewrite pandas-doctor produces for it.

Every case runs through `pandas_doctor.verify_code`, the same check `run` uses, so each
timing comes with a verdict on the same data. Data is synthetic and seeded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    rule: str
    setup: str  # `{n}` is replaced by the size
    original: str
    rewrite: str
    outputs: tuple[str, ...]
    sizes: tuple[int, ...] = (10_000, 100_000, 1_000_000)


ORDERS = """
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
n = {n}
df = pd.DataFrame({{
    "price": rng.uniform(1, 500, n).round(2),
    "qty": rng.integers(1, 20, n).astype("float64"),
    "cost": rng.uniform(1, 400, n).round(2),
}})
"""

ORDERS_MIXED = """
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
n = {n}
df = pd.DataFrame({{
    "price": rng.uniform(1, 500, n).round(2),
    "qty": rng.integers(1, 20, n),
    "name": rng.choice(["Ann ", " BOB", "cy", "Dee"], n),
    "k": rng.integers(0, 100, n),
    "x": rng.random(n),
    "d": pd.date_range("2024-01-01", periods=n, freq="min"),
}})
"""

CASES = [
    Case(
        "iterrows: write a computed column",
        "PDD001",
        ORDERS + 'df["revenue"] = 0.0\n',
        'for i, row in df.iterrows():\n    df.at[i, "revenue"] = row["price"] * row["qty"]',
        'df["revenue"] = df["price"] * df["qty"]',
        ("df",),
    ),
    Case(
        "iterrows: running total",
        "PDD001",
        ORDERS,
        'total = 0.0\nfor _, row in df.iterrows():\n    total += row["price"] * row["qty"]',
        'total = 0.0\ntotal = np.add.accumulate(np.concatenate(([total], np.asarray(df["price"] * df["qty"]))))[-1]',
        ("total",),
    ),
    Case(
        "iterrows: build a list",
        "PDD001",
        ORDERS,
        'out = []\nfor _, row in df.iterrows():\n    out.append(row["price"] - row["cost"])',
        'out = []\nout.extend(np.asarray(df["price"] - df["cost"]))',
        ("out",),
    ),
    Case(
        "apply(axis=1): if/else label",
        "PDD002",
        ORDERS,
        'df["size"] = df.apply(lambda r: "bulk" if r["qty"] >= 10 else "single", axis=1)',
        'df["size"] = pd.Series(np.where(df["qty"] >= 10, "bulk", "single"), index=df.index)',
        ("df",),
    ),
    Case(
        "apply(axis=1): arithmetic",
        "PDD002",
        ORDERS,
        'df["margin"] = df.apply(lambda r: r["price"] - r["cost"], axis=1)',
        'df["margin"] = df["price"] - df["cost"]',
        ("df",),
    ),
    Case(
        "concat in a loop (100-row parts)",
        "PDD003",
        ORDERS + "parts = [df.iloc[i:i + 100] for i in range(0, n, 100)]\n",
        "out = pd.DataFrame()\nfor p in parts:\n    out = pd.concat([out, p], ignore_index=True)",
        "out = pd.DataFrame()\nout_parts = []\nfor p in parts:\n    out_parts.append(p)\nif out_parts:\n    out = pd.concat([out, *out_parts], ignore_index=True)",
        ("out",),
        sizes=(10_000, 100_000),
    ),
    Case(
        "apply on strings -> .str",
        "PDD004",
        ORDERS_MIXED,
        'df["n"] = df["name"].apply(lambda x: x.strip().lower())',
        'df["n"] = df["name"].str.strip().str.lower()',
        ("df",),
    ),
    Case(
        "apply on timestamps -> .dt",
        "PDD004",
        ORDERS_MIXED,
        'df["yr"] = df["d"].apply(lambda t: t.year)',
        'df["yr"] = df["d"].dt.year.astype("int64")',
        ("df",),
    ),
    Case(
        "map with arithmetic",
        "PDD004",
        ORDERS_MIXED,
        'df["x2"] = df["x"].map(lambda v: v * 2)',
        'df["x2"] = df["x"] * 2',
        ("df",),
    ),
    Case(
        "groupby.apply -> built-in count",
        "PDD005",
        ORDERS_MIXED,
        's = df.groupby("k").apply(lambda g: g["x"].count())',
        's = df.groupby("k")["x"].count().rename(None)',
        ("s",),
    ),
    Case(
        "groupby.apply -> built-in sum",
        "PDD005",
        ORDERS_MIXED,
        's = df.groupby("k").apply(lambda g: g["x"].sum())',
        's = df.groupby("k")["x"].sum().rename(None)',
        ("s",),
    ),
    Case(
        "cell-by-cell loop over the index",
        "PDD007",
        ORDERS + 'df["z"] = 0.0\n',
        'for i in df.index:\n    df.at[i, "z"] = df.at[i, "price"] * 3',
        'df["z"] = df["price"] * 3',
        ("df",),
        sizes=(10_000, 100_000),
    ),
    Case(
        "per-column astype loop (100 x 2000, in place)",
        "PDD006",
        "import numpy as np\nimport pandas as pd\nrng = np.random.default_rng(1)\nwide = pd.DataFrame(rng.integers(0, 100, (100, {n})), columns=[f'c{{i}}' for i in range({n})])\ncols = list(wide.columns)\n",
        'for c in cols:\n    wide[c] = wide[c].astype("float32")',
        'wide[list(cols)] = wide.astype({c: "float32" for c in cols})[list(cols)]',
        ("wide",),
        sizes=(2_000,),
    ),
]
