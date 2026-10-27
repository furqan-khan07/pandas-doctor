"""A small feature-engineering script with the slow patterns pandas-doctor looks for.

Run `pandas-doctor check examples/` to see them, or `pandas-doctor run
examples/slow_features.py` to have each rewrite checked and timed on the data.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 20_000
orders = pd.DataFrame(
    {
        "price": rng.uniform(1, 500, n).round(2),
        "qty": rng.integers(1, 20, n).astype("float64"),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "coupon": rng.choice(["", "SAVE10", "FREESHIP"], n),
    }
)

# revenue per order
for i, row in orders.iterrows():
    orders.at[i, "revenue"] = row["price"] * row["qty"]

# order size bucket
orders["size"] = orders.apply(lambda r: "bulk" if r["qty"] >= 10 else "single", axis=1)

# total discounted revenue
total = 0.0
for _, row in orders.iterrows():
    if row["coupon"] == "SAVE10":
        total += row["revenue"] * 0.9

# one summary frame per region, stacked
summary = pd.DataFrame()
for region, group in orders.groupby("region"):
    stats = pd.DataFrame({"region": [region], "orders": [len(group)], "revenue": [group["revenue"].sum()]})
    summary = pd.concat([summary, stats], ignore_index=True)

print(summary)
print(f"discounted revenue: {total:,.2f}")
