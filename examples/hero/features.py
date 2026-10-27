"""Feature engineering on one million taxi trips, written the way a lot of notebooks are.

The data is synthetic (seeded), so anyone can reproduce the run:

    python examples/hero/features.py               # the slow way
    pandas-doctor run examples/hero/features.py    # same run, plus checked rewrites
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 1_000_000
trips = pd.DataFrame(
    {
        "pickup": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 3600, n), unit="s"),
        "miles": rng.gamma(2.0, 1.6, n).round(2),
        "minutes": rng.gamma(3.0, 5.0, n).round(1),
        "fare": rng.gamma(3.0, 6.0, n).round(2),
        "tip": rng.gamma(1.2, 2.0, n).round(2),
        "passengers": rng.integers(1, 6, n).astype("float64"),
        "payment": rng.choice(["card", "cash", "app"], n, p=[0.6, 0.25, 0.15]),
    }
)

# price per mile
trips["fare_per_mile"] = 0.0
for i, row in trips.iterrows():
    trips.at[i, "fare_per_mile"] = row["fare"] / (row["miles"] + 0.1)

# tip as a share of the fare
trips["tip_share"] = trips.apply(lambda r: r["tip"] / r["fare"] if r["fare"] > 0 else 0.0, axis=1)

# long trips
trips["long_trip"] = trips.apply(lambda r: "long" if r["minutes"] > 30 else "short", axis=1)

# total revenue from card payments
card_revenue = 0.0
for _, row in trips.iterrows():
    if row["payment"] == "card":
        card_revenue += row["fare"] + row["tip"]

# pickup month
trips["month"] = trips["pickup"].apply(lambda t: t.month)

print(trips[["fare_per_mile", "tip_share", "long_trip", "month"]].describe(include="all").round(3))
print(f"card revenue: {card_revenue:,.2f}")
