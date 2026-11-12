"""The hero example on real data: NYC yellow taxi trips for January 2024 (2.96M rows).

Download the month from the NYC TLC (about 48 MB, needs pyarrow to read):

    curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
    python examples/taxi/features.py yellow_tripdata_2024-01.parquet
    pandas-doctor run examples/taxi/features.py yellow_tripdata_2024-01.parquet
"""

import sys

import pandas as pd

trips = pd.read_parquet(
    sys.argv[1],
    columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "payment_type",
    ],
)
trips["minutes"] = (
    trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# price per mile
trips["fare_per_mile"] = 0.0
for i, row in trips.iterrows():
    trips.at[i, "fare_per_mile"] = row["fare_amount"] / (row["trip_distance"] + 0.1)

# tip as a share of the fare
trips["tip_share"] = trips.apply(
    lambda r: r["tip_amount"] / r["fare_amount"] if r["fare_amount"] > 0 else 0.0, axis=1
)

# long trips
trips["long_trip"] = trips.apply(lambda r: "long" if r["minutes"] > 30 else "short", axis=1)

# total revenue from card payments (payment_type 1)
card_revenue = 0.0
for _, row in trips.iterrows():
    if row["payment_type"] == 1:
        card_revenue += row["fare_amount"] + row["tip_amount"]

# pickup hour
trips["hour"] = trips["tpep_pickup_datetime"].apply(lambda t: t.hour)

print(trips[["fare_per_mile", "tip_share", "long_trip", "hour"]].describe(include="all").round(3))
print(f"card revenue: {card_revenue:,.2f}")
