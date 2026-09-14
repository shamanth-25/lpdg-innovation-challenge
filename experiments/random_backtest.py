import pandas as pd
import numpy as np

# Load field visits
visits = pd.read_csv("data/field_visits.csv")

# Convert outcome into a real fault
visits["fault"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)

# Normalize gateway IDs
visits["gateway_id"] = (
    visits["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

# Historical test weeks
test_mondays = [
    "2025-09-01",
    "2025-10-06",
    "2025-11-03",
    "2025-12-01",
    "2026-01-05"
]

# All gateways
all_gateways = visits["gateway_id"].unique()

rng = np.random.default_rng(42)

total_hits = 0
total_predictions = 0

for monday in test_mondays:

    start = pd.Timestamp(monday)
    end = start + pd.Timedelta(days=7)

    # Actual faults during this week
    actual = visits[
        (visits["visited_on"].astype(str) >= str(start.date()))
        &
        (visits["visited_on"].astype(str) < str(end.date()))
        &
        (visits["fault"] == 1)
    ]

    actual_faults = set(actual["gateway_id"])

    # Randomly choose 15 gateways
    predicted = rng.choice(
        all_gateways,
        size=15,
        replace=False
    )

    hits = sum(
        gateway in actual_faults
        for gateway in predicted
    )

    total_hits += hits
    total_predictions += 15

    print(
        monday,
        "hits:",
        hits,
        "/ 15",
        "actual faults:",
        len(actual_faults)
    )

print()
print("Random benchmark")
print("----------------")
print("Total predictions:", total_predictions)
print("Total hits:", total_hits)
print(
    "Hit rate:",
    round(
        total_hits / total_predictions * 100,
        1
    ),
    "%"
)
