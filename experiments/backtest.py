import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

# Load telemetry
telemetry = pd.read_parquet("data/telemetry")

# Load historical field visits
visits = pd.read_csv("data/field_visits.csv")

# Convert dates
telemetry["ts"] = pd.to_datetime(
    telemetry["ts_utc"],
    utc=True
)

visits["requested_on"] = pd.to_datetime(
    visits["requested_on"]
)

visits["visited_on"] = pd.to_datetime(
    visits["visited_on"]
)

# A real fault means the engineer fixed something
visits["fault"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)


print("Telemetry rows:", len(telemetry))
print(
    "Telemetry gateways:",
    telemetry["gateway_id"].nunique()
)
print("Field visits:", len(visits))
print(
    "Historical faults:",
    visits["fault"].sum()
)


# ============================================================
# 2. BASELINE SETTINGS
# ============================================================

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt"
]

BASELINE_DAYS = 28
RECENT_DAYS = 7
SIGMA = 3


# ============================================================
# 3. BASELINE RANKING FUNCTION
# ============================================================

def baseline_rank(telemetry, monday):
    """
    Reproduce the supplied 3-sigma baseline.

    For a given Monday:

    - use the previous 28 days as the baseline
    - calculate mean and standard deviation for each gateway
    - inspect the previous 7 days
    - flag values more than 3 standard deviations above
      the gateway's own baseline
    - rank gateways by number of flagged hours
    - return top 15
    """

    end = pd.Timestamp(
        monday,
        tz="UTC"
    )

    # --------------------------------------------------------
    # Previous 28 days
    # --------------------------------------------------------

    window = telemetry[
        (telemetry["ts"] >=
         end - pd.Timedelta(days=BASELINE_DAYS))
        &
        (telemetry["ts"] < end)
    ].copy()

    if window.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "score"
            ]
        )

    # --------------------------------------------------------
    # Calculate gateway-specific mean and std
    # --------------------------------------------------------

    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # --------------------------------------------------------
    # Last 7 days
    # --------------------------------------------------------

    recent = window[
        window["ts"] >=
        end - pd.Timedelta(days=RECENT_DAYS)
    ].copy()

    recent["flagged"] = 0

    # --------------------------------------------------------
    # Detect 3-sigma anomalies
    # --------------------------------------------------------

    for metric in METRICS:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = recent["gateway_id"].map(
            stats[(metric, "std")]
        )

        # If standard deviation is zero,
        # don't treat it as an anomaly.
        std = std.replace(
            0,
            np.nan
        )

        exceeded = (
            recent[metric] - mean
        ) > SIGMA * std

        exceeded = exceeded.fillna(False)

        recent["flagged"] += (
            exceeded.astype(int)
        )

    # --------------------------------------------------------
    # Count flagged hours per gateway
    # --------------------------------------------------------

    ranking = (
        recent
        .groupby("gateway_id")["flagged"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
        .reset_index()
    )

    ranking.columns = [
        "gateway_id",
        "score"
    ]

    return ranking


# ============================================================
# 4. HISTORICAL WEEKS TO TEST
# ============================================================

test_mondays = [
    "2025-09-01",
    "2025-10-06",
    "2025-11-03",
    "2025-12-01",
    "2026-01-05"
]


# ============================================================
# 5. RUN BACKTEST
# ============================================================

results = []

for test_monday in test_mondays:

    print(
        "\nTesting:",
        test_monday
    )

    # --------------------------------------------------------
    # Generate baseline top 15
    # --------------------------------------------------------

    ranking = baseline_rank(
        telemetry,
        test_monday
    )

    # --------------------------------------------------------
    # Following 7 days
    # --------------------------------------------------------

    start = pd.Timestamp(
        test_monday
    )

    end = (
        start +
        pd.Timedelta(days=7)
    )

    future_visits = visits[
        (visits["visited_on"] >= start)
        &
        (visits["visited_on"] < end)
    ].copy()

    # --------------------------------------------------------
    # Find gateways with real faults
    # --------------------------------------------------------

    actual_faults = set(
        future_visits.loc[
            future_visits["fault"] == 1,
            "gateway_id"
        ]
    )

    # --------------------------------------------------------
    # Normalize gateway IDs
    #
    # Telemetry:
    #   02A80292CB97
    #
    # Field visits:
    #   02:A8:02:92:CB:97
    # --------------------------------------------------------

    actual_faults = {
        str(gateway_id)
        .replace(":", "")
        .upper()
        for gateway_id in actual_faults
    }

    ranking["id_norm"] = (
        ranking["gateway_id"]
        .astype(str)
        .str.replace(
            ":",
            "",
            regex=False
        )
        .str.upper()
    )

    # --------------------------------------------------------
    # Check whether each predicted gateway
    # actually had a fault
    # --------------------------------------------------------

    ranking["actual_fault"] = (
        ranking["id_norm"]
        .isin(actual_faults)
    )

    # Number of correct predictions
    hits = int(
        ranking["actual_fault"].sum()
    )

    # Hit rate
    hit_rate = (
        hits / len(ranking) * 100
        if len(ranking) > 0
        else 0
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "week": test_monday,
        "predicted": len(ranking),
        "actual_faults": len(actual_faults),
        "hits": hits,
        "hit_rate": round(
            hit_rate,
            1
        )
    })

    # --------------------------------------------------------
    # Print this week's result
    # --------------------------------------------------------

    print(
        "Predicted gateways:",
        len(ranking)
    )

    print(
        "Actual fault gateways:",
        len(actual_faults)
    )

    print(
        "Correct predictions:",
        hits
    )

    print(
        "Hit rate:",
        round(hit_rate, 1),
        "%"
    )


# ============================================================
# 6. DISPLAY ALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\n\nHistorical Backtest Results"
)

print(
    "==========================="
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 7. OVERALL RESULTS
# ============================================================

total_predictions = (
    results_df["predicted"].sum()
)

total_actual_faults = (
    results_df["actual_faults"].sum()
)

total_hits = (
    results_df["hits"].sum()
)

overall_hit_rate = (
    total_hits /
    total_predictions *
    100
)


print("\nOverall:")
print(
    "Total predictions:",
    total_predictions
)

print(
    "Total actual fault gateways:",
    total_actual_faults
)

print(
    "Total hits:",
    total_hits
)

print(
    "Overall hit rate:",
    round(
        overall_hit_rate,
        1
    ),
    "%"
)
