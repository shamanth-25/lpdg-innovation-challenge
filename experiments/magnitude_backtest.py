import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt"
]

# Weight individual telemetry metrics
METRIC_WEIGHTS = {
    "offline_duration_sec": 2,
    "disconnection_cnt": 1,
    "reboot_cnt": 1
}

SIGMA = 3

BASELINE_DAYS = 28
RECENT_DAYS = 7

# Meter risk weight
METER_WEIGHT = 1

TEST_WEEKS = [
    "2025-09-01",
    "2025-10-06",
    "2025-11-03",
    "2025-12-01",
    "2026-01-05"
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

telemetry = pd.read_parquet(
    "data/telemetry",
    columns=[
        "gateway_id",
        "ts_utc",
        *METRICS
    ]
)

meter = pd.read_csv(
    "data/meter_read_success.csv"
)

visits = pd.read_csv(
    "data/field_visits.csv"
)


# ============================================================
# NORMALIZE GATEWAY IDS
# ============================================================

telemetry["gateway_id"] = (
    telemetry["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

meter["gateway_id"] = (
    meter["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

visits["gateway_id"] = (
    visits["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)


# ============================================================
# DATES
# ============================================================

telemetry["ts"] = pd.to_datetime(
    telemetry["ts_utc"],
    utc=True
)

meter["week_start"] = pd.to_datetime(
    meter["week_start"]
)

visits["visited_on"] = pd.to_datetime(
    visits["visited_on"]
)


# ============================================================
# REAL FAULT
# ============================================================

visits["fault"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)


# ============================================================
# FUNCTION: ANOMALY MAGNITUDE SCORE
# ============================================================

def sigma_scores(monday):

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
                "sigma_score"
            ]
        )

    # --------------------------------------------------------
    # Gateway-specific mean and standard deviation
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

    recent["magnitude_score"] = 0.0

    # --------------------------------------------------------
    # Calculate anomaly magnitude
    # --------------------------------------------------------

    for metric in METRICS:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = recent["gateway_id"].map(
            stats[(metric, "std")]
        )

        # Prevent division by zero
        std = std.replace(
            0,
            np.nan
        )

        # Z-score
        z = (
            recent[metric] - mean
        ) / std

        # Only count the amount beyond 3 sigma
        excess = (
            z - SIGMA
        ).clip(lower=0)

        # Apply metric-specific weight
        recent["magnitude_score"] += (
            excess *
            METRIC_WEIGHTS[metric]
        )

    # --------------------------------------------------------
    # Sum magnitude by gateway
    # --------------------------------------------------------

    result = (
        recent
        .groupby("gateway_id")["magnitude_score"]
        .sum()
        .reset_index()
    )

    result.columns = [
        "gateway_id",
        "sigma_score"
    ]

    return result


# ============================================================
# FUNCTION: METER RISK
# ============================================================

def meter_scores(monday):

    start = pd.Timestamp(monday)

    # Only use meter information available BEFORE
    # the prediction week.
    available = meter[
        meter["week_start"] < start
    ].copy()

    if available.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "meter_risk"
            ]
        )

    # --------------------------------------------------------
    # Most recent meter week available before prediction
    # --------------------------------------------------------

    latest_week = available["week_start"].max()

    latest = available[
        available["week_start"] == latest_week
    ].copy()

    # --------------------------------------------------------
    # Meter success rate
    # --------------------------------------------------------

    latest["success_rate"] = np.where(
        latest["meters_expected"] > 0,
        latest["meters_read"] /
        latest["meters_expected"],
        1.0
    )

    # --------------------------------------------------------
    # Convert success rate to risk
    #
    # 1.0 success -> 0 risk
    # 0.5 success -> 0.5 risk
    # 0.0 success -> 1 risk
    # --------------------------------------------------------

    latest["meter_risk"] = (
        1 -
        latest["success_rate"]
    ).clip(
        lower=0,
        upper=1
    )

    return latest[
        [
            "gateway_id",
            "meter_risk"
        ]
    ]


# ============================================================
# FUNCTION: ACTUAL FAULT GATEWAYS
# ============================================================

def actual_faults(monday):

    start = pd.Timestamp(monday)
    end = start + pd.Timedelta(days=7)

    faults = visits[
        (visits["visited_on"] >= start)
        &
        (visits["visited_on"] < end)
        &
        (visits["fault"] == 1)
    ]

    return set(
        faults["gateway_id"]
    )


# ============================================================
# BACKTEST
# ============================================================

results = []

total_hits = 0
total_predictions = 0


for week in TEST_WEEKS:

    print()
    print("=" * 50)
    print(
        "Testing:",
        week
    )
    print("=" * 50)

    # --------------------------------------------------------
    # Telemetry magnitude score
    # --------------------------------------------------------

    sigma = sigma_scores(
        week
    )

    # --------------------------------------------------------
    # Meter risk
    # --------------------------------------------------------

    meter_risk = meter_scores(
        week
    )

    # --------------------------------------------------------
    # Merge scores
    # --------------------------------------------------------

    scores = sigma.merge(
        meter_risk,
        on="gateway_id",
        how="left"
    )

    scores["meter_risk"] = (
        scores["meter_risk"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Normalize telemetry score
    #
    # This keeps the meter contribution comparable.
    # --------------------------------------------------------

    if len(scores) > 0:

        max_sigma = (
            scores["sigma_score"].max()
        )

        if (
            pd.notna(max_sigma)
            and max_sigma > 0
        ):
            scores["sigma_norm"] = (
                scores["sigma_score"] /
                max_sigma *
                15
            )
        else:
            scores["sigma_norm"] = 0.0

    else:
        scores["sigma_norm"] = 0.0

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    scores["combined_score"] = (
        scores["sigma_norm"]
        +
        METER_WEIGHT *
        scores["meter_risk"]
    )

    # --------------------------------------------------------
    # Top 15 gateways
    # --------------------------------------------------------

    predictions = (
        scores
        .sort_values(
            "combined_score",
            ascending=False
        )
        .head(15)
        .copy()
    )

    predicted_gateways = set(
        predictions["gateway_id"]
    )

    # --------------------------------------------------------
    # Actual faults
    # --------------------------------------------------------

    actual = actual_faults(
        week
    )

    # --------------------------------------------------------
    # Hits
    # --------------------------------------------------------

    hits = len(
        predicted_gateways &
        actual
    )

    hit_rate = (
        hits / 15 * 100
    )

    print(
        f"Predicted: 15 | "
        f"Actual: {len(actual)} | "
        f"Hits: {hits} | "
        f"Hit rate: {hit_rate:.1f}%"
    )

    total_hits += hits
    total_predictions += 15

    results.append({
        "week": week,
        "predictions": 15,
        "actual_faults": len(actual),
        "hits": hits,
        "hit_rate": round(
            hit_rate,
            1
        )
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_hit_rate = (
    total_hits /
    total_predictions *
    100
)

print()
print()
print(
    "Magnitude Backtest Results"
)
print(
    "=========================="
)

print(
    results_df.to_string(
        index=False
    )
)

print()
print("Overall:")
print(
    f"Total predictions: "
    f"{total_predictions}"
)

print(
    f"Total hits: "
    f"{total_hits}"
)

print(
    f"Overall hit rate: "
    f"{overall_hit_rate:.1f}%"
)

print()
print(
    "Current benchmark: 21.3%"
)

if overall_hit_rate > 21.3:

    print(
        "RESULT: MAGNITUDE MODEL IS BETTER"
    )

elif overall_hit_rate < 21.3:

    print(
        "RESULT: CURRENT 21.3% MODEL IS STILL BETTER"
    )

else:

    print(
        "RESULT: SAME AS CURRENT MODEL"
    )
