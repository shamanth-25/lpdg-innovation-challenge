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

# Give more importance to offline duration
METRIC_WEIGHTS = {
    "offline_duration_sec": 2,
    "disconnection_cnt": 1,
    "reboot_cnt": 1
}

SIGMA = 3

BASELINE_DAYS = 28
RECENT_DAYS = 7

# Meter signal is useful, based on previous experiment
METER_WEIGHT = 1

# Test weeks
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
# NORMALIZE IDS
# ============================================================

for df in [telemetry, meter, visits]:

    df["gateway_id"] = (
        df["gateway_id"]
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
# ACTUAL FAULT
# ============================================================

visits["fault"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)


# ============================================================
# SIGMA / ANOMALY FEATURES
# ============================================================

def anomaly_features(monday):

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
        return pd.DataFrame()

    # --------------------------------------------------------
    # Gateway-specific baseline
    # --------------------------------------------------------

    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # --------------------------------------------------------
    # Recent 7 days
    # --------------------------------------------------------

    recent = window[
        window["ts"] >=
        end - pd.Timedelta(days=RECENT_DAYS)
    ].copy()

    if recent.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Calculate weighted anomaly score
    # --------------------------------------------------------

    recent["anomaly_score"] = 0.0

    # Count abnormal metrics
    recent["abnormal_count"] = 0

    for metric in METRICS:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = recent["gateway_id"].map(
            stats[(metric, "std")]
        )

        # Avoid zero standard deviation
        std = std.replace(0, np.nan)

        # Z-score
        z = (
            recent[metric] - mean
        ) / std

        z = z.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        # Only positive abnormal values
        positive_z = z.clip(lower=0)

        # Weighted anomaly
        recent["anomaly_score"] += (
            positive_z *
            METRIC_WEIGHTS[metric]
        )

        # 3-sigma breach
        exceeded = (
            z > SIGMA
        )

        recent["abnormal_count"] += (
            exceeded.astype(int)
        )

    # --------------------------------------------------------
    # Daily / hourly persistence
    # --------------------------------------------------------

    recent["abnormal"] = (
        recent["abnormal_count"] > 0
    ).astype(int)

    persistence = (
        recent
        .groupby("gateway_id")["abnormal"]
        .sum()
        .reset_index()
    )

    persistence.columns = [
        "gateway_id",
        "abnormal_hours"
    ]

    # --------------------------------------------------------
    # Maximum consecutive abnormal records
    # --------------------------------------------------------

    recent = recent.sort_values(
        ["gateway_id", "ts"]
    )

    recent["group"] = (
        recent.groupby("gateway_id")["abnormal"]
        .transform(lambda x: x.ne(x.shift()).cumsum())
    )

    consecutive = (
        recent[recent["abnormal"] == 1]
        .groupby(
            ["gateway_id", "group"]
        )
        .size()
        .groupby(level=0)
        .max()
        .reset_index()
    )

    consecutive.columns = [
        "gateway_id",
        "max_consecutive"
    ]

    # --------------------------------------------------------
    # Total anomaly magnitude
    # --------------------------------------------------------

    magnitude = (
        recent
        .groupby("gateway_id")["anomaly_score"]
        .sum()
        .reset_index()
    )

    magnitude.columns = [
        "gateway_id",
        "anomaly_magnitude"
    ]

    # --------------------------------------------------------
    # Maximum anomaly magnitude
    # --------------------------------------------------------

    max_magnitude = (
        recent
        .groupby("gateway_id")["anomaly_score"]
        .max()
        .reset_index()
    )

    max_magnitude.columns = [
        "gateway_id",
        "max_anomaly"
    ]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = persistence.merge(
        consecutive,
        on="gateway_id",
        how="left"
    )

    result = result.merge(
        magnitude,
        on="gateway_id",
        how="left"
    )

    result = result.merge(
        max_magnitude,
        on="gateway_id",
        how="left"
    )

    result = result.fillna(0)

    return result


# ============================================================
# METER RISK
# ============================================================

def meter_scores(monday):

    start = pd.Timestamp(monday)

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

    # Most recent meter observation
    latest_week = (
        available["week_start"]
        .max()
    )

    latest = available[
        available["week_start"] == latest_week
    ].copy()

    latest["meters_expected"] = (
        pd.to_numeric(
            latest["meters_expected"],
            errors="coerce"
        )
    )

    latest["meters_read"] = (
        pd.to_numeric(
            latest["meters_read"],
            errors="coerce"
        )
    )

    latest["success_rate"] = (
        latest["meters_read"] /
        latest["meters_expected"].replace(
            0,
            np.nan
        )
    )

    latest["success_rate"] = (
        latest["success_rate"]
        .fillna(0)
        .clip(0, 1)
    )

    latest["meter_risk"] = (
        1 - latest["success_rate"]
    )

    return latest[
        [
            "gateway_id",
            "meter_risk"
        ]
    ]


# ============================================================
# ACTUAL FAULTS
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
# NORMALIZE
# ============================================================

def normalize(series):

    maximum = series.max()

    if pd.isna(maximum) or maximum == 0:
        return pd.Series(
            0.0,
            index=series.index
        )

    return series / maximum


# ============================================================
# BUILD SCORES
# ============================================================

def build_scores(monday):

    anomalies = anomaly_features(monday)

    if anomalies.empty:
        return pd.DataFrame()

    meter_risk = meter_scores(monday)

    scores = anomalies.merge(
        meter_risk,
        on="gateway_id",
        how="left"
    )

    scores["meter_risk"] = (
        scores["meter_risk"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Normalize individual features
    # --------------------------------------------------------

    scores["abnormal_norm"] = normalize(
        scores["abnormal_hours"]
    )

    scores["consecutive_norm"] = normalize(
        scores["max_consecutive"]
    )

    scores["magnitude_norm"] = normalize(
        scores["anomaly_magnitude"]
    )

    scores["max_anomaly_norm"] = normalize(
        scores["max_anomaly"]
    )

    # --------------------------------------------------------
    # Advanced score
    #
    # Persistence is deliberately important.
    # Magnitude captures severity.
    # Meter risk adds independent evidence.
    # --------------------------------------------------------

    scores["advanced_score"] = (
        0.30 * scores["abnormal_norm"]
        +
        0.25 * scores["consecutive_norm"]
        +
        0.25 * scores["magnitude_norm"]
        +
        0.10 * scores["max_anomaly_norm"]
        +
        METER_WEIGHT * scores["meter_risk"]
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    scores = scores.sort_values(
        "advanced_score",
        ascending=False
    ).reset_index(drop=True)

    scores["rank"] = (
        np.arange(len(scores)) + 1
    )

    return scores


# ============================================================
# BACKTEST
# ============================================================

results = []

for week in TEST_WEEKS:

    print()
    print("=" * 50)
    print("Testing:", week)
    print("=" * 50)

    scores = build_scores(week)

    if scores.empty:
        print("No scores available")
        continue

    # Top 15
    predictions = scores.head(15)

    predicted_ids = set(
        predictions["gateway_id"]
    )

    actual_ids = actual_faults(week)

    hits = len(
        predicted_ids & actual_ids
    )

    hit_rate = (
        hits / 15 * 100
    )

    print(
        f"Predicted: 15 | "
        f"Actual: {len(actual_ids)} | "
        f"Hits: {hits} | "
        f"Hit rate: {hit_rate:.1f}%"
    )

    results.append({
        "week": week,
        "predictions": 15,
        "actual_faults": len(actual_ids),
        "hits": hits,
        "hit_rate": hit_rate
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print()
print()
print("Advanced Backtest Results")
print("==========================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "hit_rate": "{:.1f}".format
        }
    )
)

total_predictions = (
    results_df["predictions"].sum()
)

total_hits = (
    results_df["hits"].sum()
)

overall_rate = (
    total_hits /
    total_predictions *
    100
)

print()
print("Overall:")
print(
    f"Total predictions: {total_predictions}"
)

print(
    f"Total hits: {total_hits}"
)

print(
    f"Overall hit rate: {overall_rate:.1f}%"
)

print()
print("Current best model: 21.3%")
print(
    f"Advanced model: {overall_rate:.1f}%"
)

if overall_rate > 21.3:
    print("RESULT: IMPROVEMENT")
elif overall_rate == 21.3:
    print("RESULT: SAME")
else:
    print("RESULT: WORSE")

