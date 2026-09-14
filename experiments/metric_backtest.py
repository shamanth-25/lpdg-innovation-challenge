import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# SETTINGS
# ============================================================

ALL_METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt"
]

SIGMA = 3
BASELINE_DAYS = 28
RECENT_DAYS = 7

# Keep the best meter weight found so far
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
        *ALL_METRICS
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

    latest_week = available["week_start"].max()

    latest = available[
        available["week_start"] == latest_week
    ].copy()

    latest["meters_expected"] = pd.to_numeric(
        latest["meters_expected"],
        errors="coerce"
    )

    latest["meters_read"] = pd.to_numeric(
        latest["meters_read"],
        errors="coerce"
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
# TELEMETRY SCORE
# ============================================================

def telemetry_scores(monday, metrics):

    end = pd.Timestamp(
        monday,
        tz="UTC"
    )

    # Previous 28 days
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
                "telemetry_score"
            ]
        )

    # Gateway-specific baseline
    stats = (
        window
        .groupby("gateway_id")[metrics]
        .agg(["mean", "std"])
    )

    # Last 7 days
    recent = window[
        window["ts"] >=
        end - pd.Timedelta(days=RECENT_DAYS)
    ].copy()

    recent["telemetry_score"] = 0.0

    for metric in metrics:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = recent["gateway_id"].map(
            stats[(metric, "std")]
        )

        std = std.replace(
            0,
            np.nan
        )

        exceeded = (
            recent[metric] - mean
        ) > SIGMA * std

        exceeded = exceeded.fillna(False)

        recent["telemetry_score"] += (
            exceeded.astype(int)
        )

    result = (
        recent
        .groupby("gateway_id")["telemetry_score"]
        .sum()
        .reset_index()
    )

    return result


# ============================================================
# BUILD FINAL RANKING
# ============================================================

def build_scores(monday, metrics):

    telemetry_result = telemetry_scores(
        monday,
        metrics
    )

    if telemetry_result.empty:
        return pd.DataFrame()

    meter_result = meter_scores(monday)

    scores = telemetry_result.merge(
        meter_result,
        on="gateway_id",
        how="left"
    )

    scores["meter_risk"] = (
        scores["meter_risk"]
        .fillna(0)
    )

    # Meter risk is deliberately small compared
    # with telemetry anomaly counts.
    scores["final_score"] = (
        scores["telemetry_score"]
        +
        METER_WEIGHT *
        scores["meter_risk"]
    )

    scores = scores.sort_values(
        [
            "final_score",
            "telemetry_score",
            "meter_risk"
        ],
        ascending=False
    ).reset_index(drop=True)

    return scores


# ============================================================
# ACTUAL FAULTS
# ============================================================

def actual_faults(monday):

    start = pd.Timestamp(monday)

    end = (
        start +
        pd.Timedelta(days=7)
    )

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
# METRIC COMBINATIONS
# ============================================================

metric_sets = []

for r in range(1, len(ALL_METRICS) + 1):

    for combo in combinations(
        ALL_METRICS,
        r
    ):
        metric_sets.append(
            list(combo)
        )


# ============================================================
# BACKTEST EVERY COMBINATION
# ============================================================

all_results = []

for metrics in metric_sets:

    name = " + ".join(metrics)

    print()
    print("=" * 60)
    print("Metrics:", name)
    print("=" * 60)

    total_hits = 0
    total_predictions = 0

    for week in TEST_WEEKS:

        scores = build_scores(
            week,
            metrics
        )

        if scores.empty:
            continue

        predictions = scores.head(15)

        predicted_ids = set(
            predictions["gateway_id"]
        )

        actual_ids = actual_faults(
            week
        )

        hits = len(
            predicted_ids & actual_ids
        )

        total_hits += hits
        total_predictions += 15

        print(
            f"{week} | "
            f"hits: {hits} | "
            f"actual: {len(actual_ids)}"
        )

    hit_rate = (
        total_hits /
        total_predictions *
        100
    )

    print()
    print(
        f"TOTAL: {total_hits} / "
        f"{total_predictions}"
    )

    print(
        f"Hit rate: {hit_rate:.1f}%"
    )

    all_results.append({
        "metrics": name,
        "hits": total_hits,
        "predictions": total_predictions,
        "hit_rate": hit_rate
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

results = pd.DataFrame(
    all_results
)

results = results.sort_values(
    [
        "hit_rate",
        "hits"
    ],
    ascending=False
).reset_index(drop=True)


print()
print()
print("=" * 60)
print("FINAL METRIC COMPARISON")
print("=" * 60)

print(
    results.to_string(
        index=False,
        formatters={
            "hit_rate": "{:.1f}".format
        }
    )
)

print()
print("Current benchmark:")
print("Meter weight = 1")
print("Hit rate = 21.3%")

best = results.iloc[0]

print()
print("BEST METRIC SET:")
print(best["metrics"])
print(
    f"Hits: {int(best['hits'])} / "
    f"{int(best['predictions'])}"
)
print(
    f"Hit rate: {best['hit_rate']:.1f}%"
)

if best["hit_rate"] > 21.3:
    print("RESULT: NEW BEST MODEL")
elif best["hit_rate"] == 21.3:
    print("RESULT: SAME AS CURRENT BEST")
else:
    print("RESULT: CURRENT 21.3% MODEL IS STILL BETTER")

