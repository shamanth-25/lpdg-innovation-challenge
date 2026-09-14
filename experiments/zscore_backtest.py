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

BASELINE_DAYS = 28
RECENT_DAYS = 7

TOP_N = 15

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
# REAL FAULT
# ============================================================

visits["fault"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)


# ============================================================
# Z-SCORE MAGNITUDE
# ============================================================

def zscore_scores(monday):

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
                "zscore_score"
            ]
        )

    # --------------------------------------------------------
    # Gateway-specific mean/std
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
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "zscore_score"
            ]
        )

    recent["zscore_score"] = 0.0

    # --------------------------------------------------------
    # Calculate positive anomaly magnitude
    # --------------------------------------------------------

    for metric in METRICS:

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

        z = (
            recent[metric] - mean
        ) / std

        # Only count positive anomalies.
        # Normal values below the baseline contribute 0.
        positive_z = (
            z - 3
        ).clip(lower=0)

        positive_z = positive_z.fillna(0)

        recent["zscore_score"] += positive_z

    # --------------------------------------------------------
    # Gateway total
    # --------------------------------------------------------

    result = (
        recent
        .groupby("gateway_id")["zscore_score"]
        .sum()
        .reset_index()
    )

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

    # Most recent meter week available
    latest_week = (
        available["week_start"]
        .max()
    )

    latest = available[
        available["week_start"] == latest_week
    ].copy()

    latest["success_rate"] = (
        latest["meters_read"]
        /
        latest["meters_expected"]
        .replace(0, np.nan)
    )

    latest["success_rate"] = (
        latest["success_rate"]
        .fillna(0)
    )

    # Low success = high risk
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

    end = (
        start +
        pd.Timedelta(days=7)
    )

    actual = visits[
        (visits["visited_on"] >= start)
        &
        (visits["visited_on"] < end)
    ]

    return set(
        actual.loc[
            actual["fault"] == 1,
            "gateway_id"
        ]
    )


# ============================================================
# TEST DIFFERENT METER WEIGHTS
# ============================================================

METER_WEIGHTS = [
    0,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2
]


all_results = []


for meter_weight in METER_WEIGHTS:

    print()
    print("=" * 50)
    print(
        "Z-score model | Meter weight:",
        meter_weight
    )
    print("=" * 50)

    total_hits = 0
    total_predictions = 0

    for monday in TEST_WEEKS:

        # ----------------------------------------------------
        # Z-score
        # ----------------------------------------------------

        zscores = zscore_scores(monday)

        # ----------------------------------------------------
        # Meter
        # ----------------------------------------------------

        meters = meter_scores(monday)

        scores = zscores.merge(
            meters,
            on="gateway_id",
            how="left"
        )

        scores["meter_risk"] = (
            scores["meter_risk"]
            .fillna(0)
        )

        if scores.empty:
            continue

        # ----------------------------------------------------
        # Normalize z-score
        # ----------------------------------------------------

        max_z = (
            scores["zscore_score"]
            .max()
        )

        if max_z > 0:

            scores["zscore_norm"] = (
                scores["zscore_score"]
                /
                max_z
            )

        else:

            scores["zscore_norm"] = 0

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        scores["final_score"] = (
            scores["zscore_norm"]
            +
            meter_weight *
            scores["meter_risk"]
        )

        # ----------------------------------------------------
        # Top 15
        # ----------------------------------------------------

        predictions = (
            scores
            .sort_values(
                "final_score",
                ascending=False
            )
            .head(TOP_N)
        )

        predicted_ids = set(
            predictions["gateway_id"]
        )

        actual_ids = actual_faults(
            monday
        )

        hits = len(
            predicted_ids &
            actual_ids
        )

        total_hits += hits
        total_predictions += len(
            predictions
        )

        print(
            f"{monday} | "
            f"hits: {hits} | "
            f"actual: {len(actual_ids)}"
        )

    hit_rate = (
        total_hits /
        total_predictions *
        100
        if total_predictions
        else 0
    )

    print()
    print(
        f"TOTAL: {total_hits} / "
        f"{total_predictions}"
    )

    print(
        f"Hit rate: {hit_rate:.1f} %"
    )

    all_results.append({
        "meter_weight": meter_weight,
        "hits": total_hits,
        "predictions": total_predictions,
        "hit_rate": round(
            hit_rate,
            1
        )
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 50)
print("FINAL Z-SCORE COMPARISON")
print("=" * 50)

results = pd.DataFrame(
    all_results
)

print(
    results.to_string(
        index=False
    )
)

best = results.loc[
    results["hit_rate"].idxmax()
]

print()
print("=" * 50)
print("BEST Z-SCORE MODEL")
print("=" * 50)

print(
    f"Meter weight: "
    f"{best['meter_weight']}"
)

print(
    f"Hits: "
    f"{best['hits']} / "
    f"{best['predictions']}"
)

print(
    f"Hit rate: "
    f"{best['hit_rate']}%"
)

print()
print("Current benchmark: 21.3%")

if best["hit_rate"] > 21.3:

    print(
        "RESULT: Z-SCORE MODEL IS BETTER"
    )

else:

    print(
        "RESULT: CURRENT 21.3% MODEL IS STILL BETTER"
    )
