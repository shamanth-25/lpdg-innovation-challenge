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

SIGMA = 3
BASELINE_DAYS = 28
RECENT_DAYS = 7

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
# 3-SIGMA SCORE
# ============================================================

def sigma_scores(monday):

    end = pd.Timestamp(
        monday,
        tz="UTC"
    )

    # 28 days BEFORE prediction week
    window = telemetry[
        (telemetry["ts"] >=
         end - pd.Timedelta(days=28))
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

    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    recent = window[
        window["ts"] >=
        end - pd.Timedelta(days=7)
    ].copy()

    recent["flagged"] = 0

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

        exceeded = (
            recent[metric] - mean
        ) > SIGMA * std

        exceeded = exceeded.fillna(False)

        recent["flagged"] += (
            exceeded.astype(int)
        )

    result = (
        recent
        .groupby("gateway_id")["flagged"]
        .sum()
        .reset_index()
    )

    result.columns = [
        "gateway_id",
        "sigma_score"
    ]

    return result


# ============================================================
# METER RISK
# ============================================================

def meter_scores(monday):

    start = pd.Timestamp(monday)

    # Only meter information available
    # before prediction date
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

    # Most recent completed meter week
    latest_week = (
        available["week_start"]
        .max()
    )

    latest = available[
        available["week_start"] == latest_week
    ].copy()

    latest["success_rate"] = (
        latest["meters_read"] /
        latest["meters_expected"]
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
# HISTORICAL FAULT SCORE
# ============================================================

def historical_fault_scores(monday):

    start = pd.Timestamp(monday)

    # IMPORTANT:
    # Only visits that happened BEFORE
    # the prediction week are allowed.
    previous = visits[
        visits["visited_on"] < start
    ].copy()

    if previous.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "historical_faults"
            ]
        )

    result = (
        previous
        .groupby("gateway_id")["fault"]
        .sum()
        .reset_index()
    )

    result.columns = [
        "gateway_id",
        "historical_faults"
    ]

    return result


# ============================================================
# RUN EXPERIMENT
# ============================================================

# We test different strengths of the historical fault signal.
#
# Meter weight is kept at 1 because our previous experiment
# showed that weight 1 was the best.
#
# Historical fault weights:
#
# 0 = don't use historical faults
# 0.25
# 0.5
# 1
# 2

HISTORICAL_WEIGHTS = [
    0,
    0.25,
    0.5,
    1,
    2
]


final_results = []


for historical_weight in HISTORICAL_WEIGHTS:

    print()
    print(
        "=========================================="
    )

    print(
        "Historical fault weight:",
        historical_weight
    )

    print(
        "=========================================="
    )

    total_hits = 0
    total_predictions = 0

    for monday in TEST_WEEKS:

        # ----------------------------------------------------
        # TELEMETRY
        # ----------------------------------------------------

        sigma = sigma_scores(
            monday
        )

        # ----------------------------------------------------
        # METER
        # ----------------------------------------------------

        meter_risk = meter_scores(
            monday
        )

        # ----------------------------------------------------
        # HISTORICAL FAULTS
        # ----------------------------------------------------

        historical = historical_fault_scores(
            monday
        )

        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        scores = sigma.merge(
            meter_risk,
            on="gateway_id",
            how="left"
        )

        scores = scores.merge(
            historical,
            on="gateway_id",
            how="left"
        )

        scores["meter_risk"] = (
            scores["meter_risk"]
            .fillna(0)
        )

        scores["historical_faults"] = (
            scores["historical_faults"]
            .fillna(0)
        )

        # ----------------------------------------------------
        # NORMALIZE SIGMA
        # ----------------------------------------------------

        max_sigma = (
            scores["sigma_score"].max()
        )

        if max_sigma > 0:

            scores["sigma_norm"] = (
                scores["sigma_score"]
                /
                max_sigma
            )

        else:

            scores["sigma_norm"] = 0

        # ----------------------------------------------------
        # NORMALIZE HISTORICAL FAULTS
        # ----------------------------------------------------

        max_faults = (
            scores["historical_faults"].max()
        )

        if max_faults > 0:

            scores["fault_norm"] = (
                scores["historical_faults"]
                /
                max_faults
            )

        else:

            scores["fault_norm"] = 0

        # ----------------------------------------------------
        # COMBINED SCORE
        # ----------------------------------------------------

        scores["combined_score"] = (

            scores["sigma_norm"]

            +

            scores["meter_risk"]

            +

            historical_weight
            *
            scores["fault_norm"]

        )

        # ----------------------------------------------------
        # TOP 15
        # ----------------------------------------------------

        predicted = (
            scores
            .sort_values(
                [
                    "combined_score",
                    "sigma_score",
                    "historical_faults"
                ],
                ascending=False
            )
            .head(15)
        )

        predicted_ids = set(
            predicted["gateway_id"]
        )

        # ----------------------------------------------------
        # ACTUAL FAULTS
        # ----------------------------------------------------

        start = pd.Timestamp(
            monday
        )

        end = (
            start
            +
            pd.Timedelta(days=7)
        )

        actual = visits[
            (visits["visited_on"] >= start)
            &
            (visits["visited_on"] < end)
            &
            (visits["fault"] == 1)
        ]

        actual_ids = set(
            actual["gateway_id"]
        )

        # ----------------------------------------------------
        # HITS
        # ----------------------------------------------------

        hits = len(
            predicted_ids
            &
            actual_ids
        )

        total_hits += hits
        total_predictions += len(
            predicted_ids
        )

        print(
            monday,
            "| hits:",
            hits,
            "| actual:",
            len(actual_ids)
        )

    # ========================================================
    # OVERALL RESULT
    # ========================================================

    hit_rate = (
        total_hits
        /
        total_predictions
        *
        100
    )

    final_results.append({
        "historical_weight":
            historical_weight,

        "hits":
            total_hits,

        "predictions":
            total_predictions,

        "hit_rate":
            round(
                hit_rate,
                1
            )
    })

    print()
    print(
        "TOTAL:",
        total_hits,
        "/",
        total_predictions
    )

    print(
        "Hit rate:",
        round(
            hit_rate,
            1
        ),
        "%"
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print(
    "FINAL COMPARISON"
)

print(
    "================"
)

results = pd.DataFrame(
    final_results
)

print(
    results.to_string(
        index=False
    )
)
