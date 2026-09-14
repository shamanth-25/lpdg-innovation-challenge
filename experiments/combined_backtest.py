import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

METRIC_WEIGHTS = {
    "offline_duration_sec":1,
    "disconnection_cnt":1,
    "reboot_cnt":1
}
METRICS=list(METRIC_WEIGHTS.keys())

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
# FUNCTION: 3-SIGMA SCORE
# ============================================================

def sigma_scores(monday):

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
                "sigma_score"
            ]
        )

    # Gateway-specific mean/std
    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # Last 7 days
    recent = window[
        window["ts"] >=
        end - pd.Timedelta(days=RECENT_DAYS)
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
            exceeded.astype(int)*METRIC_WEIGHTS[metric]
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
# FUNCTION: METER RISK
# ============================================================

def meter_scores(monday):

    start = pd.Timestamp(monday)

    # Only information available BEFORE Monday
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

    # Most recent available meter week
    latest_week = (
        available["week_start"]
        .max()
    )

    latest = available[
        available["week_start"] == latest_week
    ].copy()

    # Success rate
    latest["success_rate"] = (
        latest["meters_read"] /
        latest["meters_expected"]
    )

    # Convert success rate into risk.
    #
    # 1.0 success  -> 0 risk
    # 0.5 success  -> 0.5 risk
    # 0.0 success  -> 1.0 risk
    #
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
# RUN EXPERIMENT
# ============================================================

# Different weights for meter risk
weights = [
    0,
    1,
    2,
    5,
    10
]

all_results = []


for weight in weights:

    print()
    print(
        "=========================================="
    )
    print(
        "Meter weight:",
        weight
    )
    print(
        "=========================================="
    )

    total_hits = 0
    total_predictions = 0

    for monday in TEST_WEEKS:

        # ----------------------------------------------------
        # Get sigma scores
        # ----------------------------------------------------

        sigma = sigma_scores(
            monday
        )

        # ----------------------------------------------------
        # Get meter risk
        # ----------------------------------------------------

        meter_risk = meter_scores(
            monday
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        scores = sigma.merge(
            meter_risk,
            on="gateway_id",
            how="left"
        )

        scores["meter_risk"] = (
            scores["meter_risk"]
            .fillna(0)
        )

        # ----------------------------------------------------
        # Normalize sigma score
        #
        # This prevents the raw sigma count from completely
        # dominating the meter signal.
        # ----------------------------------------------------

        if len(scores) > 0:

            max_sigma = (
                scores["sigma_score"].max()
            )

            if max_sigma > 0:
                scores["sigma_norm"] = (
                    scores["sigma_score"] /
                    max_sigma
                )
            else:
                scores["sigma_norm"] = 0

        else:
            scores["sigma_norm"] = 0

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        scores["combined_score"] = (
            scores["sigma_norm"]
            +
            weight * scores["meter_risk"]
        )

        # ----------------------------------------------------
        # Pick top 15
        # ----------------------------------------------------

        predicted = (
            scores
            .sort_values(
                [
                    "combined_score",
                    "sigma_score"
                ],
                ascending=False
            )
            .head(15)
        )

        predicted_ids = set(
            predicted["gateway_id"]
        )

        # ----------------------------------------------------
        # Actual faults during following week
        # ----------------------------------------------------

        start = pd.Timestamp(
            monday
        )

        end = (
            start +
            pd.Timedelta(days=7)
        )

        actual = visits[
            (visits["visited_on"] >= start)
            &
            (visits["visited_on"] < end)
            &
            (visits["fault"] == 1)
        ]

        actual_faults = set(
            actual["gateway_id"]
        )

        # ----------------------------------------------------
        # Hits
        # ----------------------------------------------------

        hits = len(
            predicted_ids &
            actual_faults
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
            len(actual_faults)
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    hit_rate = (
        total_hits /
        total_predictions *
        100
    )

    all_results.append({
        "meter_weight": weight,
        "hits": total_hits,
        "predictions": total_predictions,
        "hit_rate": round(
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
# FINAL COMPARISON
# ============================================================

print()
print()
print(
    "FINAL COMPARISON"
)

print(
    "================"
)

result_df = pd.DataFrame(
    all_results
)

print(
    result_df.to_string(
        index=False
    )
)
