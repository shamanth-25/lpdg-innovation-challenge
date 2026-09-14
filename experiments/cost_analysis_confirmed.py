#!/usr/bin/env python3

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

SIGMA_VALUES = [
    2.0,
    2.5,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0
]

BASELINE_DAYS = 28
RECENT_DAYS = 7

METER_WEIGHT = 1.0
TOP_N = 15

FALSE_POSITIVE_COST = 380
FALSE_NEGATIVE_COST = 600

FAULT_OUTCOME = "Fehler behoben"

# Historical evaluation weeks
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

print("Loading telemetry...")

telemetry = pd.read_parquet(
    "data/telemetry",
    columns=[
        "gateway_id",
        "ts_utc",
        *METRICS
    ]
)

print("Loading meter data...")

meter = pd.read_csv(
    "data/meter_read_success.csv"
)

print("Loading field visits...")

visits = pd.read_csv(
    "data/field_visits.csv"
)


# ============================================================
# NORMALIZE IDS
# ============================================================

def normalize_ids(series):

    return (
        series
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


telemetry["gateway_id"] = normalize_ids(
    telemetry["gateway_id"]
)

meter["gateway_id"] = normalize_ids(
    meter["gateway_id"]
)

visits["gateway_id"] = normalize_ids(
    visits["gateway_id"]
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
# TELEMETRY SCORE
# ============================================================

def telemetry_scores(monday, sigma):

    end = pd.Timestamp(
        monday,
        tz="UTC"
    )

    # Previous 28 days only
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
                "telemetry_score",
                "worst_metric"
            ]
        )

    # Gateway-specific baseline
    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # Recent 7 days
    recent = window[
        window["ts"] >=
        end - pd.Timedelta(days=RECENT_DAYS)
    ].copy()

    if recent.empty:

        return pd.DataFrame(
            columns=[
                "gateway_id",
                "telemetry_score",
                "worst_metric"
            ]
        )

    recent["flagged"] = 0
    recent["worst_metric"] = ""

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
        ) > sigma * std

        exceeded = exceeded.fillna(False)

        recent["flagged"] += (
            exceeded.astype(int)
        )

        recent.loc[
            exceeded &
            (recent["worst_metric"] == ""),
            "worst_metric"
        ] = metric

    result = (
        recent
        .groupby("gateway_id")
        .agg(
            telemetry_score=(
                "flagged",
                "sum"
            ),
            worst_metric=(
                "worst_metric",
                lambda values: next(
                    (
                        value
                        for value in values
                        if value != ""
                    ),
                    ""
                )
            )
        )
        .reset_index()
    )

    return result


# ============================================================
# METER SCORE
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

    # Only the latest meter week available
    # before the prediction Monday.
    latest_week = (
        available["week_start"].max()
    )

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
        latest["meters_read"]
        /
        latest["meters_expected"].replace(
            0,
            np.nan
        )
    )

    latest["success_rate"] = (
        latest["success_rate"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .clip(0, 1)
    )

    latest["meter_risk"] = (
        1.0 -
        latest["success_rate"]
    )

    return latest[
        [
            "gateway_id",
            "meter_risk"
        ]
    ]


# ============================================================
# PREDICT TOP 15
# ============================================================

def predict_week(monday, sigma):

    telemetry_result = telemetry_scores(
        monday,
        sigma
    )

    meter_result = meter_scores(
        monday
    )

    result = pd.merge(
        telemetry_result,
        meter_result,
        on="gateway_id",
        how="outer"
    )

    result["telemetry_score"] = (
        result["telemetry_score"]
        .fillna(0)
    )

    result["meter_risk"] = (
        result["meter_risk"]
        .fillna(0)
    )

    result["worst_metric"] = (
        result["worst_metric"]
        .fillna("")
    )

    # Normalize telemetry score
    max_score = (
        result["telemetry_score"].max()
    )

    if (
        pd.isna(max_score)
        or max_score <= 0
    ):

        result["sigma_norm"] = 0.0

    else:

        result["sigma_norm"] = (
            result["telemetry_score"]
            /
            max_score
        )

    # Same scoring logic as final_score.py
    result["score"] = (
        result["sigma_norm"]
        +
        METER_WEIGHT *
        result["meter_risk"]
    )

    result = result.sort_values(
        [
            "score",
            "telemetry_score",
            "meter_risk",
            "gateway_id"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ]
    )

    return result.head(TOP_N)


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
        (visits["outcome"] == FAULT_OUTCOME)
    ]

    return set(
        faults["gateway_id"]
    )


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_sigma(sigma):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    evaluated_weeks = 0

    print(
        f"\nTesting sigma = {sigma}"
    )

    for monday in TEST_WEEKS:

        predicted = predict_week(
            monday,
            sigma
        )

        predicted_ids = set(
            predicted["gateway_id"]
        )

        actual_ids = actual_faults(
            monday
        )

        if not actual_ids:
            print(
                f"  {monday}: no confirmed faults"
            )
            continue

        tp = len(
            predicted_ids & actual_ids
        )

        fp = len(
            predicted_ids - actual_ids
        )

        fn = len(
            actual_ids - predicted_ids
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        evaluated_weeks += 1

        print(
            f"  {monday}: "
            f"TP={tp}, "
            f"FP={fp}, "
            f"FN={fn}, "
            f"actual faults={len(actual_ids)}"
        )

    fp_cost = (
        total_fp *
        FALSE_POSITIVE_COST
    )

    fn_cost = (
        total_fn *
        FALSE_NEGATIVE_COST
    )

    total_cost = (
        fp_cost +
        fn_cost
    )

    return {
        "sigma": sigma,
        "weeks_evaluated": evaluated_weeks,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "false_positive_cost_eur": fp_cost,
        "false_negative_cost_eur": fn_cost,
        "total_cost_eur": total_cost
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    for sigma in SIGMA_VALUES:

        result = evaluate_sigma(
            sigma
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # Sort by total cost
    results_df = results_df.sort_values(
        "total_cost_eur"
    ).reset_index(drop=True)

    print(
        "\n"
        + "=" * 80
    )

    print(
        "THRESHOLD COST ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # Save complete results
    results_df.to_csv(
        "threshold_cost_analysis.csv",
        index=False
    )

    best = results_df.iloc[0]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOWEST HISTORICAL COST"
    )

    print(
        "=" * 80
    )

    print(
        f"Threshold: "
        f"{best['sigma']} sigma"
    )

    print(
        f"TP: "
        f"{int(best['true_positives'])}"
    )

    print(
        f"FP: "
        f"{int(best['false_positives'])}"
    )

    print(
        f"FN: "
        f"{int(best['false_negatives'])}"
    )

    print(
        f"FP cost: "
        f"€{int(best['false_positive_cost_eur']):,}"
    )

    print(
        f"FN cost: "
        f"€{int(best['false_negative_cost_eur']):,}"
    )

    print(
        f"TOTAL COST: "
        f"€{int(best['total_cost_eur']):,}"
    )

    print(
        "\nResults saved to "
        "threshold_cost_analysis.csv"
    )


if __name__ == "__main__":
    main()
