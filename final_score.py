#!/usr/bin/env python3

import argparse
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

SIGMA = 3.5

BASELINE_DAYS = 28
RECENT_DAYS = 7

METER_WEIGHT = 1.0

TOP_N = 15

# Official challenge prediction weeks
DEFAULT_WEEKS = [
    "2026-02-02",
    "2026-02-09",
    "2026-02-16",
    "2026-02-23",
    "2026-03-02",
    "2026-03-09",
    "2026-03-16",
    "2026-03-23"
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="LPDG gateway visit predictor"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Prediction Monday, e.g. 2026-02-02"
    )

    parser.add_argument(
        "--weeks",
        type=int,
        default=8,
        help="Number of consecutive Mondays to predict"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help="Anomaly threshold in standard deviations"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV filename"
    )

    return parser.parse_args()


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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

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

    # Normalize IDs
    telemetry["gateway_id"] = normalize_ids(
        telemetry["gateway_id"]
    )

    meter["gateway_id"] = normalize_ids(
        meter["gateway_id"]
    )

    # Parse dates
    telemetry["ts"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True
    )

    meter["week_start"] = pd.to_datetime(
        meter["week_start"]
    )

    # Remove invalid telemetry rows
    telemetry = telemetry.dropna(
        subset=["gateway_id", "ts"]
    )

    print(
        f"Telemetry rows: {len(telemetry):,}"
    )

    print(
        f"Telemetry gateways: "
        f"{telemetry['gateway_id'].nunique()}"
    )

    print(
        f"Meter rows: {len(meter):,}"
    )

    print(
        f"Meter gateways: "
        f"{meter['gateway_id'].nunique()}"
    )

    return telemetry, meter


# ============================================================
# METER RISK
#
# IMPORTANT:
# Only use the latest meter week that existed BEFORE
# the prediction Monday.
# ============================================================

def meter_scores(meter, monday):

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

    # If expected is zero or data is invalid,
    # treat it as high risk.
    latest["success_rate"] = (
        latest["success_rate"]
        .replace([np.inf, -np.inf], np.nan)
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
# TELEMETRY SIGMA SCORE
#
# 28-day historical baseline
# 7-day recent window
# 3.5 sigma threshold
# ============================================================

def telemetry_scores(telemetry, monday):

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
                "telemetry_score",
                "worst_metric"
            ]
        )

    # --------------------------------------------------------
    # Gateway-specific baseline
    # --------------------------------------------------------

    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # --------------------------------------------------------
    # Most recent 7 days
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Check each metric
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

        exceeded = (
            recent[metric] - mean
        ) > SIGMA * std

        exceeded = exceeded.fillna(False)

        recent["flagged"] += (
            exceeded.astype(int)
        )

        # Store the first metric that breached
        recent.loc[
            exceeded &
            (recent["worst_metric"] == ""),
            "worst_metric"
        ] = metric

    # --------------------------------------------------------
    # Gateway-level score
    # --------------------------------------------------------

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
# BUILD PREDICTIONS FOR ONE WEEK
# ============================================================

def predict_week(
    telemetry,
    meter,
    monday
):

    print(
        f"\nPredicting week: {monday}"
    )

    telemetry_result = telemetry_scores(
        telemetry,
        monday
    )

    meter_result = meter_scores(
        meter,
        monday
    )

    # --------------------------------------------------------
    # Combine telemetry and meter information
    #
    # Outer join means a gateway can be considered even if
    # only one data source contains it.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize telemetry score
    #
    # This keeps telemetry and meter risk on comparable
    # scales.
    # --------------------------------------------------------

    max_telemetry = (
        result["telemetry_score"].max()
    )

    if (
        pd.isna(max_telemetry)
        or max_telemetry <= 0
    ):
        result["sigma_norm"] = 0.0
    else:
        result["sigma_norm"] = (
            result["telemetry_score"]
            /
            max_telemetry
        )

    # --------------------------------------------------------
    # Final score
    #
    # Higher = higher visit priority
    # --------------------------------------------------------

    result["score"] = (
        result["sigma_norm"]
        +
        METER_WEIGHT *
        result["meter_risk"]
    )

    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

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
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Select exactly TOP_N gateways
    # --------------------------------------------------------

    result = result.head(TOP_N).copy()

    # --------------------------------------------------------
    # Generate explanations
    # --------------------------------------------------------

    reasons = []

    for _, row in result.iterrows():

        telemetry_score = int(
            row["telemetry_score"]
        )

        meter_risk = float(
            row["meter_risk"]
        )

        worst_metric = row["worst_metric"]

        if telemetry_score > 0:

            reason = (
                f"{telemetry_score} recent telemetry "
                f"breach(es) beyond the gateway's own "
                f"{SIGMA}-sigma baseline "
                f"({worst_metric}); "
                f"meter-read risk "
                f"{meter_risk:.0%}."
            )

        elif meter_risk > 0:

            reason = (
                f"Meter-read success is reduced, "
                f"giving {meter_risk:.0%} "
                f"meter-read risk."
            )

        else:

            reason = (
                "Highest combined telemetry and "
                "meter-risk score among available gateways."
            )

        reasons.append(reason)

    result["reason"] = reasons

    # --------------------------------------------------------
    # Final output format
    # --------------------------------------------------------

    result["week_start"] = (
        pd.Timestamp(monday)
        .strftime("%Y-%m-%d")
    )

    result["rank"] = (
        range(1, len(result) + 1)
    )

    return result[
        [
            "week_start",
            "rank",
            "gateway_id",
            "score",
            "reason"
        ]
    ]


# ============================================================
# DETERMINE PREDICTION WEEKS
# ============================================================

def get_prediction_weeks(args):

    # Explicit start date:
    # useful for unseen-data/live testing.
    if args.start_date is not None:

        start = pd.Timestamp(
            args.start_date
        )

        if start.weekday() != 0:

            raise ValueError(
                f"{args.start_date} is not a Monday. "
                "Please provide a Monday."
            )

        return [
            start + pd.Timedelta(days=7 * i)
            for i in range(args.weeks)
        ]

    # No argument:
    # generate the official challenge weeks.
    if args.weeks != 8:

        raise ValueError(
            "--weeks can only be changed when "
            "--start-date is provided."
        )

    return [
        pd.Timestamp(date)
        for date in DEFAULT_WEEKS
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    global SIGMA
    SIGMA = args.sigma

    telemetry, meter = load_data()

    weeks = get_prediction_weeks(args)

    all_predictions = []

    for monday in weeks:

        weekly = predict_week(
            telemetry,
            meter,
            monday
        )

        if len(weekly) < TOP_N:

            print(
                f"WARNING: only {len(weekly)} "
                f"gateways available for {monday:%Y-%m-%d}"
            )

        all_predictions.append(
            weekly
        )

    # --------------------------------------------------------
    # Combine all weeks
    # --------------------------------------------------------

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    predictions.to_csv(
        args.output,
        index=False
    )

    print(
        "\n========================================"
    )

    print(
        "Prediction complete"
    )

    print(
        "========================================"
    )

    print(
        f"Weeks predicted: {len(weeks)}"
    )

    print(
        f"Rows written: {len(predictions)}"
    )

    print(
        f"Output: {args.output}"
    )

    print(
        f"Threshold: {SIGMA} sigma"
    )

    print(
        f"Meter weight: {METER_WEIGHT}"
    )

    print(
        f"Visits per week: {TOP_N}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
