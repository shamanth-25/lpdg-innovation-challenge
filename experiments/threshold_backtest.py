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

METER_WEIGHT = 1

SIGMA_VALUES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0
]

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
# ORIGINAL 3-SIGMA STYLE SCORE
# ============================================================

def sigma_scores(monday, sigma):

    end = pd.Timestamp(
        monday,
        tz="UTC"
    )

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

    stats = (
        window
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

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
        ) > sigma * std

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
# TEST THRESHOLDS
# ============================================================

results = []


for sigma in SIGMA_VALUES:

    print()
    print("=" * 55)
    print(
        f"Testing sigma threshold: {sigma}"
    )
    print("=" * 55)

    total_hits = 0
    total_predictions = 0

    for monday in TEST_WEEKS:

        sigma_df = sigma_scores(
            monday,
            sigma
        )

        meter_df = meter_scores(
            monday
        )

        scores = sigma_df.merge(
            meter_df,
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
        # ORIGINAL MODEL NORMALIZATION
        # ----------------------------------------------------

        max_score = (
            scores["sigma_score"]
            .max()
        )

        if max_score > 0:

            scores["sigma_norm"] = (
                scores["sigma_score"]
                /
                max_score
            )

        else:

            scores["sigma_norm"] = 0

        # ----------------------------------------------------
        # COMBINE TELEMETRY + METER
        # ----------------------------------------------------

        scores["final_score"] = (
            scores["sigma_norm"]
            +
            METER_WEIGHT *
            scores["meter_risk"]
        )

        # ----------------------------------------------------
        # TOP 15
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
        f"Hit rate: {hit_rate:.1f}%"
    )

    results.append({
        "sigma": sigma,
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
print("=" * 55)
print("FINAL THRESHOLD COMPARISON")
print("=" * 55)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)

best = results_df.loc[
    results_df["hit_rate"].idxmax()
]

print()
print("=" * 55)
print("BEST THRESHOLD")
print("=" * 55)

print(
    f"Sigma: {best['sigma']}"
)

print(
    f"Hits: {best['hits']} / "
    f"{best['predictions']}"
)

print(
    f"Hit rate: {best['hit_rate']}%"
)

print()
print("Current benchmark: 21.3%")

if best["hit_rate"] > 21.3:

    print(
        "RESULT: NEW THRESHOLD BEATS "
        "CURRENT MODEL"
    )

else:

    print(
        "RESULT: CURRENT 21.3% MODEL "
        "IS STILL BETTER"
    )
