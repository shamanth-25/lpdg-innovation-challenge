import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

meter = pd.read_csv(
    "data/meter_read_success.csv"
)

visits = pd.read_csv(
    "data/field_visits.csv"
)


# ============================================================
# NORMALIZE IDS
# ============================================================

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
# CALCULATE METER SUCCESS RATE
# ============================================================

meter["success_rate"] = (
    meter["meters_read"] /
    meter["meters_expected"]
)


# ============================================================
# REAL FAULTS
# ============================================================

visits["fault"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)

visits["visited_on"] = pd.to_datetime(
    visits["visited_on"]
)


# ============================================================
# TEST WEEKS
# ============================================================

test_mondays = [
    "2025-09-01",
    "2025-10-06",
    "2025-11-03",
    "2025-12-01",
    "2026-01-05"
]


# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90
]


for threshold in thresholds:

    total_predictions = 0
    total_hits = 0

    print()
    print(
        "======================================"
    )
    print(
        "Threshold:",
        threshold
    )
    print(
        "======================================"
    )

    for monday in test_mondays:

        start = pd.Timestamp(monday)
        end = start + pd.Timedelta(days=7)

        # ----------------------------------------------------
        # Use the most recent meter week BEFORE the Monday
        # ----------------------------------------------------

        available = meter[
            pd.to_datetime(
                meter["week_start"]
            ) < start
        ]

        if available.empty:
            continue

        latest_week = (
            available["week_start"]
            .max()
        )

        weekly = available[
            available["week_start"]
            == latest_week
        ].copy()

        # ----------------------------------------------------
        # Select gateways below threshold
        # ----------------------------------------------------

        predicted = set(
            weekly.loc[
                weekly["success_rate"] < threshold,
                "gateway_id"
            ]
        )

        # ----------------------------------------------------
        # If more than 15, choose the 15 worst
        # ----------------------------------------------------

        if len(predicted) > 15:

            predicted = set(
                weekly
                .sort_values(
                    "success_rate"
                )
                .head(15)["gateway_id"]
            )

        # ----------------------------------------------------
        # Actual faults during following week
        # ----------------------------------------------------

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
        # Calculate hits
        # ----------------------------------------------------

        hits = len(
            predicted &
            actual_faults
        )

        total_predictions += len(
            predicted
        )

        total_hits += hits

        print(
            monday,
            "| predicted:",
            len(predicted),
            "| actual:",
            len(actual_faults),
            "| hits:",
            hits
        )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    if total_predictions > 0:

        hit_rate = (
            total_hits /
            total_predictions *
            100
        )

        print(
            "\nTOTAL:",
            total_hits,
            "/",
            total_predictions,
            "hits"
        )

        print(
            "Hit rate:",
            round(
                hit_rate,
                1
            ),
            "%"
        )
