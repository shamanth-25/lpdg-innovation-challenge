# LPDG Innovation Hub Selection Challenge 2026

## Overview

This project predicts which 15 gateways should be visited each week.

The prediction uses recent telemetry behaviour together with meter-read success risk. Gateways are ranked by a combined risk score and the top 15 are selected.

For Part 2, I chose **Data Science** and focused on analysing the anomaly threshold and its effect on false-positive and false-negative costs.

---

## Part 1 - Gateway Prediction

The predictor uses three telemetry metrics:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

For each prediction Monday:

1. A 28-day historical window is used to calculate a gateway-specific baseline.
2. The most recent 7 days are compared with that baseline.
3. Telemetry values exceeding the selected threshold are counted as breaches.
4. Meter-read success from the latest available meter week before the prediction Monday is converted into meter risk.
5. Telemetry risk and meter risk are combined into a final score.
6. The top 15 gateways are selected.

The current telemetry threshold is **3.5 sigma**.

The final output contains:

```text
week_start
rank
gateway_id
score
reason
