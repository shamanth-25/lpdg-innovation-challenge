# Decisions

## 1. Choosing Part 2: Data Science

I chose Data Science for Part 2.

I chose this because the challenge gives different costs for false positives and false negatives. So instead of only looking at prediction accuracy, I wanted to understand how changing the anomaly threshold affects the actual cost.

The main question I focused on was:

- What happens if the threshold is lower?
- What happens if the threshold is higher?
- Which threshold gives the best cost while selecting only 15 gateways?

---

## 2. Using a 3.5 Sigma Threshold

I tested different sigma thresholds:

- 2.0
- 2.5
- 3.0
- 3.25
- 3.5
- 3.75
- 4.0

The results from five historical test weeks were:

| Sigma | True Positives | False Positives | False Negatives | Total Cost |
|---:|---:|---:|---:|---:|
| 2.0 | 13 | 62 | 10 | €29,560 |
| 2.5 | 16 | 59 | 7 | €26,620 |
| 3.0 | 16 | 59 | 7 | €26,620 |
| 3.25 | 14 | 61 | 9 | €28,580 |
| 3.5 | 17 | 58 | 6 | €25,640 |
| 3.75 | 16 | 59 | 7 | €26,620 |
| 4.0 | 13 | 62 | 10 | €29,560 |

I selected 3.5 sigma because it gave the lowest estimated cost in these tests.

At 3.5 sigma:

- True positives = 17
- False positives = 58
- False negatives = 6
- False-positive cost = €22,040
- False-negative cost = €3,600
- Total estimated cost = €25,640

The next best tested thresholds had a cost of €26,620.
### How much the result moves across weeks

The selected 3.5 sigma threshold was also checked separately on each of the five historical evaluation weeks:

| Week | True Positives | False Positives | False Negatives | Estimated Cost |
|---|---:|---:|---:|---:|
| 2025-09-01 | 3 | 12 | 3 | €6,360 |
| 2025-10-06 | 3 | 12 | 0 | €4,560 |
| 2025-11-03 | 4 | 11 | 2 | €5,380 |
| 2025-12-01 | 3 | 12 | 1 | €5,160 |
| 2026-01-05 | 4 | 11 | 0 | €4,180 |

The estimated weekly cost therefore ranged from **€4,180 to €6,360**, with an average of **€5,128** across these five weeks.

This variation shows why I do not treat €25,640 as a guaranteed future cost. The result depends on which gateways fail and which faults occur in the evaluation week. The five weeks provide useful evidence, but they are still a small historical sample.

I did not choose 3.5 sigma just because of the number of correct predictions. I used the cost given in the challenge to make the decision.

The historical calculation applies the €600 false-negative cost once for each missed confirmed fault in the evaluation week. It does not model additional €600 charges in later weeks if the same gateway remains broken, because the available labels do not reliably show whether a missed gateway stayed broken in subsequent weeks.
---

## 3. Using a Gateway-Specific Baseline

For each gateway, I use the previous 28 days to calculate its own mean and standard deviation.

Then I look at the most recent 7 days and check whether the telemetry values are more than the selected sigma threshold above that gateway's normal behaviour.

I chose this instead of using one global mean and standard deviation for all gateways.

The reason is that different gateways can have different normal behaviour. A value that is normal for one gateway may be unusual for another gateway.

---

## 4. Combining Telemetry and Meter Data

I use three telemetry metrics:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

I also use meter-read success as another risk signal.

For the meter data, I use the latest meter week that was available before the prediction Monday. I calculate:

`meter risk = 1 - meter success rate`

The final score combines the normalized telemetry score with meter risk.

The meter weight used in the final model is 1.0.

I tested different meter weights during backtesting. A weight of 1 performed better than the larger weights that I tested.

I also tested several other approaches, including metric-only, z-score, magnitude and historical-fault weighting approaches. They did not improve on the combined approach used here.

---

## 5. Selecting 15 Gateways

The challenge has a hard limit of 15 visits per week.

Therefore, I rank all available gateways using the final score and select the top 15.

The output contains:

- week
- rank
- gateway ID
- score
- reason

This also gives the engineer a priority order instead of just saying whether a gateway is abnormal or not.

---

## Field Visit Outcome Handling

The field visit data contains three outcomes:

- `Fehler behoben`
- `Kein Fehler gefunden`
- `Kein Zugang`

For the historical evaluation, I used `Fehler behoben` as a confirmed fault.

`Kein Fehler gefunden` means that no fault was found during the visit.

I did not treat `Kein Zugang` as a fault or as a healthy gateway because the technician could not access the gateway.

There were two `Kein Zugang` visits in the five historical evaluation weeks.

---

## What the Model Cannot Do

The threshold analysis is based on five historical evaluation weeks, so the cost result is not a guarantee of future performance.

The model also depends on having enough historical telemetry to calculate a gateway-specific baseline.

Another limitation is that the available field-visit data does not always tell us what happened when a gateway could not be accessed.

Two more weeks of labelled field outcomes would give more data for comparing thresholds and would make the cost estimates more reliable.

More telemetry would also give more observations for calculating each gateway's normal behaviour.
The historical cost calculation also does not model recurring false-negative costs across later weeks. More complete longitudinal fault labels would be needed to estimate that persistence cost reliably.
