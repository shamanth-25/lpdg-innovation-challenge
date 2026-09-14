import pandas as pd

# Load meter data
meter = pd.read_csv("data/meter_read_success.csv")

# Calculate success rate
meter["success_rate"] = (
    meter["meters_read"] /
    meter["meters_expected"]
)

print("Rows:", len(meter))
print("Gateways:", meter["gateway_id"].nunique())

print("\nSuccess rate statistics:")
print(meter["success_rate"].describe())

print("\nWorst gateway-week records:")
print(
    meter.sort_values("success_rate")
    [["week_start", "gateway_id",
      "meters_expected", "meters_read",
      "success_rate"]]
    .head(20)
    .to_string(index=False)
)

