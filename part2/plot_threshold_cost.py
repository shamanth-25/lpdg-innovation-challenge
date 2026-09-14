import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "part2/threshold_cost_analysis.csv"
)

df = df.sort_values("sigma")

plt.figure(figsize=(8, 5))

plt.plot(
    df["sigma"],
    df["total_cost_eur"],
    marker="o"
)

plt.xlabel("Sigma threshold")
plt.ylabel("Estimated cost (€)")
plt.title("Threshold vs Estimated Historical Cost")

plt.xticks(df["sigma"])

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "part2/threshold_cost_plot.png",
    dpi=150
)

print(
    "Saved part2/threshold_cost_plot.png"
)
