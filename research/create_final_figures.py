
"""Generate publication-ready figures and summary tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results/comparison"
OUTPUT = ROOT / "results/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

COLORS = {
    "SGA": "#4472C4",
    "MGA": "#E07A35",
}


def load_data(filename, expected_rows):
    path = INPUT / filename

    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)

    if len(data) != expected_rows:
        raise ValueError(
            f"{filename}: expected {expected_rows} rows, "
            f"got {len(data)}"
        )

    if data.isna().any().any():
        raise ValueError(f"{filename}: missing values found")

    return data


train = load_data("sga_mga_30_runs.csv", 60)
validation = load_data(
    "sga_mga_validation_30_runs.csv", 60
)
test = load_data("sga_mga_test_30_runs.csv", 60)
matched = load_data(
    "sga_mga_matched_runtime.csv", 30
)

for name, data in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    counts = data.groupby("method")["seed"].nunique()

    if counts.to_dict() != {"MGA": 30, "SGA": 30}:
        raise ValueError(f"Invalid {name} seed counts")


def save_figure(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name}.png", bbox_inches="tight")
    plt.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}.png and {name}.pdf")



# Figure 1: Annualized volatility across datasets

periods = ["Training", "Validation", "Test"]
datasets = [train, validation, test]
volatility_columns = [
    "volatility_pct",
    "annual_volatility_pct",
    "annual_volatility_pct",
]

x = np.arange(len(periods))
width = 0.34

fig, ax = plt.subplots(figsize=(8, 5))

for method, offset in [
    ("SGA", -width / 2),
    ("MGA", width / 2),
]:
    means = []
    stds = []

    for data, column in zip(datasets, volatility_columns):
        values = data.loc[data["method"] == method, column]
        means.append(values.mean())
        stds.append(values.std(ddof=1))

    bars = ax.bar(
        x + offset,
        means,
        width,
        yerr=stds,
        capsize=4,
        label=method,
        color=COLORS[method],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 1, "capthick": 1},
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.3f}%" for value in means],
        padding=5,
        fontsize=9,
    )

ax.set_xticks(x, periods)
ax.set_ylabel("Annualized volatility (%)")
ax.set_xlabel("Evaluation period")
ax.set_title("Annualized Portfolio Volatility")
ax.set_ylim(0, 19)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax.legend(title="Algorithm")

save_figure("figure_1_volatility")



# Figure 2: Out-of-sample annualized returns

x = np.arange(2)
width = 0.34

fig, ax = plt.subplots(figsize=(8, 5))

for offset, method in [
    (-width / 2, "SGA"),
    (width / 2, "MGA"),
]:
    means = [
        data.loc[
            data["method"] == method, "annual_return_pct"
        ].mean()
        for data in [validation, test]
    ]

    stds = [
        data.loc[
            data["method"] == method, "annual_return_pct"
        ].std(ddof=1)
        for data in [validation, test]
    ]

    positions = x + offset

    bars = ax.bar(
        positions,
        means,
        width,
        label=method,
        color=COLORS[method],
        yerr=stds,
        capsize=4,
        error_kw={"elinewidth": 1, "capthick": 1},
    )

    ax.bar_label(bars, fmt="%.2f%%", padding=5)

ax.set_xticks(x, ["Validation", "Test"])
ax.set_ylabel("Annualized mean return (%)")
ax.set_title("Out-of-Sample Portfolio Returns")
ax.set_ylim(0, 24)
ax.legend(title="Algorithm")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

save_figure("figure_2_returns")





# Figure 3: Mean execution time

time_data = {
    "SGA (100 gen)": train.loc[
        train["method"] == "SGA", "execution_seconds"
    ].mean(),
    "MGA (100 gen)": train.loc[
        train["method"] == "MGA", "execution_seconds"
    ].mean(),
    "SGA (135 gen)": matched["sga_time_seconds"].mean(),
    "MGA (matched run)": matched["mga_time_seconds"].mean(),
}

names = list(time_data.keys())
times = list(time_data.values())

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    names,
    times,
    color=[
        COLORS["SGA"],
        COLORS["MGA"],
        COLORS["SGA"],
        COLORS["MGA"],
    ],
)

ax.bar_label(bars, fmt="%.2f s", padding=3)
ax.set_ylabel("Mean execution time (seconds)")
ax.set_title("Computational Cost Comparison")
ax.set_ylim(0, max(times) * 1.2)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.2)

save_figure("figure_3_runtime")


# Table 1: Summary statistics

rows = []

for period, data, vol_column in zip(
    periods, datasets, volatility_columns
):
    for method in ["SGA", "MGA"]:
        subset = data.loc[data["method"] == method]

        return_column = (
            "annual_return_pct"
            if period != "Training"
            else None
        )

        rows.append({
            "period": period,
            "method": method,
            "n_runs": len(subset),
            "mean_volatility_pct": subset[vol_column].mean(),
            "std_volatility_pp": subset[vol_column].std(ddof=1),
            "mean_return_pct": (
                subset[return_column].mean()
                if return_column
                else np.nan
            ),
            "std_return_pp": (
                subset[return_column].std(ddof=1)
                if return_column
                else np.nan
            ),
        })

summary = pd.DataFrame(rows)

# Training returns were not stored in the original benchmark.
# Do not substitute single-seed returns for 30-run averages.

summary_path = OUTPUT / "table_1_summary.csv"
summary.to_csv(summary_path, index=False, float_format="%.6f")

print("\nFINAL SUMMARY TABLE")
print(summary.to_string(index=False))
print("\nSaved:", summary_path)
print("\nAll figures generated successfully.")
