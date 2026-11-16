
"""Evaluate SGA and MGA portfolios on validation data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from markowitz import estimate_covariance
from genetic_algorithm import run_variable_cardinality_sga
from modified_genetic_algorithm import run_mga


def evaluate_portfolio(weights, returns):
    daily_returns = returns.to_numpy() @ weights

    return {
        "annual_return_pct": float(
            daily_returns.mean() * 252 * 100
        ),
        "annual_volatility_pct": float(
            daily_returns.std(ddof=1) * np.sqrt(252) * 100
        ),
    }


def main():
    folder = ROOT / "data/processed"

    train = pd.read_csv(
        folder / "returns_train.csv",
        index_col="date",
        parse_dates=True,
    )
    validation = pd.read_csv(
        folder / "returns_validation.csv",
        index_col="date",
        parse_dates=True,
    )

    if not train.columns.equals(validation.columns):
        raise ValueError("Train and validation assets do not match.")

    covariance = np.asarray(estimate_covariance(train))

    results = []

    for seed in range(2026, 2056):
        for name, method in [
            ("SGA", run_variable_cardinality_sga),
            ("MGA", run_mga),
        ]:
            kwargs = {
                "population_size": 100,
                "generations": 100,
                "seed": seed,
                "max_assets": 10,
            }

            if name == "MGA":
                kwargs["local_search_interval"] = 20

            weights, history = method(covariance, **kwargs)
            active = weights > 1e-8

            if not (
                np.isclose(weights.sum(), 1.0, atol=1e-8)
                and 4 <= active.sum() <= 10
                and np.all(weights[active] >= 0.02 - 1e-8)
                and np.all(weights <= 0.25 + 1e-8)
            ):
                raise RuntimeError(
                    f"Invalid portfolio: {name}, seed={seed}"
                )

            metrics = evaluate_portfolio(weights, validation)

            results.append({
                "seed": seed,
                "method": name,
                "selected_assets": int(active.sum()),
                "train_volatility_pct": float(
                    np.sqrt(history[-1] * 252) * 100
                ),
                **metrics,
            })

        print(f"Completed seed {seed}", flush=True)

    output = ROOT / "results/comparison"
    output.mkdir(parents=True, exist_ok=True)

    path = output / "sga_mga_validation_30_runs.csv"
    data = pd.DataFrame(results)
    data.to_csv(path, index=False)

    print("\n--- VALIDATION SUMMARY ---")
    print(
        data.groupby("method").agg(
            mean_return=("annual_return_pct", "mean"),
            std_return=("annual_return_pct", "std"),
            mean_volatility=("annual_volatility_pct", "mean"),
            std_volatility=("annual_volatility_pct", "std"),
        ).to_string()
    )

    paired = data.pivot(
        index="seed",
        columns="method",
        values="annual_volatility_pct",
    )

    print(
        "\nMGA lower volatility:",
        int((paired["MGA"] < paired["SGA"]).sum()),
        "out of",
        len(paired),
    )
    print("Saved:", path)


if __name__ == "__main__":
    main()
