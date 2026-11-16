
"""Evaluate frozen SGA and MGA portfolios on the test dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from markowitz import estimate_covariance
from genetic_algorithm import run_variable_cardinality_sga
from modified_genetic_algorithm import run_mga


def evaluate(weights, returns):
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
    output = ROOT / "results/comparison"
    config_path = output / "experiment_config.json"

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    folder = ROOT / "data/processed"

    train = pd.read_csv(
        folder / "returns_train.csv",
        index_col="date",
        parse_dates=True,
    )
    test = pd.read_csv(
        folder / "returns_test.csv",
        index_col="date",
        parse_dates=True,
    )

    if not train.columns.equals(test.columns):
        raise ValueError("Asset columns do not match.")

    if not train.index.max() < test.index.min():
        raise ValueError("Training and test dates overlap.")

    if len(train.columns) != config["number_of_assets"]:
        raise ValueError("Unexpected number of assets.")

    if (
        str(train.index.min().date()) != config["training_start"]
        or str(train.index.max().date()) != config["training_end"]
        or str(test.index.min().date()) != config["test_start"]
        or str(test.index.max().date()) != config["test_end"]
    ):
        raise ValueError("Data dates do not match frozen configuration.")

    covariance = np.asarray(estimate_covariance(train))

    results = []
    path = output / "sga_mga_test_30_runs.csv"

    for seed in config["seeds"]:
        for name, method in [
            ("SGA", run_variable_cardinality_sga),
            ("MGA", run_mga),
        ]:
            kwargs = {
                "population_size": config["population_size"],
                "generations": config["generations"],
                "seed": seed,
                "max_assets": config["maximum_selected_assets"],
            }

            if name == "MGA":
                kwargs["local_search_interval"] = config[
                    "mga_local_search_interval"
                ]

            weights, history = method(covariance, **kwargs)
            active = weights > 1e-8

            valid = (
                np.isfinite(weights).all()
                and np.isclose(weights.sum(), 1.0, atol=1e-8)
                and 4 <= active.sum() <= config["maximum_selected_assets"]
                and np.all(
                    weights[active]
                    >= config["minimum_selected_weight"] - 1e-8
                )
                and np.all(
                    weights <= config["maximum_selected_weight"] + 1e-8
                )
                and np.all(np.diff(history) <= 1e-12)
            )

            if not valid:
                raise RuntimeError(
                    f"Invalid portfolio: {name}, seed={seed}"
                )

            metrics = evaluate(weights, test)

            results.append({
                "seed": seed,
                "method": name,
                "selected_assets": int(active.sum()),
                "train_volatility_pct": float(
                    np.sqrt(history[-1] * 252) * 100
                ),
                **metrics,
            })

        # Save completed pairs to avoid losing earlier results.
        pd.DataFrame(results).to_csv(path, index=False)

        print(f"Completed seed {seed}", flush=True)

    data = pd.DataFrame(results)

    print("\n--- TEST SUMMARY ---")
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
