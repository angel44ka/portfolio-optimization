
"""Benchmark SGA and MGA on the same training data."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from markowitz import estimate_covariance
from genetic_algorithm import run_variable_cardinality_sga
from modified_genetic_algorithm import run_mga


def main():
    returns = pd.read_csv(
        ROOT / "data/processed/returns_train.csv",
        index_col="date",
        parse_dates=True,
    )

    covariance = np.asarray(estimate_covariance(returns))

    results = []

    for seed in range(2026, 2056):
        for method_name, method in [
            ("SGA", run_variable_cardinality_sga),
            ("MGA", run_mga),
        ]:
            kwargs = {
                "population_size": 100,
                "generations": 100,
                "seed": seed,
                "max_assets": 10,
            }

            if method_name == "MGA":
                kwargs["local_search_interval"] = 20

            start = time.perf_counter()
            weights, history = method(covariance, **kwargs)
            elapsed = time.perf_counter() - start

            active = weights > 1e-8
            volatility = np.sqrt(history[-1] * 252) * 100

            valid = (
                np.isclose(weights.sum(), 1.0, atol=1e-8)
                and 4 <= np.count_nonzero(active) <= 10
                and np.all(weights[active] >= 0.02 - 1e-8)
                and np.all(weights <= 0.25 + 1e-8)
                and np.all(np.diff(history) <= 1e-12)
            )

            if not valid:
                raise RuntimeError(
                    f"Invalid portfolio: {method_name}, seed={seed}"
                )

            results.append({
                "method": method_name,
                "seed": seed,
                "volatility_pct": volatility,
                "variance_daily": float(history[-1]),
                "execution_seconds": elapsed,
                "selected_assets": int(np.count_nonzero(active)),
            })

            print(
                f"{method_name} seed={seed}: "
                f"volatility={volatility:.6f}%, "
                f"time={elapsed:.4f}s",
                flush=True,
            )

    output = ROOT / "results/comparison"
    output.mkdir(parents=True, exist_ok=True)

    path = output / "sga_mga_30_runs.csv"

    pd.DataFrame(results).to_csv(path, index=False)

    print("\nSaved:", path)
    print("\nSUMMARY")
    print(
        pd.DataFrame(results).groupby("method").agg(
            mean_volatility=("volatility_pct", "mean"),
            std_volatility=("volatility_pct", "std"),
            mean_time=("execution_seconds", "mean"),
            std_time=("execution_seconds", "std"),
        ).to_string()
    )


if __name__ == "__main__":
    main()
