
"""Compare SGA and MGA using approximately matched runtime budgets."""

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

    output = ROOT / "results/comparison"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "sga_mga_matched_runtime.csv"

    results = []

    for seed in range(2026, 2056):
        row = {"seed": seed}

        for name, method, generations in [
            ("sga", run_variable_cardinality_sga, 135),
            ("mga", run_mga, 100),
        ]:
            kwargs = {
                "population_size": 100,
                "generations": generations,
                "seed": seed,
                "max_assets": 10,
            }

            if name == "mga":
                kwargs["local_search_interval"] = 20

            start = time.perf_counter()
            weights, history = method(covariance, **kwargs)
            elapsed = time.perf_counter() - start

            active = weights > 1e-8

            valid = (
                np.isclose(weights.sum(), 1.0, atol=1e-8)
                and 4 <= np.count_nonzero(active) <= 10
                and np.all(weights[active] >= 0.02 - 1e-8)
                and np.all(weights <= 0.25 + 1e-8)
                and np.all(np.diff(history) <= 1e-12)
            )

            if not valid:
                raise RuntimeError(
                    f"Invalid portfolio for {name}, seed={seed}"
                )

            row[f"{name}_volatility_pct"] = (
                np.sqrt(history[-1] * 252) * 100
            )
            row[f"{name}_time_seconds"] = elapsed
            row[f"{name}_selected_assets"] = int(active.sum())

        row["improvement_percentage_points"] = (
            row["sga_volatility_pct"] - row["mga_volatility_pct"]
        )

        results.append(row)

        pd.DataFrame(results).to_csv(path, index=False)

        print(
            f"Seed {seed}: "
            f"SGA={row['sga_volatility_pct']:.6f}%, "
            f"MGA={row['mga_volatility_pct']:.6f}%, "
            f"SGA_time={row['sga_time_seconds']:.3f}s, "
            f"MGA_time={row['mga_time_seconds']:.3f}s",
            flush=True,
        )

    data = pd.DataFrame(results)

    print("\n--- SUMMARY ---")
    print("SGA mean volatility:", f"{data['sga_volatility_pct'].mean():.6f}%")
    print("MGA mean volatility:", f"{data['mga_volatility_pct'].mean():.6f}%")
    print("SGA mean time:", f"{data['sga_time_seconds'].mean():.4f}s")
    print("MGA mean time:", f"{data['mga_time_seconds'].mean():.4f}s")
    print(
        "MGA wins:",
        int((data["improvement_percentage_points"] > 0).sum()),
        "out of",
        len(data),
    )
    print("Saved:", path)


if __name__ == "__main__":
    main()
