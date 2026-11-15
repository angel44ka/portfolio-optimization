"""Benchmark solver runtime for GMV and cardinality-constrained GMV.

Run from repository root:
    python3 research/benchmark_optimizers.py

Preprocessing and I/O are deliberately excluded from solver timing.
The heuristic is not a certified global optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from markowitz import estimate_covariance, fit_gmv  # noqa: E402
from cardinality_gmv import solve_cardinality_heuristic  # noqa: E402


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "std_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("At least two timed repeats are required.")

    path = ROOT / "data" / "processed" / "returns_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run data_preparation.py first.")
    returns = pd.read_csv(path, index_col="date", parse_dates=True)
    cov = estimate_covariance(returns)
    n = cov.shape[0]
    if n != 30:
        print(f"Note: benchmark contains {n} assets, not the planned 30.")

    cases = [
        ("GMV_SLSQP", lambda seed: fit_gmv(cov, max_weight=0.25)),
        (
            "Cardinality_GMV_Heuristic",
            lambda seed: solve_cardinality_heuristic(
                cov, k=10, min_weight=0.02, max_weight=0.25,
                seed=seed, local_passes=2,
            ),
        ),
    ]
    records: list[dict] = []
    for name, solve in cases:
        samples: list[float] = []
        print(f"Measuring {name} ...", flush=True)
        for i in range(args.repeats):
            seed = args.seed + i if name != "GMV_SLSQP" else args.seed
            start = time.perf_counter()
            weights, details = solve(seed)
            elapsed = time.perf_counter() - start
            if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1, atol=1e-7):
                raise RuntimeError(f"Invalid weights from {name}, repetition {i + 1}.")
            if name != "GMV_SLSQP":
                active = weights > 1e-8
                if (active.sum() > 10 or np.any(weights[active] < 0.02 - 1e-7)
                        or np.any(weights > 0.25 + 1e-7)):
                    raise RuntimeError("Cardinality solution violated constraints.")
            variance = float(weights @ cov @ weights)
            samples.append(elapsed)
            records.append({
                "method": name,
                "repetition": i + 1,
                "seed": seed,
                "solver_seconds": f"{elapsed:.9f}",
                "daily_variance": f"{variance:.12g}",
                "selected_assets": int(np.count_nonzero(weights > 1e-8)),
            })
        stats = summarize(samples)
        print(
            f"  mean={stats['mean_seconds']:.6f}s; "
            f"median={stats['median_seconds']:.6f}s; "
            f"std={stats['std_seconds']:.6f}s "
            f"(n={len(samples)})"
        )

    outdir = ROOT / "results" / "benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "solver_runtime_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timer": "time.perf_counter",
        "measured_scope": "solver call only; excludes CSV loading, covariance estimation, printing and saving",
        "asset_count": n,
        "training_rows": len(returns),
        "repeats_per_method": args.repeats,
        "seed_first": args.seed,
        "heuristic_seeds_vary_by_repetition": True,
        "heuristic_optimality_certified": False,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (outdir / "benchmark_environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {outdir / 'solver_runtime_runs.csv'}")
    print(f"Saved: {outdir / 'benchmark_environment.json'}")
    print("Note: runtimes are machine- and workload-dependent; solver methods have different workloads.")


if __name__ == "__main__":
    main()
