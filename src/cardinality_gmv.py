"""Cardinality-constrained minimum-variance portfolio: reproducible heuristic.

Run: python src/cardinality_gmv.py

Finds a feasible portfolio with AT MOST K assets. This is a heuristic search,
NOT a mixed-integer optimizer and NOT a certificate of global optimality.
It uses TRAINING returns only and never reads validation/test data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from markowitz import estimate_covariance, fit_gmv

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "returns_train.csv"
OUTPUT = ROOT / "results" / "cardinality_gmv"
K = 10
MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.25
SEED = 2026
TRADING_DAYS = 252


def solve_support(cov: np.ndarray, support: tuple[int, ...],
                  min_weight: float, max_weight: float) -> np.ndarray | None:
    """Optimize continuous weights on a FIXED set of selected securities.

    All selected assets have a minimum positive weight, all other assets zero.
    Returns None when numerically infeasible or optimization does not converge.
    """
    n = cov.shape[0]
    ids = np.asarray(support, dtype=int)
    m = len(ids)
    if m == 0 or min_weight * m > 1 + 1e-10 or max_weight * m < 1 - 1e-10:
        return None
    sub = cov[np.ix_(ids, ids)]
    scale = max(float(np.max(np.abs(sub))), 1e-12)
    x0 = np.full(m, 1.0 / m)
    result = minimize(
        lambda x: float(x @ sub @ x / scale),
        x0,
        jac=lambda x: 2 * (sub @ x) / scale,
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * m,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1,
                      "jac": lambda x: np.ones(m)}],
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not result.success:
        return None
    x = np.asarray(result.x, dtype=float)
    if abs(x.sum() - 1) > 1e-7 or x.min() < min_weight - 1e-7 or x.max() > max_weight + 1e-7:
        return None
    full = np.zeros(n)
    full[ids] = x
    return full


def solve_cardinality_heuristic(cov: np.ndarray, k: int = K,
                                min_weight: float = MIN_WEIGHT,
                                max_weight: float = MAX_WEIGHT,
                                seed: int = SEED,
                                local_passes: int = 2) -> tuple[np.ndarray, dict]:
    """Search subset sizes <= k, then improve the best support by 1-for-1 swaps.

    Deterministic for the same covariance and seed. A good feasible solution
    does NOT establish that the cardinality-constrained global optimum was found.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("Covariance must be a finite square matrix.")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("Covariance must be symmetric.")
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")
    n = len(cov)
    if not isinstance(k, int) or k < 1 or k > n:
        raise ValueError("k must be between 1 and the number of assets.")
    if not (0 < min_weight <= max_weight <= 1):
        raise ValueError("Require 0 < min_weight <= max_weight <= 1.")
    smallest = int(np.ceil(1 / max_weight - 1e-12))
    largest = min(k, n, int(np.floor(1 / min_weight + 1e-12)))
    if smallest > largest:
        raise ValueError("No feasible portfolio satisfies the cardinality and weight bounds.")
    if local_passes < 0:
        raise ValueError("local_passes must be nonnegative.")

    gmv, _ = fit_gmv(cov, max_weight=max_weight)
    rank_gmv = list(np.argsort(-gmv, kind="stable"))
    rank_variance = list(np.argsort(np.diag(cov), kind="stable"))
    rng = np.random.default_rng(seed)
    best = None
    best_value = np.inf
    evaluated = 0

    def consider(indices):
        nonlocal best, best_value, evaluated
        support = tuple(sorted(map(int, indices)))
        weights = solve_support(cov, support, min_weight, max_weight)
        evaluated += 1
        if weights is not None:
            objective = float(weights @ cov @ weights)
            if objective < best_value - 1e-16:
                best, best_value = weights, objective

    for size in range(smallest, largest + 1):
        consider(rank_gmv[:size])
        consider(rank_variance[:size])
        for _ in range(3):
            consider(rng.choice(n, size=size, replace=False))
    if best is None:
        raise RuntimeError("Failed to find a numerically feasible starting portfolio.")

    # Best-improvement 1-for-1 exchanges; size stays fixed, so all iterates
    # remain cardinality-feasible. This cannot prove global optimality.
    for _ in range(local_passes):
        current = tuple(np.flatnonzero(best > 1e-8).tolist())
        outside = sorted(set(range(n)) - set(current))
        previous_value = best_value
        for removed in current:
            for inserted in outside:
                trial = tuple(sorted((set(current) - {removed}) | {inserted}))
                consider(trial)
        if best_value >= previous_value - 1e-15:
            break

    nonzero = int(np.count_nonzero(best > 1e-8))
    if (nonzero > k or not np.isclose(best.sum(), 1, atol=1e-7)
            or best.min() < -1e-8 or best.max() > max_weight + 1e-7
            or np.any(best[best > 1e-8] < min_weight - 1e-7)):
        raise RuntimeError("Final portfolio violated a constraint.")
    diagnostics = {
        "method": "subset_screening_and_local_swaps_heuristic",
        "global_optimality_certified": False,
        "k_max": k,
        "selected_assets": nonzero,
        "minimum_selected_weight": min_weight,
        "maximum_weight": max_weight,
        "sum_weights": float(best.sum()),
        "candidate_subsets_evaluated": evaluated,
        "seed": seed,
        "local_passes_max": local_passes,
        "in_sample_daily_variance": best_value,
        "in_sample_annualized_volatility": float(np.sqrt(best_value * TRADING_DAYS)),
    }
    return best, diagnostics


def main() -> None:
    if not TRAIN.exists():
        raise FileNotFoundError(f"Missing {TRAIN}. Run src/data_preparation.py first.")
    returns = pd.read_csv(TRAIN, index_col="date", parse_dates=True)
    cov = estimate_covariance(returns)
    weights, info = solve_cardinality_heuristic(cov)
    table = pd.DataFrame({"ticker": returns.columns, "weight": weights})
    table = table.sort_values("weight", ascending=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "cardinality_weights.csv", index=False)
    info.update({
        "n_universe_assets": len(returns.columns),
        "training_rows": len(returns),
        "training_start": str(returns.index.min().date()),
        "training_end": str(returns.index.max().date()),
        "note": "Heuristic, not certified global optimum. Training-period performance only.",
    })
    (OUTPUT / "cardinality_diagnostics.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print("Cardinality-constrained GMV: HEURISTIC (not a certified global optimum)")
    print(f"Universe: {len(returns.columns)} assets; maximum selected: {K}")
    print(table[table.weight > 1e-8].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Selected: {info['selected_assets']}; sum of weights: {weights.sum():.10f}")
    print(f"In-sample annualized volatility: {info['in_sample_annualized_volatility']:.4%}")
    print(f"Saved outputs: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
