
"""Hybrid genetic algorithm components for GMV optimization."""

import numpy as np
from scipy.optimize import minimize


def optimize_selected_weights(
    weights: np.ndarray,
    covariance: np.ndarray,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Optimize portfolio weights for a fixed set of selected assets."""

    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if weights.ndim != 1:
        raise ValueError("Weights must be one-dimensional.")

    if covariance.shape != (len(weights), len(weights)):
        raise ValueError("Covariance dimensions do not match weights.")

    if not np.all(np.isfinite(weights)) or not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError("Inputs must be finite.")

    selected = np.flatnonzero(weights > 1e-8)

    if len(selected) == 0:
        raise ValueError("Portfolio must contain selected assets.")

    sub_covariance = covariance[np.ix_(selected, selected)]

    result = minimize(
        fun=lambda w: float(w @ sub_covariance @ w),
        x0=weights[selected].copy(),
        jac=lambda w: 2.0 * sub_covariance @ w,
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * len(selected),
        constraints=[{
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones_like(w),
        }],
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")

    optimized = np.zeros_like(weights)
    optimized[selected] = result.x

    if (
        not np.isclose(optimized.sum(), 1.0, atol=1e-8)
        or np.any(optimized[selected] < min_weight - 1e-8)
        or np.any(optimized[selected] > max_weight + 1e-8)
    ):
        raise RuntimeError("Optimized portfolio violates constraints.")

    return optimized

def one_swap_local_search(
    weights: np.ndarray,
    covariance: np.ndarray,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Find the best improving single-asset replacement."""

    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    selected = np.flatnonzero(weights > 1e-8)
    inactive = np.flatnonzero(weights <= 1e-8)

    best_weights = optimize_selected_weights(
        weights,
        covariance,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    best_variance = float(
        best_weights @ covariance @ best_weights
    )

    for removed in selected:
        for added in inactive:
            candidate = np.zeros_like(weights)

            retained = selected[selected != removed]
            new_selected = np.append(retained, added)

            candidate[new_selected] = 1.0 / len(new_selected)

            optimized = optimize_selected_weights(
                candidate,
                covariance,
                min_weight=min_weight,
                max_weight=max_weight,
            )

            variance = float(
                optimized @ covariance @ optimized
            )

            if variance < best_variance - 1e-14:
                best_variance = variance
                best_weights = optimized.copy()

    return best_weights

def run_mga(
    covariance: np.ndarray,
    population_size: int = 100,
    generations: int = 100,
    seed: int = 2026,
    max_assets: int = 10,
    local_search_interval: int = 20,
) -> tuple[np.ndarray, list[float]]:
    """Run a hybrid GA with periodic elite local improvement."""

    from genetic_algorithm import (
        create_variable_cardinality_population,
        create_variable_cardinality_next_generation,
        evaluate_population,
    )

    covariance = np.asarray(covariance, dtype=float)

    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, atol=1e-10)
    ):
        raise ValueError("Invalid covariance matrix.")

    if population_size < 2 or generations < 0:
        raise ValueError("Invalid GA parameters.")

    if local_search_interval < 1:
        raise ValueError("Local search interval must be positive.")

    rng = np.random.default_rng(seed)

    population = create_variable_cardinality_population(
        rng,
        population_size=population_size,
        n_assets=covariance.shape[0],
        max_assets=max_assets,
    )

    history = []
    best_weights = None
    best_variance = float("inf")

    for generation in range(generations + 1):
        variances = evaluate_population(population, covariance)
        best_index = int(np.argmin(variances))

        if (
            generation > 0
            and (
                generation % local_search_interval == 0
                or generation == generations
            )
        ):
            candidate = one_swap_local_search(
                population[best_index],
                covariance,
            )

            candidate_variance = float(
                candidate @ covariance @ candidate
            )

            if candidate_variance < variances[best_index]:
                population[best_index] = candidate
                variances[best_index] = candidate_variance

        best_index = int(np.argmin(variances))

        if variances[best_index] < best_variance:
            best_variance = float(variances[best_index])
            best_weights = population[best_index].copy()

        history.append(best_variance)

        if generation < generations:
            population = create_variable_cardinality_next_generation(
                population,
                variances,
                rng,
                max_assets=max_assets,
            )

    return best_weights, history
