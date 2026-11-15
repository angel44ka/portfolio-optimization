
"""Standard Genetic Algorithm for cardinality-constrained GMV optimization."""

import numpy as np


def create_individual(
    rng: np.random.Generator,
    n_assets: int = 30,
    k: int = 10,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Generate a random feasible portfolio with exactly k active assets."""

    if not (1 <= k <= n_assets):
        raise ValueError("Invalid number of selected assets.")

    if k * min_weight > 1 or k * max_weight < 1:
        raise ValueError("Infeasible weight constraints.")

    weights = np.zeros(n_assets)

    selected = rng.choice(n_assets, size=k, replace=False)
    weights[selected] = 1.0 / k

    for _ in range(100):
        i, j = rng.choice(selected, size=2, replace=False)

        max_transfer = min(
            weights[i] - min_weight,
            max_weight - weights[j],
        )

        if max_transfer > 0:
            amount = rng.uniform(0, max_transfer)
            weights[i] -= amount
            weights[j] += amount

    return weights

def portfolio_variance(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate portfolio variance for a given weight vector."""
    return float(weights @ covariance @ weights)

def create_population(
    rng: np.random.Generator,
    population_size: int = 100,
    n_assets: int = 30,
    k: int = 10,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Create an initial population of feasible portfolios."""
    if population_size < 1:
        raise ValueError("Population size must be positive.")

    return np.array([
        create_individual(
            rng,
            n_assets=n_assets,
            k=k,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        for _ in range(population_size)
    ])

def evaluate_population(
    population: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Calculate portfolio variance for every individual."""
    return np.array([
        portfolio_variance(weights, covariance)
        for weights in population
    ])

def tournament_selection(
    population: np.ndarray,
    variances: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> np.ndarray:
    """Select one parent using tournament selection."""

    if not 1 <= tournament_size <= len(population):
        raise ValueError("Invalid tournament size.")

    candidates = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )

    winner = candidates[np.argmin(variances[candidates])]

    return population[winner].copy()

def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    k: int = 10,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Combine two portfolios and repair weight constraints."""

    if parent1.shape != parent2.shape:
        raise ValueError("Parent shapes must match.")

    if not (1 <= k <= len(parent1)):
        raise ValueError("Invalid number of active assets.")

    if not (k * min_weight <= 1 <= k * max_weight):
        raise ValueError("Infeasible weight constraints.")

    union = np.flatnonzero((parent1 > 1e-8) | (parent2 > 1e-8))

    if len(union) < k:
        raise ValueError("Not enough active assets in parents.")

    selected = rng.choice(union, size=k, replace=False)

    target = (parent1[selected] + parent2[selected]) / 2.0

    # Project weights onto the bounded simplex.
    low = np.min(target - max_weight) - 1.0
    high = np.max(target - min_weight) + 1.0

    for _ in range(100):
        midpoint = (low + high) / 2.0
        projected = np.clip(
            target - midpoint,
            min_weight,
            max_weight,
        )

        if projected.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint

    child = np.zeros(len(parent1))
    child[selected] = np.clip(
        target - (low + high) / 2.0,
        min_weight,
        max_weight,
    )

    return child

def mutate(
    individual: np.ndarray,
    rng: np.random.Generator,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Mutate a portfolio while preserving its constraints."""

    child = individual.copy()

    active = np.flatnonzero(child > 1e-8)
    inactive = np.flatnonzero(child <= 1e-8)

    if rng.random() < 0.5 and len(inactive) > 0:
        # Replace one selected asset with an unselected asset.
        remove_idx = rng.choice(active)
        add_idx = rng.choice(inactive)

        child[add_idx] = child[remove_idx]
        child[remove_idx] = 0.0

    elif len(active) >= 2:
        # Transfer weight between two selected assets.
        i, j = rng.choice(active, size=2, replace=False)

        max_transfer = min(
            child[i] - min_weight,
            max_weight - child[j],
        )

        if max_transfer > 0:
            amount = rng.uniform(0, max_transfer)
            child[i] -= amount
            child[j] += amount

    return child

def create_next_generation(
    population: np.ndarray,
    variances: np.ndarray,
    rng: np.random.Generator,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
) -> np.ndarray:
    """Create the next generation with elitism."""

    if len(population) != len(variances):
        raise ValueError("Population and variance sizes must match.")

    if len(population) < 3:
        raise ValueError("Population must contain at least 3 individuals.")

    if not 0 <= crossover_probability <= 1:
        raise ValueError("Invalid crossover probability.")

    if not 0 <= mutation_probability <= 1:
        raise ValueError("Invalid mutation probability.")

    best_idx = np.argmin(variances)
    next_population = [population[best_idx].copy()]

    while len(next_population) < len(population):
        parent1 = tournament_selection(population, variances, rng)
        parent2 = tournament_selection(population, variances, rng)

        if rng.random() < crossover_probability:
            child = crossover(parent1, parent2, rng)
        else:
            child = parent1.copy()

        if rng.random() < mutation_probability:
            child = mutate(child, rng)

        next_population.append(child)

    return np.array(next_population)

def run_sga(
    covariance: np.ndarray,
    population_size: int = 100,
    generations: int = 100,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the standard GA and record the best variance by generation."""

    if generations < 1:
        raise ValueError("Generations must be positive.")

    rng = np.random.default_rng(seed)

    population = create_population(
        rng,
        population_size=population_size,
        n_assets=covariance.shape[0],
    )

    history = []

    for generation in range(generations + 1):
        variances = evaluate_population(population, covariance)

        best_variance = float(np.min(variances))
        history.append(best_variance)

        if generation == generations:
            break

        population = create_next_generation(
            population,
            variances,
            rng,
        )

    best_index = int(np.argmin(variances))
    best_weights = population[best_index].copy()

    return best_weights, np.array(history)
