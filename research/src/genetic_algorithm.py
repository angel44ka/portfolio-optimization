
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

def create_unconstrained_individual(
    rng: np.random.Generator,
    n_assets: int = 30,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Generate a feasible portfolio without cardinality constraints."""

    if n_assets < 1 or not (0 < max_weight <= 1):
        raise ValueError("Invalid portfolio parameters.")

    if n_assets * max_weight < 1:
        raise ValueError("Infeasible maximum weight.")

    target = rng.dirichlet(np.ones(n_assets))

    low = np.min(target - max_weight) - 1.0
    high = np.max(target) + 1.0

    for _ in range(100):
        midpoint = (low + high) / 2.0
        weights = np.clip(
            target - midpoint,
            0.0,
            max_weight,
        )

        if weights.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint

    weights = np.clip(
        target - (low + high) / 2.0,
        0.0,
        max_weight,
    )

    return weights

def create_unconstrained_population(
    rng: np.random.Generator,
    population_size: int = 100,
    n_assets: int = 30,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Create a population without cardinality constraints."""

    if population_size < 1:
        raise ValueError("Population size must be positive.")

    return np.array([
        create_unconstrained_individual(
            rng,
            n_assets=n_assets,
            max_weight=max_weight,
        )
        for _ in range(population_size)
    ])

def unconstrained_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a child using arithmetic crossover."""

    if parent1.shape != parent2.shape:
        raise ValueError("Parent shapes must match.")

    alpha = rng.uniform(0.0, 1.0)

    child = alpha * parent1 + (1.0 - alpha) * parent2

    return child

def unconstrained_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Mutate portfolio weights without cardinality constraints."""

    child = individual.copy()

    if len(child) < 2:
        return child

    i, j = rng.choice(len(child), size=2, replace=False)

    combined_weight = child[i] + child[j]

    lower = max(0.0, combined_weight - max_weight)
    upper = min(max_weight, combined_weight)


    if rng.random() < 0.20:
        new_weight_i = rng.choice([lower, upper])
    else:
        new_weight_i = rng.uniform(lower, upper)


    child[i] = new_weight_i
    child[j] = combined_weight - new_weight_i

    return child

def create_unconstrained_next_generation(
    population: np.ndarray,
    variances: np.ndarray,
    rng: np.random.Generator,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
) -> np.ndarray:
    """Create the next generation for GMV without cardinality constraints."""

    if len(population) != len(variances):
        raise ValueError("Population and variance sizes must match.")

    if len(population) < 3:
        raise ValueError("Population must contain at least 3 individuals.")

    if not 0 <= crossover_probability <= 1:
        raise ValueError("Invalid crossover probability.")

    if not 0 <= mutation_probability <= 1:
        raise ValueError("Invalid mutation probability.")

    best_idx = int(np.argmin(variances))
    next_population = [population[best_idx].copy()]

    while len(next_population) < len(population):
        parent1 = tournament_selection(population, variances, rng)
        parent2 = tournament_selection(population, variances, rng)

        if rng.random() < crossover_probability:
            child = unconstrained_crossover(parent1, parent2, rng)
        else:
            child = parent1.copy()

        if rng.random() < mutation_probability:
            child = unconstrained_mutation(child, rng)

        next_population.append(child)

    return np.array(next_population)

def run_unconstrained_sga(
    covariance: np.ndarray,
    population_size: int = 100,
    generations: int = 100,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SGA for GMV without cardinality constraints."""

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if generations < 1:
        raise ValueError("Generations must be positive.")

    rng = np.random.default_rng(seed)

    population = create_unconstrained_population(
        rng,
        population_size=population_size,
        n_assets=covariance.shape[0],
        max_weight=0.25,
    )

    history = []

    for generation in range(generations + 1):
        variances = evaluate_population(population, covariance)
        history.append(float(np.min(variances)))

        if generation == generations:
            break

        population = create_unconstrained_next_generation(
            population,
            variances,
            rng,
        )

    best_index = int(np.argmin(variances))
    best_weights = population[best_index].copy()

    return best_weights, np.array(history)

def create_variable_cardinality_individual(
    rng: np.random.Generator,
    n_assets: int = 30,
    max_assets: int = 10,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Create a feasible portfolio with variable cardinality."""

    if n_assets < 1 or max_assets < 1:
        raise ValueError("Asset counts must be positive.")

    if not (0 < min_weight <= max_weight <= 1):
        raise ValueError("Invalid weight bounds.")

    min_assets = int(np.ceil((1.0 - 1e-12) / max_weight))
    upper_assets = min(
        n_assets,
        max_assets,
        int(np.floor((1.0 + 1e-12) / min_weight)),
    )

    if min_assets > upper_assets:
        raise ValueError("No feasible number of active assets.")

    k = int(rng.integers(min_assets, upper_assets + 1))

    return create_individual(
        rng,
        n_assets=n_assets,
        k=k,
        min_weight=min_weight,
        max_weight=max_weight,
    )

def variable_cardinality_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    max_assets: int = 10,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Crossover for portfolios with variable cardinality."""

    if parent1.shape != parent2.shape or parent1.ndim != 1:
        raise ValueError("Parents must have matching 1D shapes.")

    if not (0 < min_weight <= max_weight <= 1):
        raise ValueError("Invalid weight bounds.")

    if max_assets < 1:
        raise ValueError("Maximum asset count must be positive.")

    union = np.flatnonzero(
        (parent1 > 1e-8) | (parent2 > 1e-8)
    )

    min_assets = int(np.ceil(1.0 / max_weight - 1e-12))
    upper_assets = min(
        max_assets,
        len(union),
        int(np.floor(1.0 / min_weight + 1e-12)),
    )

    if min_assets > upper_assets:
        raise ValueError("No feasible offspring cardinality.")

    k = int(rng.integers(min_assets, upper_assets + 1))

    selected = rng.choice(union, size=k, replace=False)

    target = (parent1[selected] + parent2[selected]) / 2.0

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

def variable_cardinality_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    max_assets: int = 10,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Mutate portfolio composition while preserving feasibility."""

    weights = np.asarray(individual, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Individual must be a 1D array.")

    active = np.flatnonzero(weights > 1e-8)
    inactive = np.flatnonzero(weights <= 1e-8)

    min_assets = int(np.ceil((1.0 - 1e-12) / max_weight))
    upper_assets = min(
        len(weights),
        max_assets,
        int(np.floor((1.0 + 1e-12) / min_weight)),
    )

    if not min_assets <= len(active) <= upper_assets:
        raise ValueError("Individual has invalid cardinality.")

    operations = []

    if len(active) < upper_assets and len(inactive) > 0:
        operations.append("add")

    if len(active) > min_assets:
        operations.append("remove")

    if len(inactive) > 0:
        operations.append("replace")

    if not operations:
        return weights.copy()

    operation = str(rng.choice(operations))
    selected = active.copy()

    if operation == "add":
        new_asset = int(rng.choice(inactive))
        selected = np.append(selected, new_asset)

    elif operation == "remove":
        removed = int(rng.choice(active))
        selected = selected[selected != removed]

    else:
        removed = int(rng.choice(active))
        new_asset = int(rng.choice(inactive))
        selected = selected[selected != removed]
        selected = np.append(selected, new_asset)

    target = weights[selected].copy()

    # Project selected weights onto the bounded simplex.
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

    mutated = np.zeros_like(weights)
    mutated[selected] = np.clip(
        target - (low + high) / 2.0,
        min_weight,
        max_weight,
    )

    return mutated

def create_variable_cardinality_population(
    rng: np.random.Generator,
    population_size: int = 100,
    n_assets: int = 30,
    max_assets: int = 10,
) -> np.ndarray:
    """Create a population with variable asset cardinality."""

    if population_size < 1:
        raise ValueError("Population size must be positive.")

    return np.array([
        create_variable_cardinality_individual(
            rng,
            n_assets=n_assets,
            max_assets=max_assets,
        )
        for _ in range(population_size)
    ])


def create_variable_cardinality_next_generation(
    population: np.ndarray,
    variances: np.ndarray,
    rng: np.random.Generator,
    max_assets: int = 10,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
) -> np.ndarray:
    """Create the next generation using selection and elitism."""

    population_size = len(population)

    if population_size < 2:
        raise ValueError("Population must contain at least two individuals.")

    if len(variances) != population_size:
        raise ValueError("Variance count must match population size.")

    if not (0 <= crossover_probability <= 1):
        raise ValueError("Invalid crossover probability.")

    if not (0 <= mutation_probability <= 1):
        raise ValueError("Invalid mutation probability.")

    best_index = int(np.argmin(variances))
    next_population = [population[best_index].copy()]

    def select_parent():
        indices = rng.choice(
            population_size,
            size=3,
            replace=True,
        )
        winner = indices[np.argmin(variances[indices])]
        return population[winner].copy()

    while len(next_population) < population_size:
        parent1 = select_parent()
        parent2 = select_parent()

        if rng.random() < crossover_probability:
            child = variable_cardinality_crossover(
                parent1,
                parent2,
                rng,
                max_assets=max_assets,
            )
        else:
            child = parent1.copy()

        if rng.random() < mutation_probability:
            child = variable_cardinality_combined_mutation(
                child,
                rng,
                max_assets=max_assets,
            )

        next_population.append(child)

    return np.array(next_population)

def run_variable_cardinality_sga(
    covariance: np.ndarray,
    population_size: int = 100,
    generations: int = 100,
    seed: int = 2026,
    max_assets: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """Minimize portfolio variance with variable cardinality."""

    covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance must be a square matrix.")

    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance contains non-finite values.")

    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("Covariance must be symmetric.")

    if population_size < 2:
        raise ValueError("Population size must be at least two.")

    if generations < 0:
        raise ValueError("Generations cannot be negative.")

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

        index = int(np.argmin(variances))

        if variances[index] < best_variance:
            best_variance = float(variances[index])
            best_weights = population[index].copy()

        history.append(best_variance)

        if generation < generations:
            population = create_variable_cardinality_next_generation(
                population,
                variances,
                rng,
                max_assets=max_assets,
            )

    return best_weights, history

def variable_cardinality_weight_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Redistribute weights without changing the selected assets."""

    mutated = np.asarray(individual, dtype=float).copy()
    active = np.flatnonzero(mutated > 1e-8)

    if len(active) < 2:
        return mutated

    i, j = rng.choice(active, size=2, replace=False)

    combined = mutated[i] + mutated[j]

    lower = max(min_weight, combined - max_weight)
    upper = min(max_weight, combined - min_weight)

    if upper <= lower:
        return mutated

    new_weight = rng.uniform(lower, upper)

    mutated[i] = new_weight
    mutated[j] = combined - new_weight

    return mutated

def variable_cardinality_combined_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    max_assets: int = 10,
    structural_probability: float = 0.5,
) -> np.ndarray:
    """Apply structural or weight mutation."""

    if not 0.0 <= structural_probability <= 1.0:
        raise ValueError(
            "Structural mutation probability must be between 0 and 1."
        )

    if rng.random() < structural_probability:
        return variable_cardinality_mutation(
            individual,
            rng,
            max_assets=max_assets,
        )

    return variable_cardinality_weight_mutation(
        individual,
        rng,
    )
