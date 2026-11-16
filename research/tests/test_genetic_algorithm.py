
"""Unit tests for standard genetic algorithm components."""

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from genetic_algorithm import (
    create_individual,
    create_population,
    portfolio_variance,
    create_unconstrained_individual,
    create_unconstrained_population,
)


class TestGeneticAlgorithm(unittest.TestCase):

    def test_cardinality_individual(self):
        rng = np.random.default_rng(2026)
        weights = create_individual(rng)

        active = weights > 1e-8

        self.assertEqual(len(weights), 30)
        self.assertEqual(np.count_nonzero(active), 10)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue(np.all(weights[active] >= 0.02 - 1e-8))
        self.assertTrue(np.all(weights <= 0.25 + 1e-8))

    def test_cardinality_population(self):
        population = create_population(
            np.random.default_rng(2026),
            population_size=100,
        )

        self.assertEqual(population.shape, (100, 30))
        self.assertTrue(np.allclose(population.sum(axis=1), 1.0))
        self.assertTrue(
            np.all(np.count_nonzero(population > 1e-8, axis=1) == 10)
        )

    def test_unconstrained_individual(self):
        weights = create_unconstrained_individual(
            np.random.default_rng(2026)
        )

        self.assertEqual(len(weights), 30)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue(np.all(weights >= -1e-8))
        self.assertTrue(np.all(weights <= 0.25 + 1e-8))

    def test_unconstrained_population(self):
        population = create_unconstrained_population(
            np.random.default_rng(2026),
            population_size=100,
        )

        self.assertEqual(population.shape, (100, 30))
        self.assertTrue(np.allclose(population.sum(axis=1), 1.0))
        self.assertTrue(np.all(population >= -1e-8))
        self.assertTrue(np.all(population <= 0.25 + 1e-8))

    def test_portfolio_variance(self):
        weights = np.array([0.5, 0.5])
        covariance = np.array([
            [0.04, 0.01],
            [0.01, 0.09],
        ])

        result = portfolio_variance(weights, covariance)

        self.assertAlmostEqual(result, 0.0375)

import genetic_algorithm as ga


class TestGeneticOperators(unittest.TestCase):

    def check_feasibility(self, weights, cardinality):
        self.assertTrue(np.isclose(weights.sum(), 1.0, atol=1e-8))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= -1e-8))
        self.assertTrue(np.all(weights <= 0.25 + 1e-8))

        if cardinality:
            active = weights > 1e-8
            self.assertEqual(np.count_nonzero(active), 10)
            self.assertTrue(np.all(weights[active] >= 0.02 - 1e-8))

    def test_cardinality_crossover(self):
        rng = np.random.default_rng(2026)
        parents = ga.create_population(rng, population_size=2)

        for _ in range(100):
            child = ga.crossover(parents[0], parents[1], rng)
            self.check_feasibility(child, cardinality=True)

    def test_cardinality_mutation(self):
        rng = np.random.default_rng(2026)
        parent = ga.create_individual(rng)

        for _ in range(100):
            child = ga.mutate(parent, rng)
            self.check_feasibility(child, cardinality=True)

    def test_unconstrained_crossover(self):
        rng = np.random.default_rng(2026)
        parents = ga.create_unconstrained_population(
            rng, population_size=2
        )

        for _ in range(100):
            child = ga.unconstrained_crossover(
                parents[0], parents[1], rng
            )
            self.check_feasibility(child, cardinality=False)

    def test_unconstrained_mutation(self):
        rng = np.random.default_rng(2026)
        parent = ga.create_unconstrained_individual(rng)

        for _ in range(100):
            child = ga.unconstrained_mutation(parent, rng)
            self.check_feasibility(child, cardinality=False)

    def test_cardinality_elitism(self):
        rng = np.random.default_rng(2026)
        population = ga.create_population(rng)
        covariance = np.eye(30)
        variances = ga.evaluate_population(population, covariance)

        new_population = ga.create_next_generation(
            population, variances, rng
        )

        best = population[np.argmin(variances)]

        self.assertTrue(np.array_equal(new_population[0], best))
        self.assertEqual(new_population.shape, population.shape)
        for child in new_population:
            self.check_feasibility(child, cardinality=True)

    def test_unconstrained_elitism(self):
        rng = np.random.default_rng(2026)
        population = ga.create_unconstrained_population(rng)
        covariance = np.eye(30)
        variances = ga.evaluate_population(population, covariance)

        new_population = ga.create_unconstrained_next_generation(
            population, variances, rng
        )

        best = population[np.argmin(variances)]

        self.assertTrue(np.array_equal(new_population[0], best))
        self.assertEqual(new_population.shape, population.shape)
        for child in new_population:
            self.check_feasibility(child, cardinality=False)


class TestFullSGA(unittest.TestCase):
    """Integration tests for both SGA optimization modes."""

    def test_unconstrained_sga(self):
        covariance = np.array([
            [0.04, 0.01, 0.00, 0.00, 0.00],
            [0.01, 0.06, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.05, 0.01, 0.00],
            [0.00, 0.00, 0.01, 0.07, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.03],
        ])

        weights, history = ga.run_unconstrained_sga(
            covariance,
            population_size=20,
            generations=10,
            seed=2026,
        )

        self.assertEqual(len(weights), 5)
        self.assertEqual(len(history), 11)
        self.check_constraints_unconstrained(weights)
        self.assertTrue(np.all(np.isfinite(history)))
        self.assertTrue(np.all(np.diff(history) <= 1e-12))
        self.assertAlmostEqual(
            history[-1],
            ga.portfolio_variance(weights, covariance),
            places=10,
        )

    def check_constraints_unconstrained(self, weights):
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue(np.all(weights >= -1e-8))
        self.assertTrue(np.all(weights <= 0.25 + 1e-8))

    def test_cardinality_sga(self):
        covariance = np.eye(30) * 0.04

        weights, history = ga.run_sga(
            covariance,
            population_size=20,
            generations=10,
            seed=2026,
        )

        active = weights > 1e-8

        self.assertEqual(len(weights), 30)
        self.assertEqual(len(history), 11)
        self.assertEqual(np.count_nonzero(active), 10)
        self.assertTrue(np.all(weights[active] >= 0.02 - 1e-8))
        self.assertTrue(np.all(weights <= 0.25 + 1e-8))
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue(np.all(np.diff(history) <= 1e-12))
        self.assertAlmostEqual(
            history[-1],
            ga.portfolio_variance(weights, covariance),
            places=10,
        )

if __name__ == "__main__":
    unittest.main()
