"""Unit tests for feasible cardinality-constrained GMV candidate search."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cardinality_gmv import solve_cardinality_heuristic, solve_support


class TestCardinalityGMV(unittest.TestCase):
    def test_at_most_k_and_weights(self):
        cov = np.diag([.03, .04, .05, .09, .12, .2, .3])
        weights, info = solve_cardinality_heuristic(cov, k=4, max_weight=.4,
                                                   min_weight=.02, local_passes=1)
        self.assertLessEqual(np.count_nonzero(weights > 1e-8), 4)
        self.assertAlmostEqual(weights.sum(), 1, places=7)
        self.assertTrue(np.all(weights >= -1e-8))
        self.assertTrue(np.all(weights <= .4 + 1e-7))
        self.assertTrue(np.all(weights[weights > 1e-8] >= .02 - 1e-7))
        self.assertFalse(info["global_optimality_certified"])

    def test_infeasible(self):
        with self.assertRaises(ValueError):
            solve_cardinality_heuristic(np.eye(8), k=3, max_weight=.25)

    def test_repeatable(self):
        cov = np.diag(np.linspace(.01, .08, 8))
        a, _ = solve_cardinality_heuristic(cov, k=5, max_weight=.4, seed=7,
                                          local_passes=1)
        b, _ = solve_cardinality_heuristic(cov, k=5, max_weight=.4, seed=7,
                                          local_passes=1)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_fixed_support_zero_outside(self):
        cov = np.eye(6)
        weights = solve_support(cov, (0, 2, 4, 5), .05, .4)
        self.assertIsNotNone(weights)
        self.assertEqual(np.count_nonzero(weights > 1e-8), 4)
        self.assertEqual(weights[1], 0)
        self.assertEqual(weights[3], 0)
        self.assertAlmostEqual(weights.sum(), 1, places=7)

    def test_covariance_validation(self):
        with self.assertRaises(ValueError):
            solve_cardinality_heuristic(np.array([[1., 2.], [2., 1.]]),
                                        k=2, max_weight=1)


if __name__ == "__main__":
    unittest.main()
