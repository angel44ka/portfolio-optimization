import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from markowitz import estimate_covariance, fit_gmv


class TestGMV(unittest.TestCase):
    def test_gmv_variance_and_constraints(self):
        cov=np.diag([0.01,0.04,0.09,0.16,0.25])
        w,meta=fit_gmv(cov,max_weight=.35)
        self.assertAlmostEqual(w.sum(),1,places=7)
        self.assertTrue(np.all(w >= -1e-7))
        self.assertTrue(np.all(w <= .35+1e-7))
        self.assertFalse(meta['cardinality_limit_applied'])
        self.assertLessEqual(w@cov@w, np.ones(5)@cov@np.ones(5)/25+1e-8)

    def test_identical_assets_equal_start(self):
        w,_=fit_gmv(np.eye(6)*.04,max_weight=.25)
        np.testing.assert_allclose(w, np.full(6,1/6),rtol=0,atol=1e-6)

    def test_infeasible_bounds(self):
        with self.assertRaises(ValueError):
            fit_gmv(np.eye(3),max_weight=.25)

    def test_nan_returns_rejected(self):
        x=pd.DataFrame([[.01,.02],[np.nan,.03]],columns=['A','B'])
        with self.assertRaises(ValueError):
            estimate_covariance(x)

    def test_covariance_matches_pandas(self):
        x=pd.DataFrame([[.01,.02],[.02,-.01],[.03,.05]],columns=['A','B'])
        np.testing.assert_allclose(estimate_covariance(x),x.cov().values)

if __name__ == '__main__':
    unittest.main()
