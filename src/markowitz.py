"""Long-only Markowitz global minimum variance (GMV) baseline.

Run from repository root:
    python src/markowitz.py

Fits ONLY the training returns. Validation and test are not accessed.
The unconstrained-cardinality GMV is a baseline, NOT a K<=10 solution.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data' / 'processed' / 'returns_train.csv'
OUTPUT = ROOT / 'results' / 'mvo'
MAX_WEIGHT = 0.25
TRADING_DAYS = 252


def estimate_covariance(training_returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance of daily simple returns, estimated on training only."""
    if training_returns.shape[0] < 2 or training_returns.shape[1] < 2:
        raise ValueError('Need at least 2 days and 2 assets.')
    if training_returns.isna().any().any():
        raise ValueError('Missing values found in training returns.')
    arr = training_returns.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError('Non-finite training returns.')
    cov = np.cov(arr, rowvar=False, ddof=1)
    if not np.isfinite(cov).all():
        raise ValueError('Non-finite covariance.')
    # Symmetrize to eliminate harmless floating-point asymmetry.
    return (cov + cov.T) / 2


def fit_gmv(cov: np.ndarray, max_weight: float = MAX_WEIGHT) -> tuple[np.ndarray, dict]:
    """Minimize variance w.T @ cov @ w, sum(w)=1, 0<=w_i<=max_weight.

    With a positive-semidefinite covariance matrix, this is a convex QP.
    SLSQP success is checked with independent feasibility diagnostics.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError('Covariance must be square.')
    n = cov.shape[0]
    if not np.isfinite(cov).all():
        raise ValueError('Covariance contains invalid entries.')
    if max_weight <= 0 or max_weight > 1 or n * max_weight < 1 - 1e-12:
        raise ValueError('Infeasible box constraints.')
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError('Covariance is not symmetric.')
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError('Covariance is not positive semidefinite.')

    scale = max(float(np.max(np.abs(cov))), 1e-12)
    def objective(w):
        return float(w @ cov @ w / scale)
    def gradient(w):
        return 2 * (cov @ w) / scale

    x0 = np.full(n, 1/n)
    result = minimize(
        objective, x0, jac=gradient, method='SLSQP',
        bounds=[(0., max_weight)]*n,
        constraints=[{'type':'eq', 'fun':lambda w: np.sum(w)-1.,
                      'jac':lambda w: np.ones(n)}],
        options={'ftol': 1e-12, 'maxiter': 20000},
    )
    w = np.asarray(result.x, dtype=float)
    sum_error = abs(w.sum()-1)
    violation = max(0., -w.min(), w.max()-max_weight)
    if not result.success or sum_error > 1e-7 or violation > 1e-7:
        raise RuntimeError(f'GMV failed: {result.message}; sum_error={sum_error:.3g}; bound_violation={violation:.3g}')
    info = {
        'optimizer':'SLSQP', 'success':bool(result.success),
        'message':str(result.message), 'iterations':int(result.nit),
        'sum_weights':float(w.sum()), 'sum_error':float(sum_error),
        'bound_violation':float(violation),
        'daily_portfolio_variance':float(w @ cov @ w),
        'annualized_volatility_sqrt_252':float(np.sqrt(max(0., w @ cov @ w)*TRADING_DAYS)),
        'nonzero_weights_threshold_1e-6':int(np.count_nonzero(w > 1e-6)),
        'max_weight':float(max_weight),
        'cardinality_limit_applied':False,
    }
    return w, info


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing {INPUT}. Run src/data_preparation.py first.')
    returns = pd.read_csv(INPUT, index_col='date', parse_dates=True)
    if returns.columns.duplicated().any():
        raise ValueError('Duplicate ticker columns in training CSV.')
    tickers = returns.columns.tolist()
    cov = estimate_covariance(returns)
    weights, diagnostics = fit_gmv(cov)
    weights_table = pd.DataFrame({'ticker':tickers, 'weight':weights})
    weights_table = weights_table.sort_values('weight', ascending=False)
    diagnostics.update({
        'n_assets_universe':len(tickers),
        'training_rows':len(returns),
        'training_first':str(returns.index.min().date()),
        'training_last':str(returns.index.max().date()),
        'objective':'Minimum in-sample daily variance, long-only, fully invested, max 25% per asset',
        'note':'No K<=10 constraint; no transaction costs; no out-of-sample performance claims.',
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights_table.to_csv(OUTPUT/'gmv_weights.csv', index=False)
    pd.DataFrame(cov,index=tickers,columns=tickers).to_csv(OUTPUT/'train_covariance.csv',index_label='ticker')
    (OUTPUT/'gmv_diagnostics.json').write_text(json.dumps(diagnostics,indent=2), encoding='utf-8')
    print(f'Assets: {len(tickers)}; training days: {len(returns)}')
    print(f'Training period: {diagnostics["training_first"]} to {diagnostics["training_last"]}')
    print('GMV model (no cardinality constraint; long-only; max weight 25%)')
    print(weights_table.head(15).to_string(index=False, float_format=lambda x:f'{x:.6f}'))
    print(f'Sum of weights: {diagnostics["sum_weights"]:.10f}')
    print(f'Nonzero weights (>1e-6): {diagnostics["nonzero_weights_threshold_1e-6"]}')
    print(f'Annualized in-sample volatility: {diagnostics["annualized_volatility_sqrt_252"]:.4%}')
    print(f'Saved outputs: {OUTPUT}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
