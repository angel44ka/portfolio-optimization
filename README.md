# Portfolio Optimization: Markowitz and Genetic Algorithms

This repository contains a research project on investment portfolio optimization using the Markowitz mean-variance framework, numerical optimization, and genetic algorithms.

## Project Structure

- `docs/` — Theoretical background and supporting figures.
- `research/src/` — Data preparation and portfolio optimization implementations.
- `research/tests/` — Automated tests for optimization methods.
- `research/data/` — Preliminary asset universe and locally generated datasets.
- `research/results/` — Locally generated optimization results.
- `research/README.md` — Computational research documentation.

## Current Implementation

- Historical adjusted price data retrieval and daily return calculation.
- Chronological training, validation, and testing datasets.
- Global Minimum Variance (GMV) portfolio optimization using Sequential Least Squares Programming (SLSQP).
- Cardinality-constrained GMV optimization using heuristic subset search and SLSQP.
- Automated unit tests for both optimization approaches.

## Portfolio Constraints

The cardinality-constrained model currently uses:

- 30 candidate stocks.
- At most 10 selected stocks.
- Fully invested capital.
- Long-only positions.
- Minimum selected-asset weight of 2%.
- Maximum asset weight of 25%.

The cardinality-constrained heuristic does not provide a certified global optimum.

## Data and Methodology

Historical adjusted prices are retrieved using the `yfinance` Python library.

The preliminary stock universe is stored in `research/data/selected_tickers.txt`.

The dataset covers January 2017 through December 2025 and is divided chronologically into training, validation, and testing periods.

The current stock selection may be affected by survivorship and selection bias. Complete historical price coverage alone does not establish the historical validity of the selected universe.

## Installation

Install the required Python dependencies from the repository root:

`python3 -m pip install -r requirements.txt`

## Running the Project

All commands below should be executed from the repository root.

**Prepare historical data:**

`python3 research/src/data_preparation.py`

**Run GMV optimization without a cardinality constraint:**

`python3 research/src/markowitz.py`

**Run cardinality-constrained GMV optimization:**

`python3 research/src/cardinality_gmv.py`

**Run automated tests:**

`python3 -m unittest discover -s research/tests -v`

## Research Status

The current implementation includes two portfolio optimization approaches: conventional GMV optimization and cardinality-constrained heuristic optimization.

Standard Genetic Algorithm (SGA), Modified Genetic Algorithm (MGA), additional optimization benchmarks, computational time comparisons, and out-of-sample performance evaluation are planned.

The current findings are preliminary research results and should not be interpreted as investment recommendations.