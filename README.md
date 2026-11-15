# Portfolio Optimization: Markowitz and Genetic Algorithms

This repository contains a research project on investment portfolio optimization using quadratic programming and genetic algorithms.

## Current Implementation

- Preparation of historical stock price data and daily returns.
- Chronological training, validation, and testing datasets.
- Global Minimum Variance (GMV) portfolio optimization using SLSQP.
- Cardinality-constrained GMV optimization using a heuristic search method.
- Automated unit tests for both optimization methods.

## Portfolio Constraints

The cardinality-constrained model currently uses:

- 30 candidate stocks.
- A maximum of 10 selected stocks.
- Long-only portfolio weights.
- Fully invested capital.
- Minimum weight of 2% for selected assets.
- Maximum weight of 25% per asset.

The cardinality-constrained heuristic does not guarantee a global optimum.

## Data

Historical adjusted prices are retrieved using the yfinance Python library.

The preliminary stock universe is stored in `data/selected_tickers.txt`.

The asset selection may be affected by survivorship and selection bias.

## Installation

Install the required dependencies:

`python3 -m pip install -r requirements.txt`

## Running the Project

Prepare historical data:

`python3 src/data_preparation.py`

Run the GMV optimization:

`python3 src/markowitz.py`

Run the cardinality-constrained GMV optimization:

`python3 src/cardinality_gmv.py`

Run the automated tests:

`python3 -m unittest discover -s tests -v`

## Research Status

Implementation of standard and modified genetic algorithms, further optimization benchmarks, and out-of-sample evaluation is planned.

The current results are preliminary and are not investment recommendations.