"""Download and audit adjusted daily prices for a provisional asset universe.

Run from the repository root:
    python src/data_preparation.py

This script intentionally does not silently fill missing price observations,
replace tickers, or declare the selected universe free of selection bias.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import yfinance as yf

START = "2017-01-03"
END_EXCLUSIVE = "2026-01-01"
TRAIN_END = "2022-06-30"
VALIDATION_END = "2023-06-30"
TEST_END = "2025-12-31"

ROOT = Path(__file__).resolve().parents[1]
TICKERS_FILE = ROOT / "data" / "selected_tickers.txt"
OUTPUT = ROOT / "data" / "processed"


def load_tickers() -> list[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TICKERS_FILE}")
    tickers = [line.strip().upper() for line in TICKERS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tickers:
        raise ValueError("Ticker list is empty.")
    if len(tickers) != len(set(tickers)):
        raise ValueError("Duplicate tickers in selected_tickers.txt.")
    if any(not ticker.replace("-", "").replace(".", "").isalnum() for ticker in tickers):
        raise ValueError("Unexpected ticker characters.")
    return tickers


def get_adjusted_close(tickers: list[str]) -> pd.DataFrame:
    downloaded = yf.download(
        tickers=tickers,
        start=START,
        end=END_EXCLUSIVE,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if downloaded.empty:
        raise RuntimeError("No price data returned by Yahoo Finance.")
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise RuntimeError("Downloaded data has no Close field.")
        prices = downloaded["Close"].copy()
    else:
        if len(tickers) != 1 or "Close" not in downloaded.columns:
            raise RuntimeError("Unexpected Yahoo Finance response format.")
        prices = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.loc[~prices.index.duplicated()].sort_index()
    prices = prices.loc[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(TEST_END))]
    return prices.apply(pd.to_numeric, errors="coerce")


def main() -> None:
    tickers = load_tickers()
    print(f"Requested tickers: {len(tickers)}")
    prices = get_adjusted_close(tickers)
    if prices.empty:
        raise RuntimeError("No observations within the study date range.")
    # Observed union of returned trading days: this is not an independent exchange calendar.
    quality = pd.DataFrame({
        "ticker": tickers,
        "observations": [int(prices[t].notna().sum()) for t in tickers],
        "missing_observations": [int(prices[t].isna().sum()) for t in tickers],
        "nonpositive_observations": [int((prices[t].dropna() <= 0).sum()) for t in tickers],
        "first_valid_date": [str(prices[t].first_valid_index().date()) if prices[t].first_valid_index() is not None else "" for t in tickers],
        "last_valid_date": [str(prices[t].last_valid_index().date()) if prices[t].last_valid_index() is not None else "" for t in tickers],
    })
    quality["status"] = "REVIEW"
    quality.loc[(quality["missing_observations"] == 0) & (quality["nonpositive_observations"] == 0), "status"] = "COMPLETE_ON_OBSERVED_CALENDAR"
    quality.loc[quality["observations"] == 0, "status"] = "NO_DATA"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT / "adjusted_prices.csv", index_label="date")
    quality.to_csv(OUTPUT / "price_quality.csv", index=False)
    metadata = {
        "price_source": "Yahoo Finance via yfinance",
        "price_field": "Close after auto_adjust=True (split/dividend-adjusted as served)",
        "start": START,
        "end_exclusive": END_EXCLUSIVE,
        "observed_calendar_dates": len(prices),
        "selection_status": "provisional; historical membership/identity and survivorship bias not resolved",
        "missing_data_policy": "No imputation; halt before return calculation if missing/nonpositive prices",
        "partitions": {
            "train": [START, TRAIN_END],
            "validation": ["2022-07-01", VALIDATION_END],
            "test": ["2023-07-01", TEST_END],
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Observed price dates: {len(prices)}")
    print(quality[["ticker", "status", "missing_observations"]].to_string(index=False))

    if (quality["status"] != "COMPLETE_ON_OBSERVED_CALENDAR").any():
        print("\nSTOP: At least one ticker has missing or nonpositive data. Review price_quality.csv.")
        print("No returns calculated. Raw adjusted prices were saved for auditing.")
        return

    # Calculate returns after validating prices, and only then split by dates.
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not returns.index.is_monotonic_increasing or returns.isna().any().any():
        raise RuntimeError("Returns contain missing values or unordered dates.")
    partitions = {
        "train": returns.loc[:TRAIN_END],
        "validation": returns.loc["2022-07-01":VALIDATION_END],
        "test": returns.loc["2023-07-01":TEST_END],
    }
    for name, part in partitions.items():
        if part.empty:
            raise RuntimeError(f"Empty partition: {name}")
        part.to_csv(OUTPUT / f"returns_{name}.csv", index_label="date")
        print(f"{name}: {len(part)} daily return rows; {part.index.min().date()} to {part.index.max().date()}")
    returns.to_csv(OUTPUT / "returns_all.csv", index_label="date")
    print(f"\nSaved verified price-derived outputs to: {OUTPUT}")
    print("Note: price completeness does not establish security identity or unbiased universe selection.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
