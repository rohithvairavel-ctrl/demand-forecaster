"""Companion script for notebooks/02_forecasting.ipynb."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecaster.backtest import run_backtest_suite  # noqa: E402
from demand_forecaster.data import load_demand  # noqa: E402


def main() -> None:
    _, y = load_demand(series="item_1")
    results = run_backtest_suite(y, horizon=28, min_train=365, step=28)
    lb = pd.DataFrame([{**v["overall"], "model": k} for k, v in results.items()]).sort_values("mae")
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
