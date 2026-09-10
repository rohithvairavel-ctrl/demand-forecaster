"""Companion script for notebooks/01_eda.ipynb."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecaster.data import load_demand  # noqa: E402


def main() -> None:
    df, y = load_demand(series="item_1")
    print(df.head())
    print(y.describe())
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    y.plot(ax=axes[0], title="Daily demand (item_1)")
    y.groupby(y.index.dayofweek).mean().plot(kind="bar", ax=axes[1], title="Avg by DOW")
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "eda_overview.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
