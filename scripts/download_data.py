#!/usr/bin/env python3
"""Download public daily demand data (skforecast simulated items sales).

Primary source (public GitHub raw CSV):
  https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/simulated_items_sales.csv

Daily sales for 3 items (2012–2015). Documented as simulated retail-like demand
from the skforecast project (not Favorita/Kaggle proprietary dumps).

Falls back to Facebook Prophet example retail sales (monthly) only if the
primary download fails — monthly series is resampled note is written to README
metadata JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecaster.config import (  # noqa: E402
    DATA_RAW,
    DATA_SAMPLE,
    FALLBACK_DATA_URL,
    PRIMARY_DATA_URL,
)
from demand_forecaster.data import save_processed  # noqa: E402


def download(url: str, timeout: int = 60) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(r.text))


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    meta = {"source": None, "url": None, "notes": []}

    try:
        print(f"Downloading primary dataset:\n  {PRIMARY_DATA_URL}")
        df = download(PRIMARY_DATA_URL)
        meta["source"] = "skforecast/simulated_items_sales"
        meta["url"] = PRIMARY_DATA_URL
        meta["notes"].append(
            "Simulated daily item sales (3 SKUs) from skforecast-datasets; retail-like seasonality."
        )
        raw_path = DATA_RAW / "simulated_items_sales.csv"
    except Exception as exc:  # noqa: BLE001
        print(f"Primary download failed ({exc}); trying fallback retail sales...")
        df = download(FALLBACK_DATA_URL)
        # Rename prophet columns if needed
        rename = {}
        if "ds" in df.columns:
            rename["ds"] = "date"
        if "y" in df.columns:
            rename["y"] = "retail_sales"
        df = df.rename(columns=rename)
        meta["source"] = "prophet/example_retail_sales"
        meta["url"] = FALLBACK_DATA_URL
        meta["notes"].append(
            "Fallback: monthly US retail sales from Prophet examples (not daily)."
        )
        raw_path = DATA_RAW / "retail_sales_monthly.csv"

    df.to_csv(raw_path, index=False)
    print(f"Saved raw -> {raw_path} ({len(df)} rows)")

    # Commit a small sample (first 120 days) for offline demos
    sample = df.head(120)
    sample_path = DATA_SAMPLE / "demand_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Saved sample -> {sample_path}")

    # Normalize into processed daily panel
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    processed = df.set_index(date_col).sort_index()
    numeric = processed.select_dtypes(include="number")
    out = save_processed(numeric, "demand_daily.csv")
    meta["rows"] = int(len(numeric))
    meta["columns"] = list(numeric.columns)
    meta["start"] = str(numeric.index.min().date())
    meta["end"] = str(numeric.index.max().date())
    meta_path = DATA_RAW / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Processed -> {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
