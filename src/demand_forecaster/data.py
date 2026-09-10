"""Data loading and preparation for daily demand series."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE, DEFAULT_SERIES


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date index + demand columns."""
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = None
    for key in ("date", "ds", "datetime", "day"):
        if key in cols:
            date_col = cols[key]
            break
    if date_col is None:
        raise ValueError(f"No date column found in {list(df.columns)}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.set_index(date_col).sort_index()
    out.index.name = "date"

    # Keep numeric demand columns
    numeric = out.select_dtypes(include="number")
    if numeric.empty:
        # prophet-style y
        if "y" in {c.lower() for c in out.columns}:
            ycol = [c for c in out.columns if c.lower() == "y"][0]
            numeric = out[[ycol]].rename(columns={ycol: "demand"})
        else:
            raise ValueError("No numeric demand columns found")
    return numeric.astype(float)


def load_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return _normalize_frame(df)


def available_series(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def get_series(df: pd.DataFrame, name: str | None = None) -> pd.Series:
    name = name or (DEFAULT_SERIES if DEFAULT_SERIES in df.columns else df.columns[0])
    if name not in df.columns:
        raise KeyError(f"Series '{name}' not in {list(df.columns)}")
    s = df[name].astype(float).copy()
    s.name = name
    # fill tiny gaps with interpolation then ffill/bfill
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    s = s.interpolate(limit=3).ffill().bfill()
    s.index.name = "date"
    return s


def resolve_data_path() -> Path:
    """Prefer processed, then raw, then committed sample."""
    for folder in (DATA_PROCESSED, DATA_RAW, DATA_SAMPLE):
        candidates = sorted(folder.glob("*.csv"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        "No CSV found under data/. Run: python scripts/download_data.py"
    )


def load_demand(
    path: Path | str | None = None,
    series: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path) if path else resolve_data_path()
    df = load_csv(path)
    s = get_series(df, series)
    return df, s


def save_processed(df: pd.DataFrame, name: str = "demand_daily.csv") -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / name
    tmp = df.reset_index()
    tmp.to_csv(out, index=False)
    return out
