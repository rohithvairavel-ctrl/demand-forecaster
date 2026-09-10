"""Lag, calendar, and rolling features for tabular forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAGS, ROLLING_WINDOWS


def add_calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["dow"] = idx.dayofweek
    df["dom"] = idx.day
    df["month"] = idx.month
    df["weekofyear"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # cyclic encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_supervised(
    y: pd.Series,
    lags: tuple[int, ...] = LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Build feature matrix aligned to target date (no future leakage)."""
    y = y.astype(float).copy()
    y.name = y.name or "y"
    frame = pd.DataFrame({"y": y})

    for lag in lags:
        frame[f"lag_{lag}"] = y.shift(lag)

    for w in rolling_windows:
        # shift(1) so rolling stats use only past values
        rolled = y.shift(1).rolling(w)
        frame[f"roll_mean_{w}"] = rolled.mean()
        frame[f"roll_std_{w}"] = rolled.std()
        frame[f"roll_min_{w}"] = rolled.min()
        frame[f"roll_max_{w}"] = rolled.max()

    cal = add_calendar_features(frame.index)
    frame = pd.concat([frame, cal], axis=1)
    frame = frame.dropna()
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "y"]


def xy_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feats = feature_columns(frame)
    return frame[feats], frame["y"]
