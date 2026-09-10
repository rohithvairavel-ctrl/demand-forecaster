"""Baselines and ML forecasting models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LAGS, RANDOM_STATE, ROLLING_WINDOWS, SEASONAL_PERIOD
from .features import add_calendar_features, feature_columns, make_supervised, xy_split

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False


class Forecaster(Protocol):
    name: str

    def fit(self, y: pd.Series) -> "Forecaster": ...

    def predict(self, horizon: int) -> pd.Series: ...


@dataclass
class NaiveForecaster:
    """Last-value (random walk) baseline."""

    name: str = "naive"
    _last: float | None = field(default=None, init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self._last = float(y.iloc[-1])
        self._last_date = pd.Timestamp(y.index[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        idx = pd.date_range(self._last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series([self._last] * horizon, index=idx, name=self.name)


@dataclass
class SeasonalNaiveForecaster:
    """Repeat last seasonal cycle (default weekly)."""

    season: int = SEASONAL_PERIOD
    name: str = "seasonal_naive"
    _history: pd.Series | None = field(default=None, init=False, repr=False)

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        if len(y) < self.season:
            raise ValueError(f"Need at least {self.season} observations")
        self._history = y.astype(float).copy()
        return self

    def predict(self, horizon: int) -> pd.Series:
        y = self._history
        last_date = pd.Timestamp(y.index[-1])
        vals = []
        for h in range(1, horizon + 1):
            vals.append(float(y.iloc[-self.season + ((h - 1) % self.season)]))
        idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(vals, index=idx, name=self.name)


def _feature_row_from_history(hist: list[float], idx_hist: list[pd.Timestamp], next_date: pd.Timestamp) -> pd.DataFrame:
    """Build a single feature row for next_date using only past values (no leakage)."""
    s = pd.Series(hist, index=pd.DatetimeIndex(idx_hist), dtype=float)
    row: dict[str, float] = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = float(s.iloc[-lag]) if len(s) >= lag else float("nan")
    for w in ROLLING_WINDOWS:
        window = s.iloc[-w:] if len(s) >= w else s
        row[f"roll_mean_{w}"] = float(window.mean())
        row[f"roll_std_{w}"] = float(window.std(ddof=0)) if len(window) > 1 else 0.0
        row[f"roll_min_{w}"] = float(window.min())
        row[f"roll_max_{w}"] = float(window.max())
    cal = add_calendar_features(pd.DatetimeIndex([next_date])).iloc[0]
    for k, v in cal.items():
        row[str(k)] = float(v)
    return pd.DataFrame([row], index=[next_date])


@dataclass
class TabularForecaster:
    """Recursive multi-step forecasting on lag features."""

    model: Any
    name: str = "tabular"
    _y: pd.Series | None = field(default=None, init=False, repr=False)
    _fitted: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)
    feature_importances_: dict[str, float] | None = field(default=None, init=False)

    def fit(self, y: pd.Series) -> "TabularForecaster":
        frame = make_supervised(y)
        X, target = xy_split(frame)
        self._feature_names = list(X.columns)
        self._fitted = self.model.fit(X, target)
        self._y = y.astype(float).copy()
        self._capture_importance(X.columns)
        return self

    def _capture_importance(self, columns) -> None:
        est = self._fitted
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model", est)
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            imp = np.abs(np.asarray(est.coef_, dtype=float).ravel())
        else:
            self.feature_importances_ = None
            return
        self.feature_importances_ = {
            str(c): float(v) for c, v in sorted(zip(columns, imp), key=lambda t: -t[1])
        }

    def predict(self, horizon: int) -> pd.Series:
        hist = list(self._y.astype(float).values)
        idx_hist = [pd.Timestamp(t) for t in self._y.index]
        preds: list[float] = []
        for _ in range(horizon):
            next_date = idx_hist[-1] + pd.Timedelta(days=1)
            X = _feature_row_from_history(hist, idx_hist, next_date)
            X = X.reindex(columns=self._feature_names)
            yhat = float(self._fitted.predict(X)[0])
            preds.append(yhat)
            hist.append(yhat)
            idx_hist.append(next_date)
        out_idx = pd.DatetimeIndex(idx_hist[-horizon:])
        return pd.Series(preds, index=out_idx, name=self.name)


def make_ridge(alpha: float = 1.0) -> TabularForecaster:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    return TabularForecaster(model=pipe, name="ridge")


def make_lightgbm(**kwargs) -> TabularForecaster:
    if not HAS_LGBM:
        raise ImportError("lightgbm is not installed")
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    params.update(kwargs)
    model = lgb.LGBMRegressor(**params)
    return TabularForecaster(model=model, name="lightgbm")


def default_model_zoo() -> dict[str, Forecaster]:
    zoo: dict[str, Forecaster] = {
        "naive": NaiveForecaster(),
        "seasonal_naive": SeasonalNaiveForecaster(),
        "ridge": make_ridge(),
    }
    if HAS_LGBM:
        zoo["lightgbm"] = make_lightgbm()
    return zoo
