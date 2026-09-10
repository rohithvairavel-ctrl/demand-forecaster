"""Baselines and ML forecasting models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, SEASONAL_PERIOD
from .features import feature_columns, make_supervised, xy_split

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
    _index_freq: str = field(default="D", init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self._last = float(y.iloc[-1])
        self._last_date = y.index[-1]
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
        last_date = y.index[-1]
        vals = []
        for h in range(1, horizon + 1):
            vals.append(float(y.iloc[-self.season + ((h - 1) % self.season)]))
        idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(vals, index=idx, name=self.name)


@dataclass
class TabularForecaster:
    """Recursive multi-step forecasting on lag features."""

    model: Any
    name: str = "tabular"
    _y: pd.Series | None = field(default=None, init=False, repr=False)
    _fitted: Any = field(default=None, init=False, repr=False)
    feature_importances_: dict[str, float] | None = field(default=None, init=False)

    def fit(self, y: pd.Series) -> "TabularForecaster":
        frame = make_supervised(y)
        X, target = xy_split(frame)
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
        idx_hist = list(self._y.index)
        preds = []
        for _ in range(horizon):
            next_date = idx_hist[-1] + pd.Timedelta(days=1)
            series = pd.Series(hist, index=pd.DatetimeIndex(idx_hist), dtype=float)
            # extend with NaN placeholder so make_supervised can build row for next_date
            series.loc[next_date] = np.nan
            frame = make_supervised(series.ffill())
            # last row corresponds to next_date after ffill — use features only
            if next_date not in frame.index:
                # fallback: use last available feature row pattern with updated calendar
                row = frame.iloc[[-1]].copy()
            else:
                row = frame.loc[[next_date]]
            X = row[feature_columns(frame)]
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
            ("model", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
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
