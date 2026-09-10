"""Time-aware splits and rolling/expanding-window backtests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .config import BACKTEST_STEP, HORIZON, MIN_TRAIN_SIZE, TEST_SIZE
from .metrics import summarize
from .models import Forecaster, default_model_zoo


@dataclass
class SplitResult:
    train: pd.Series
    test: pd.Series


def time_train_test_split(y: pd.Series, test_size: int = TEST_SIZE) -> SplitResult:
    if len(y) <= test_size + 30:
        raise ValueError("Series too short for requested test_size")
    train = y.iloc[:-test_size]
    test = y.iloc[-test_size:]
    return SplitResult(train=train, test=test)


def expanding_origins(
    n: int,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN_SIZE,
    step: int = BACKTEST_STEP,
) -> list[tuple[int, int]]:
    """Return (train_end_exclusive, test_end_exclusive) index pairs."""
    origins = []
    train_end = min_train
    while train_end + horizon <= n:
        origins.append((train_end, train_end + horizon))
        train_end += step
    return origins


def backtest_model(
    y: pd.Series,
    model_factory: Callable[[], Forecaster],
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN_SIZE,
    step: int = BACKTEST_STEP,
) -> dict:
    y = y.astype(float)
    origins = expanding_origins(len(y), horizon=horizon, min_train=min_train, step=step)
    if not origins:
        raise ValueError("No valid backtest origins — increase series length or reduce min_train")

    all_true = []
    all_pred = []
    fold_metrics = []
    for i, (train_end, test_end) in enumerate(origins):
        train = y.iloc[:train_end]
        test = y.iloc[train_end:test_end]
        model = model_factory()
        model.fit(train)
        pred = model.predict(len(test))
        pred = pred.reindex(test.index)
        # align lengths if calendar edge cases
        pred = pred.iloc[: len(test)]
        m = summarize(test.values, pred.values)
        m["fold"] = i
        m["train_end"] = str(train.index[-1].date())
        fold_metrics.append(m)
        all_true.extend(test.values.tolist())
        all_pred.extend(pred.values.tolist())

    overall = summarize(all_true, all_pred)
    overall["folds"] = len(origins)
    overall["horizon"] = horizon
    return {
        "overall": overall,
        "folds": fold_metrics,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def run_backtest_suite(
    y: pd.Series,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN_SIZE,
    step: int = BACKTEST_STEP,
    models: dict[str, Forecaster] | None = None,
) -> dict[str, dict]:
    zoo = models or default_model_zoo()
    results = {}
    for name, prototype in zoo.items():

        def factory(p=prototype):
            return deepcopy(p)

        results[name] = backtest_model(
            y, factory, horizon=horizon, min_train=min_train, step=step
        )
        results[name]["model"] = name
    return results


def holdout_eval(
    y: pd.Series,
    model: Forecaster,
    test_size: int = TEST_SIZE,
) -> dict:
    split = time_train_test_split(y, test_size=test_size)
    model.fit(split.train)
    pred = model.predict(len(split.test)).reindex(split.test.index)
    metrics = summarize(split.test.values, pred.values)
    return {
        "metrics": metrics,
        "train": split.train,
        "test": split.test,
        "pred": pred,
        "model": model,
    }
