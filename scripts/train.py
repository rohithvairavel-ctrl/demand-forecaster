#!/usr/bin/env python3
"""Train baselines + ML models, run expanding-window backtest, save artifacts."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecaster.backtest import (  # noqa: E402
    holdout_eval,
    run_backtest_suite,
    time_train_test_split,
)
from demand_forecaster.config import (  # noqa: E402
    BACKTEST_STEP,
    DEFAULT_SERIES,
    FIGURES_DIR,
    HORIZON,
    MIN_TRAIN_SIZE,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_SIZE,
)
from demand_forecaster.data import load_demand  # noqa: E402
from demand_forecaster.models import (  # noqa: E402
    default_model_zoo,
    make_lightgbm,
    make_ridge,
)


def plot_forecast(train, test, pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(train.index[-180:], train.iloc[-180:], label="train (tail)", color="#4C78A8")
    ax.plot(test.index, test.values, label="actual", color="#333333")
    ax.plot(pred.index, pred.values, label="forecast", color="#F58518", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("demand")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_backtest_bars(summary: dict, path: Path) -> None:
    names = list(summary.keys())
    maes = [summary[n]["overall"]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#54A24B" if m == min(maes) else "#4C78A8" for m in maes]
    ax.bar(names, maes, color=colors)
    ax.set_ylabel("MAE (lower better)")
    ax.set_title("Expanding-window backtest MAE by model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_importance(imp: dict, path: Path, top_k: int = 15) -> None:
    items = list(imp.items())[:top_k]
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, vals, color="#72B7B2")
    ax.set_title("Feature importance / |coef| (top drivers)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_model_b64(obj, path: Path) -> None:
    raw = joblib.dumps(obj)
    path.write_text(base64.b64encode(raw).decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train demand forecasting models")
    parser.add_argument("--series", default=DEFAULT_SERIES)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN_SIZE)
    parser.add_argument("--step", type=int, default=BACKTEST_STEP)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, y = load_demand(series=args.series)
    print(f"Loaded series={y.name} n={len(y)} [{y.index.min().date()} → {y.index.max().date()}]")

    # --- Expanding-window backtest ---
    print("Running expanding-window backtest...")
    bt = run_backtest_suite(
        y, horizon=args.horizon, min_train=args.min_train, step=args.step
    )
    leaderboard = []
    for name, res in bt.items():
        row = {"model": name, **res["overall"]}
        leaderboard.append(row)
        print(f"  {name:16s} MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} MAPE={row['mape']:.2f}%")

    leaderboard = sorted(leaderboard, key=lambda r: r["mae"])
    best_name = leaderboard[0]["model"]
    print(f"Best backtest model: {best_name}")

    plot_backtest_bars(bt, FIGURES_DIR / "backtest_mae.svg")

    # --- Holdout eval + final fit ---
    zoo = default_model_zoo()
    holdouts = {}
    for name, proto in zoo.items():
        model = deepcopy(proto)
        result = holdout_eval(y, model, test_size=args.test_size)
        holdouts[name] = {
            "metrics": result["metrics"],
            "model_name": name,
        }
        plot_forecast(
            result["train"],
            result["test"],
            result["pred"],
            title=f"{name} holdout forecast ({y.name})",
            path=FIGURES_DIR / f"forecast_{name}.svg",
        )
        print(f"Holdout {name}: {result['metrics']}")

    # Fit best model on all data for deployment artifact
    best_model = deepcopy(zoo[best_name])
    best_model.fit(y)
    joblib.dump(
        {"model": best_model, "series": y.name, "trained_through": str(y.index.max().date())},
        MODELS_DIR / "best_model.joblib",
    )
    save_model_b64(
        {"model": best_model, "series": y.name, "trained_through": str(y.index.max().date())},
        MODELS_DIR / "best_model.joblib.b64",
    )

    # Also persist ridge + lightgbm for the app
    for label, factory in (("ridge", make_ridge), ("lightgbm", make_lightgbm)):
        try:
            m = factory()
            m.fit(y)
            joblib.dump(m, MODELS_DIR / f"{label}.joblib")
            save_model_b64(m, MODELS_DIR / f"{label}.joblib.b64")
            if getattr(m, "feature_importances_", None):
                plot_importance(m.feature_importances_, FIGURES_DIR / f"importance_{label}.svg")
        except Exception as exc:  # noqa: BLE001
            print(f"Skip {label}: {exc}")

    # Feature importance from best tabular model if available
    if getattr(best_model, "feature_importances_", None):
        plot_importance(best_model.feature_importances_, FIGURES_DIR / "importance_best.svg")

    report = {
        "series": y.name,
        "n_obs": int(len(y)),
        "start": str(y.index.min().date()),
        "end": str(y.index.max().date()),
        "backtest": {
            "horizon": args.horizon,
            "min_train": args.min_train,
            "step": args.step,
            "leaderboard": leaderboard,
            "folds_detail": {k: v["folds"] for k, v in bt.items()},
        },
        "holdout": {
            "test_size": args.test_size,
            "results": {k: v["metrics"] for k, v in holdouts.items()},
        },
        "best_model": best_name,
        "feature_importance_best": getattr(best_model, "feature_importances_", None),
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {metrics_path}")

    # Leaderboard CSV
    pd.DataFrame(leaderboard).to_csv(REPORTS_DIR / "leaderboard.csv", index=False)


if __name__ == "__main__":
    main()
