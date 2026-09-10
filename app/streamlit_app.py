"""Streamlit demo: pick series/horizon, view history + forecast + metrics."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecaster.backtest import holdout_eval  # noqa: E402
from demand_forecaster.config import DEFAULT_SERIES, HORIZON, REPORTS_DIR  # noqa: E402
from demand_forecaster.data import available_series, load_demand  # noqa: E402
from demand_forecaster.models import default_model_zoo  # noqa: E402

st.set_page_config(page_title="Demand Forecaster", layout="wide")
st.title("Demand Forecaster")
st.caption("Retail-style daily demand forecasting with time-aware validation")


@st.cache_data
def _load():
    return load_demand()


def main() -> None:
    try:
        df, default_y = _load()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python scripts/download_data.py` then `python scripts/train.py`.")
        return

    series_names = available_series(df)
    metrics_path = REPORTS_DIR / "metrics.json"
    saved = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    with st.sidebar:
        st.header("Controls")
        series = st.selectbox(
            "Series",
            series_names,
            index=series_names.index(DEFAULT_SERIES) if DEFAULT_SERIES in series_names else 0,
        )
        horizon = st.slider("Forecast horizon (days)", 7, 90, HORIZON, 7)
        model_name = st.selectbox("Model", list(default_model_zoo().keys()), index=2)
        test_size = st.slider("Holdout size (days)", 28, 180, 90, 7)
        run = st.button("Run forecast", type="primary")

    y = df[series].astype(float)
    y.index = pd.to_datetime(y.index)

    col1, col2, col3 = st.columns(3)
    col1.metric("Observations", f"{len(y):,}")
    col2.metric("Start", str(y.index.min().date()))
    col3.metric("End", str(y.index.max().date()))

    if saved:
        st.subheader("Saved backtest leaderboard")
        lb = pd.DataFrame(saved["backtest"]["leaderboard"])
        st.dataframe(lb, use_container_width=True)
        st.caption(f"Best model from last train run: **{saved.get('best_model')}**")

    st.subheader("Historical demand")
    st.line_chart(y.rename("demand"))

    if run:
        zoo = default_model_zoo()
        model = deepcopy(zoo[model_name])
        with st.spinner(f"Fitting {model_name}..."):
            result = holdout_eval(y, model, test_size=min(test_size, max(28, len(y) // 5)))
            fwd_model = deepcopy(zoo[model_name])
            fwd_model.fit(y)
            forward = fwd_model.predict(horizon)

        m = result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{m['mae']:.3f}")
        c2.metric("RMSE", f"{m['rmse']:.3f}")
        c3.metric("MAPE %", f"{m['mape']:.2f}")
        c4.metric("sMAPE %", f"{m['smape']:.2f}")

        st.subheader("Holdout: actual vs forecast")
        holdout_df = pd.DataFrame({"actual": result["test"], "forecast": result["pred"]})
        st.line_chart(holdout_df)

        st.subheader(f"Forward {horizon}-day forecast")
        hist_tail = y.iloc[-120:].rename("history")
        chart = pd.concat([hist_tail, forward.rename("forecast")], axis=1)
        st.line_chart(chart)

        if getattr(result["model"], "feature_importances_", None):
            st.subheader("Feature drivers")
            imp = result["model"].feature_importances_
            imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
            st.bar_chart(imp_df.set_index("feature").head(15))


if __name__ == "__main__":
    main()
