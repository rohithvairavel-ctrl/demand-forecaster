# Demand Forecaster

Resume-ready **time-series demand forecasting** portfolio project: retail-style daily unit sales, leak-free validation, strong baselines, gradient boosting / ridge on lag features, and a Streamlit demo recruiters can run locally.

**Repo:** https://github.com/rohithvairavel-ctrl/demand-forecaster

## Problem

Store / SKU planners need short-horizon forecasts to set inventory and staffing. Classic ML CV leaks future information on time series — this project uses **expanding-window backtests** and a chronological holdout.

## Dataset

Primary source (public CSV):

- [skforecast simulated item sales](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/simulated_items_sales.csv) — daily demand for 3 items (~2012–2015), retail-like weekly seasonality.

```bash
python scripts/download_data.py
```

A small committed sample lives in `data/sample/` so the app/notebooks work offline after clone. If the primary URL is unreachable, the downloader falls back to Prophet’s monthly retail sales example and documents that in `data/raw/dataset_meta.json`.

## Approach

1. **Features:** lags (1–28d), rolling mean/std/min/max (7/14/28), calendar + Fourier-style DOW/month encodings.
2. **Baselines:** naive (last value), seasonal naive (weekly).
3. **Models:** Ridge (standardized) and LightGBM on the same supervised frame; recursive multi-step forecasts.
4. **Validation:** expanding-origin backtest (horizon 28, step 28, min train 365) + final chronological holdout.
5. **Metrics:** MAE, RMSE, MAPE, sMAPE.
6. **Explainability:** LightGBM gain / Ridge |coef| feature importance charts.

## Results

After `python scripts/train.py`, see `reports/metrics.json` and `reports/figures/*.svg`.

> Metrics are produced by the training script (not hand-written). Re-run train locally to refresh numbers after changing models.

## Project layout

```
app/streamlit_app.py          # interactive demo
scripts/download_data.py     # fetch public CSV
scripts/train.py             # backtest + fit + figures + models
src/demand_forecaster/       # load, features, models, backtest, metrics
notebooks/01_eda.ipynb
notebooks/02_forecasting.ipynb
reports/metrics.json
reports/figures/*.svg
models/*.joblib(.b64)        # artifacts (+ text sidecar for GitHub)
data/sample/                 # small committed CSV
```

## Quickstart

```bash
git clone https://github.com/rohithvairavel-ctrl/demand-forecaster.git
cd demand-forecaster
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/train.py --series item_1
streamlit run app/streamlit_app.py
```

## How to read the metrics

| Model | Role |
|-------|------|
| `naive` | Last-value baseline |
| `seasonal_naive` | Weekly seasonal baseline |
| `ridge` | Linear model on lags + calendar |
| `lightgbm` | Gradient boosting on same features |

Lower MAE / RMSE / MAPE is better. Prefer models that beat **seasonal_naive** on the expanding-window leaderboard — that is the bar for “learned something beyond seasonality.”

## Design notes (interview talking points)

- Features for day *t* only use information ≤ *t−1* (lags + shifted rolling windows).
- Backtest origins expand forward; each fold retrains from scratch.
- Recursive forecasting feeds predictions back as lags for multi-step horizons.
- Streamlit lets you switch series (`item_1` / `item_2` / `item_3`), horizon, and model live.

## License

Portfolio / educational use. Dataset © skforecast simulated series (see upstream repo).
