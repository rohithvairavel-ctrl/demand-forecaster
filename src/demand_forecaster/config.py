"""Project paths and default hyperparameters."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_SERIES = "item_1"
SEASONAL_PERIOD = 7  # weekly seasonality for daily retail
HORIZON = 28
TEST_SIZE = 90
MIN_TRAIN_SIZE = 365
BACKTEST_STEP = 28
LAGS = (1, 2, 3, 7, 14, 21, 28)
ROLLING_WINDOWS = (7, 14, 28)
RANDOM_STATE = 42

PRIMARY_DATA_URL = (
    "https://raw.githubusercontent.com/skforecast/skforecast-datasets/"
    "main/data/simulated_items_sales.csv"
)
FALLBACK_DATA_URL = (
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/"
    "example_retail_sales.csv"
)
