"""
Constants, so the same number isn't written in three places.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "King_County_House_prices_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.bin"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Cleaning

# The notebook drops row 15856 by position (cells 26-28) because it says 33 bedrooms
# in 1620 sqft. Dropping by position breaks if the frame gets sorted or filtered
# first, so we drop by the condition instead. Next highest is 11 bedrooms, so
# anything between 12 and 32 removes exactly that one row.
MAX_PLAUSIBLE_BEDROOMS = 30

# view (63 missing) and waterfront (2376 missing) are nearly all 0, so missing = 0.
ZERO_FILLED_COLUMNS = ("view", "waterfront")

# Geography

# Bill Gates' house in Medina, the notebook's "centre of wealth".
WEALTH_CENTER_LAT = 47.62774
WEALTH_CENTER_LONG = -122.24194

# Cell 60 measures the offset from 47.62774 but corrects longitude with cos(47.6219).
# Probably a typo, but the difference is tiny (7e-5), so keep the notebook's value
# and our numbers stay comparable to theirs.
LON_CORRECTION_LAT = 47.6219

EARTH_RADIUS_KM = 6378

# Modeling

TARGET = "price"
IDENTIFIER = "id"

# price is the target, sqft_price is derived from it (leak), date is unencoded text,
# delta_lat/long are already covered by center_distance, and id means nothing.
# The notebook drops the first five in cell 78 but leaves id in X until cell 101.
EXCLUDED_FROM_MODEL = (
    "price",
    "sqft_price",
    "date",
    "delta_lat",
    "delta_long",
    "id",
)

TEST_SIZE = 0.3
RANDOM_STATE = 42
POLYNOMIAL_DEGREE = 2

ELASTICNET_PARAM_GRID = {
    "model__regressor__alpha": [0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}
CV_FOLDS = 5
