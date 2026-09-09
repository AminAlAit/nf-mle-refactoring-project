# config file for modeling paramsfrom pathlib import Path

from pathlib import Path

LEAKAGE_COLUMNS = ["price", "sqft_price", "date", "delta_lat", "delta_long"]
ID_COLUMN = "id"
TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.3
PARAM_GRID = {
    "model__regressor__alpha": [0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}
MODEL_PATH = Path("model/model.bin")
PREPROCESSING_PATH = Path("model/preprocessing.bin")