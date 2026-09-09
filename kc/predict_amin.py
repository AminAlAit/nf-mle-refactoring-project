"""
Scoring new houses with a saved model.

The pipeline does its own preprocessing, so callers pass the raw columns.
sqft_basement isn't one of them, since it gets rebuilt from sqft_living - sqft_above.
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from kc.config_amin import MODEL_PATH
from kc.modeling_amin import load_model

REQUIRED_INPUT_COLUMNS = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
)


@lru_cache(maxsize=4)
def get_model(path: str | Path = MODEL_PATH):
    """
    Load the model, cached because deserialising per request would be slow.
    """
    return load_model(path)


def to_frame(records: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    One house or a list of them into a DataFrame.
    """
    if isinstance(records, Mapping):
        records = [records]
    frame = pd.DataFrame(list(records))
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return frame


def predict_price(
    records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    model=None,
) -> list[float]:
    """
    Predicted sale price in dollars.
    """
    model = model if model is not None else get_model()
    return [round(float(value), 2) for value in model.predict(to_frame(records))]
