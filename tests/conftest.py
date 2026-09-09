"""
Shared fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from kc.config import RAW_DATA_PATH
from kc.data import load_raw


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """
    Six rows with one of each problem: missing view, missing waterfront,
    "?" basements, a renovated house, a waterfront house, and the 33-bedroom one.
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "date": ["10/13/2014"] * 6,
            "price": [300000.0, 550000.0, 420000.0, 700000.0, 1500000.0, 640000.0],
            "bedrooms": [3, 4, 2, 3, 5, 33],
            "bathrooms": [1.5, 2.5, 1.0, 2.0, 3.5, 2.0],
            "sqft_living": [1500, 2200, 1000, 1800, 3200, 1620],
            "sqft_lot": [5000, 6000, 3000, 5500, 12000, 6000],
            "floors": [1.0, 2.0, 1.0, 2.0, 2.0, 1.0],
            "waterfront": [0.0, 0.0, 0.0, np.nan, 1.0, 0.0],
            "view": [0.0, 2.0, np.nan, 0.0, 4.0, 0.0],
            "condition": [3, 4, 3, 3, 5, 3],
            "grade": [7, 8, 6, 8, 11, 7],
            "sqft_above": [1500, 1700, 1000, 1300, 3200, 1620],
            "sqft_basement": ["0.0", "500.0", "?", "?", "0.0", "0.0"],
            "yr_built": [1955, 1978, 1930, 1999, 2005, 1960],
            "yr_renovated": [0.0, 2003.0, np.nan, 0.0, np.nan, 0.0],
            "zipcode": [98103, 98115, 98118, 98052, 98039, 98106],
            "lat": [47.6567, 47.6810, 47.5620, 47.6790, 47.6270, 47.5480],
            "long": [-122.3419, -122.3120, -122.2870, -122.1210, -122.2400, -122.3600],
            "sqft_living15": [1450, 2100, 1100, 1750, 3000, 1600],
            "sqft_lot15": [5100, 6200, 3200, 5400, 11000, 6100],
        }
    )


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """
    The real dataset, skipped if it isn't there.
    """
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Dataset not available at {RAW_DATA_PATH}")
    return load_raw()


@pytest.fixture
def sample_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    1500 real rows: big enough to split and fit, small enough to stay quick.
    """
    return raw_data.sample(1500, random_state=1).reset_index(drop=True)
