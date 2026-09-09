# assemble cleaning + feature engineering steps into reusable preprocessing pipeline

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features.config import WEALTH_CENTER_LAT, WEALTH_CENTER_LONG
from src.data.cleaning import (
    drop_bad_bedroom_rows,
    fix_missing_values,
    fix_sqft_basement,
)
from src.features.feature_engineering import (
    WaterfrontDistanceAdder,
    add_distance_to_center_of_wealth,
    add_sqft_price,
)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-count-changing cleanup, applied once to the full raw dataset
    before any train/test split.

    Kept outside the sklearn Pipeline below on purpose: a step that
    drops rows can't be composed into a Pipeline whose final estimator
    calls `.fit(X, y)` — X and y would end up different lengths. It also
    isn't something you'd apply to a single new record at prediction
    time, since outlier removal only makes sense while cleaning a
    training set.
    """
    df = drop_bad_bedroom_rows(df)
    df = fix_sqft_basement(df)
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Preprocessing pipeline that applies data cleaning and feature engineering steps.
    """
    return Pipeline(
        [
            ("fix_missing_values", FunctionTransformer(fix_missing_values)),
            (
                "center_distance",
                FunctionTransformer(
                    add_distance_to_center_of_wealth,
                    kw_args={
                        "target_lat": WEALTH_CENTER_LAT,
                        "target_long": WEALTH_CENTER_LONG,
                    },
                ),
            ),
            ("water_distance", WaterfrontDistanceAdder()),
        ]
    )