"""
sklearn wrappers so the cleaning and features can live inside a pipeline.

Two reasons to bother: WaterDistance learns its reference points during fit, so the
test split can't influence a training feature (the notebook builds that list from
the whole dataset), and the saved model then carries its own preprocessing.

Only row-preserving steps go here. drop_invalid_rows stays outside.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from kc.cleaning import clean_rowwise
from kc.config import WEALTH_CENTER_LAT, WEALTH_CENTER_LONG
from kc.features import (
    add_center_distance,
    add_water_distance,
    waterfront_reference,
)


class RowwiseCleaner(BaseEstimator, TransformerMixin):
    """
    The row-preserving half of the cleaning. Stateless.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "RowwiseCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return clean_rowwise(X)


class CenterDistance(BaseEstimator, TransformerMixin):
    """
    Adds delta_lat, delta_long and center_distance. Stateless.
    """

    def __init__(
        self,
        center_lat: float = WEALTH_CENTER_LAT,
        center_long: float = WEALTH_CENTER_LONG,
    ):
        self.center_lat = center_lat
        self.center_long = center_long

    def fit(self, X: pd.DataFrame, y=None) -> "CenterDistance":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_center_distance(X, self.center_lat, self.center_long)


class WaterDistance(BaseEstimator, TransformerMixin):
    """
    Adds water_distance, using the waterfront houses seen during fit.

    Fitting the reference set is also what makes single-house predictions possible:
    one house on its own has no waterfront neighbours to measure against.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "WaterDistance":
        reference = waterfront_reference(X)
        if reference.size == 0:
            raise ValueError("No waterfront == 1 rows in the training data.")
        self.reference_ = reference
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "reference_")
        return add_water_distance(X, reference=self.reference_)


class ColumnPruner(BaseEstimator, TransformerMixin):
    """
    Picks the model's columns and pins their order.

    Pass drop (remove these) or keep (an explicit list). The column list is fixed at
    fit time, so a frame that turns up later missing a column gets an error naming
    it rather than a silently wrong prediction.
    """

    def __init__(
        self, drop: tuple[str, ...] | None = None, keep: tuple[str, ...] | None = None
    ):
        self.drop = drop
        self.keep = keep

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnPruner":
        if self.keep is not None and self.drop is not None:
            raise ValueError("Pass either `drop` or `keep`, not both.")

        if self.keep is not None:
            missing = [column for column in self.keep if column not in X.columns]
            if missing:
                raise ValueError(f"Columns requested by `keep` are absent: {missing}")
            columns = list(self.keep)
        else:
            excluded = set(self.drop or ())
            columns = [column for column in X.columns if column not in excluded]

        self.columns_ = columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        missing = [column for column in self.columns_ if column not in X.columns]
        if missing:
            raise ValueError(f"Missing columns the model was fitted on: {missing}")
        return X.loc[:, self.columns_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)
