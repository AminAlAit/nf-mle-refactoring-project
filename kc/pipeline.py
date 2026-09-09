"""
The pipelines. Notebook cells 100-105 and 117.

A fitted build_model_pipeline() takes a frame with the raw CSV columns and returns
a price: cleaning, features, scaling and all. That's what lets the API serve it
without a second copy of the preprocessing.
"""

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from kc.config import EXCLUDED_FROM_MODEL, POLYNOMIAL_DEGREE
from kc.transformers import (
    CenterDistance,
    ColumnPruner,
    RowwiseCleaner,
    WaterDistance,
)


def build_feature_pipeline(drop: tuple[str, ...] = EXCLUDED_FROM_MODEL) -> Pipeline:
    """
    Clean, derive the distances, pick the model's columns.
    """
    return Pipeline(
        [
            ("clean", RowwiseCleaner()),
            ("center_distance", CenterDistance()),
            ("water_distance", WaterDistance()),
            ("select", ColumnPruner(drop=drop)),
        ]
    )


def build_baseline_pipeline(variables: list[str]) -> Pipeline:
    """
    Plain linear regression on a few columns. Cells 86-98.

    Worth keeping: without a baseline, "explains 84% of the variance" has nothing
    to be compared against.
    """
    return Pipeline(
        [
            ("features", build_feature_pipeline()),
            ("select", ColumnPruner(keep=tuple(variables))),
            ("model", LinearRegression()),
        ]
    )


def build_model_pipeline(
    degree: int = POLYNOMIAL_DEGREE,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-4,
) -> Pipeline:
    """
    The real model. Cell 117.

    PolynomialFeatures adds squared and interaction terms (19 columns -> 209).
    StandardScaler goes after it, since ElasticNet's penalty treats every
    coefficient the same and a squared sqft is nowhere near a bedroom count.
    Both are inside the pipeline so cross-validation fits them per fold instead of
    leaking the validation fold's distribution into training.
    TransformedTargetRegressor scales the price too, so the alpha grid means the
    same thing everywhere, and converts predictions back to dollars.
    """
    return Pipeline(
        [
            ("features", build_feature_pipeline()),
            ("polynomial", PolynomialFeatures(degree, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        tol=tol,
                        precompute=True,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )
