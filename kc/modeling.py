"""
Splitting, scoring, saving, loading. Notebook cells 78-131.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from kc.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def split_dataset(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split. Cells 78-82.

    X keeps everything except the target, id and date included. The pipeline drops
    what it can't use, and the error analysis still needs id and the coordinates.
    """
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} is not in the frame.")
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """
    R^2 penalised for the number of predictors.

    Plain R^2 never goes down when you add a column, even a useless one, so you
    can't use it to compare models of different sizes. Cells 90, 97 and 105 write
    this out by hand three times.
    """
    denominator = n_samples - n_features - 1
    if denominator <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n_samples - 1) / denominator


def count_model_features(model: BaseEstimator, X: pd.DataFrame) -> int:
    """
    How many features the estimator actually saw. Not X.shape[1], since the
    pipeline drops six columns then expands the rest into polynomial terms.
    """
    try:
        estimator = model.named_steps["model"]
    except (AttributeError, KeyError):
        estimator = model

    # TransformedTargetRegressor hides the real estimator on .regressor_.
    inner = getattr(estimator, "regressor_", estimator)
    coefficients = getattr(inner, "coef_", None)
    if coefficients is not None:
        return int(np.asarray(coefficients).reshape(-1).shape[0])
    return int(X.shape[1])


def evaluate(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, label: str = "model"
) -> dict:
    """
    Score a fitted model on held-out data.

    RMSE and MAE are in there next to R^2 because "explains 84% of the variance"
    and "is usually off by $143,000" are both true, and only one is useful.
    """
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    n_features = count_model_features(model, X)
    return {
        "label": label,
        "n_samples": len(y),
        "n_features": n_features,
        "r2": round(float(r2), 4),
        "adjusted_r2": round(float(adjusted_r2(r2, len(y), n_features)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, predictions))), 2),
        "mae": round(float(mean_absolute_error(y, predictions)), 2),
    }


def error_table(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Per-house errors in dollars and percent. Cells 108-110 and 123-125.
    """
    predictions = model.predict(X)
    errors = pd.DataFrame(
        {
            "price": y.to_numpy(),
            "price_prediction": np.round(predictions, 2),
            "latitude": X["lat"].to_numpy(),
            "longitude": X["long"].to_numpy(),
        },
        index=range(len(y)),
    )
    if "id" in X.columns:
        errors.insert(0, "id", X["id"].to_numpy())
    errors["price_difference"] = (errors["price_prediction"] - errors["price"]).round(2)
    errors["price_difference_percent"] = (
        errors["price_difference"] / errors["price"] * 100
    ).round(2)
    return errors


def save_model(model: BaseEstimator, path: str | Path = MODEL_PATH) -> Path:
    """
    Save the fitted pipeline with skops. Cell 131.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        sio.dump(model, file)
    return path


def load_model(path: str | Path = MODEL_PATH, trusted: list[str] | None = None):
    """
    Load a model saved by save_model.

    skops won't rebuild arbitrary classes unless you list them, which is the point
    of using it over pickle. Passing trusted=None accepts whatever the file
    contains, which is fine here because we wrote it but not for a file from
    elsewhere.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run `python -m kc.train` to create one."
        )
    if trusted is None:
        trusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=trusted)


def save_metrics(metrics: list[dict], path: str | Path = METRICS_PATH) -> Path:
    """
    Write the scores next to the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
