# src/modeling/evaluate.py
"""
Evaluation utilities: metrics and residual/error analysis for the
trained King County house price model.

Rebuilds the notebook's repeated "Adj. R^2" calculation and the
error-analysis DataFrames (predicted vs. actual price, mapped by
id/lat/long) as reusable functions, plus a runnable script that
reproduces the exact train/test split used during training, applies the
saved fitted preprocessing pipeline, and scores the saved model on the
held-out test set.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import RegressorMixin


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Adjusted R^2 — penalizes for the number of features used. Mostly
    meaningful for simpler models; the final ElasticNet-on-polynomial
    model is scored with plain R^2 instead (see `evaluate_model`), since
    counting expanded polynomial columns isn't a clean degrees-of-freedom
    correction.
    """
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Scores a fitted model on held-out data."""
    return {
        "r2": model.score(X_test, y_test),
        "n_samples": X_test.shape[0],
        "n_features": X_test.shape[1],
    }


def error_analysis(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Builds a per-row residual table: predicted price, absolute
    difference, and percentage difference from the true price, alongside
    location and id so individual outliers can be looked up.

    `X_test`, `y_test`, and `test_ids` must come from the same split (in
    the same row order) — true whether they're passed in fresh from
    `train.prepare_training_data` or reconstructed later by this file's
    `__main__` block.
    """
    predictions = model.predict(X_test)

    df_error = pd.DataFrame(
        {
            "id": test_ids["id"].to_numpy(),
            "price": y_test.to_numpy(),
            "latitude": X_test["lat"].to_numpy(),
            "longitude": X_test["long"].to_numpy(),
            "price_prediction": predictions.round(2),
        }
    )
    df_error["price_difference"] = (
        df_error["price_prediction"] - df_error["price"]
    ).round(2)
    df_error["price_difference_percent"] = (
        (df_error["price_difference"] / df_error["price"]) * 100
    ).round(2)
    return df_error


def largest_error(df_error: pd.DataFrame) -> pd.Series:
    """Returns the row with the biggest error, by absolute percentage
    difference — catches both large over- and under-predictions.
    """
    idx = df_error["price_difference_percent"].abs().idxmax()
    return df_error.loc[idx]


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.config import ID_COLUMN, MODEL_PATH, PREPROCESSING_PATH, RANDOM_STATE, TEST_SIZE
    from src.data.loader import load_raw_data
    from src.modeling.train import split_features_and_target
    from src.persistence import load_model
    from src.pipeline import clean_raw_data

    print("Loading dataset...")
    try:
        raw_df = load_raw_data('data/King_County_House_prices_dataset.csv')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Dataset not found. Please ensure the dataset is available at 'data/King_County_House_prices_dataset.csv'.")
        exit(1)
        
    cleaned_df = clean_raw_data(raw_df)

    # Reproduce the exact split used during training: same cleaned input
    # + same random_state -> the identical rows land in the test set, so
    # this evaluates on data the model has genuinely never seen, rather
    # than a fresh random split that could overlap with training rows.
    _, test_df_raw = train_test_split(
        cleaned_df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Loading fitted preprocessing pipeline and trained model...")
    fitted_preprocessing = load_model(PREPROCESSING_PATH)
    model = load_model(MODEL_PATH)

    # .transform(), not .fit_transform(): reuses the reference waterfront
    # list learned from *training* rows, doesn't relearn it from test data.
    test_df = fitted_preprocessing.transform(test_df_raw)
    X_test, y_test = split_features_and_target(test_df)
    test_ids = test_df[[ID_COLUMN]]

    metrics = evaluate_model(model, X_test, y_test)
    print(
        f"Test R^2: {metrics['r2']:.3f}  "
        f"(n={metrics['n_samples']}, features={metrics['n_features']})"
    )

    df_error = error_analysis(model, X_test, y_test, test_ids)
    print("\nLargest single prediction error:")
    print(largest_error(df_error))