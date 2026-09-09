# main training pipeline components

from __future__ import annotations 

from dataclasses import dataclass 

import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import (
    ID_COLUMN, 
    LEAKAGE_COLUMNS,
    PARAM_GRID, 
    RANDOM_STATE, 
    TEST_SIZE, 
    TARGET_COLUMN,
)

@dataclass
class TrainingData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_ids: pd.DataFrame # 'id' column of the test set
  
  
def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    leakage_columns: list[str] = LEAKAGE_COLUMNS,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features and target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        leakage_columns (list[str]): List of columns to drop due to data leakage.
        id_column (str): The name of the ID column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target Series.
    """
    
    cols_to_drop = set(leakage_columns | {target_column, id_column})
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    return df[feature_cols], df[target_column]


def prepare_training_data(
    df: pd.DataFrame,
    preprocessing_pipeline: Pipeline,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[TrainingData, Pipeline]:
    """
    Splits the cleaned (but not yet feature-engineered) DataFrame into
    train/test, *then* fits `preprocessing_pipeline` on the training rows
    only and applies it to each half.

    Splitting before feature engineering (rather than splitting an
    already-featured DataFrame) is what keeps `WaterfrontDistanceAdder`
    from learning about test-set waterfront houses. Returns the
    fitted `preprocessing_pipeline` alongside the data, since that exact
    fitted state (its learned reference list) is what needs to be reused
    later at prediction time.
    """
    train_df_raw, test_df_raw = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_df = preprocessing_pipeline.fit_transform(train_df_raw)
    test_df = preprocessing_pipeline.transform(test_df_raw)

    X_train, y_train = split_features_and_target(train_df)
    X_test, y_test = split_features_and_target(test_df)
    test_ids = test_df[[ID_COLUMN]]

    data = TrainingData(X_train, X_test, y_train, y_test, test_ids)
    return data, preprocessing_pipeline


def build_model_pipeline() -> Pipeline:
    """
    Build a machine learning pipeline for regression.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    
    pipeline = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        (
            "model", 
            TransformedTargetRegressor(
                regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
                transformer=StandardScaler(),
            ),
        )
    ])
    
    return pipeline


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = PARAM_GRID,
    cv: int = 5,
) -> GridSearchCV:
    """
    Run a grid search to find the best hyperparameters for the model.
    """
    
    grid_search = GridSearchCV(
        build_model_pipeline(),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        error_score="raise",
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train(
    df: pd.DataFrame,
    preprocessing_pipeline: Pipeline | None = None,
) -> tuple[Pipeline, TrainingData, GridSearchCV, Pipeline]:
    """
    End-to-end training entry point. Takes the cleaned DataFrame, not yet feature engineered and returns:
    - best fitted model pipeline
    - the split used
    - the full 'GricSearchCV' object
    - the fitted preprocessing pipeline
    """
    if preprocessing_pipeline is None:
        from src.pipeline import build_preprocessing_pipeline
        
        preprocessing_pipeline = build_preprocessing_pipeline()

    data, fitted_preprocessing = prepare_training_data(df, preprocessing_pipeline)
    grid_search = run_grid_search(data.X_train, data.y_train)
    best_model = grid_search.best_estimator_  # GridSearchCV already refits on all of X_train
    return best_model, data, grid_search, fitted_preprocessing



if __name__ == "__main__":
    from src.config import MODEL_PATH, PREPROCESSING_PATH
    from src.data.loader import load_raw_data
    from src.persistence import save_model
    from src.pipeline import clean_raw_data

    raw_df = load_raw_data()
    cleaned_df = clean_raw_data(raw_df)
    model, training_data, grid_search, fitted_preprocessing = train(cleaned_df)

    print("Best params:", grid_search.best_params_)
    save_model(model, MODEL_PATH)
    save_model(fitted_preprocessing, PREPROCESSING_PATH)