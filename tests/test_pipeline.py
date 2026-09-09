"""
Tests for transformers, pipeline and modeling.
"""

import numpy as np
import pandas as pd
import pytest

from kc.cleaning import clean
from kc.config import EXCLUDED_FROM_MODEL
from kc.features import add_sqft_price
from kc.modeling import (
    adjusted_r2,
    error_table,
    evaluate,
    load_model,
    save_model,
    split_dataset,
)
from kc.pipeline import (
    build_baseline_pipeline,
    build_feature_pipeline,
    build_model_pipeline,
)
from kc.transformers import ColumnPruner, WaterDistance


@pytest.fixture
def prepared(sample_data: pd.DataFrame) -> pd.DataFrame:
    return add_sqft_price(clean(sample_data))


@pytest.fixture
def split(prepared: pd.DataFrame):
    return split_dataset(prepared)


# transformers


def test_column_pruner_rejects_a_missing_column(prepared: pd.DataFrame) -> None:
    """
    A clear error beats a silently wrong prediction.
    """
    pruner = ColumnPruner(drop=("price",)).fit(prepared)
    with pytest.raises(ValueError, match="Missing columns"):
        pruner.transform(prepared.drop(columns=["grade"]))


def test_column_pruner_fixes_column_order(prepared: pd.DataFrame) -> None:
    pruner = ColumnPruner(drop=("price",)).fit(prepared)
    shuffled = prepared[prepared.columns[::-1]]
    assert list(pruner.transform(shuffled).columns) == pruner.columns_


def test_column_pruner_rejects_both_arguments() -> None:
    with pytest.raises(ValueError, match="not both"):
        ColumnPruner(drop=("a",), keep=("b",)).fit(pd.DataFrame({"a": [1], "b": [2]}))


def test_water_distance_reference_comes_from_fit_only(prepared: pd.DataFrame) -> None:
    """
    The reference is frozen at fit time, not recomputed per transform.

    This is the leak in the notebook: it builds the waterfront list from the whole
    dataset before splitting, so test rows shape a training feature.
    """
    transformer = WaterDistance().fit(prepared.iloc[:1000])
    fitted_reference = transformer.reference_.copy()

    transformer.transform(prepared.iloc[1000:])
    assert np.array_equal(transformer.reference_, fitted_reference)


# pipeline


def test_feature_pipeline_excludes_every_leaky_column(split) -> None:
    X_train, _, y_train, _ = split
    features = build_feature_pipeline().fit(X_train, y_train)
    columns = features.named_steps["select"].columns_
    for excluded in EXCLUDED_FROM_MODEL:
        assert excluded not in columns
    assert "center_distance" in columns
    assert "water_distance" in columns


def test_model_pipeline_predicts(split) -> None:
    X_train, X_test, y_train, _ = split
    predictions = build_model_pipeline().fit(X_train, y_train).predict(X_test)
    assert len(predictions) == len(X_test)
    assert np.isfinite(predictions).all()


def test_model_pipeline_accepts_raw_columns(split, sample_data: pd.DataFrame) -> None:
    """
    A frame straight from the CSV works, no manual preprocessing. The API needs
    this to be true.
    """
    X_train, _, y_train, _ = split
    model = build_model_pipeline().fit(X_train, y_train)
    assert len(model.predict(sample_data.drop(columns=["price"]).iloc[[0]])) == 1


def test_baseline_reproduces_the_notebook(raw_data: pd.DataFrame) -> None:
    """
    Cells 90 and 97 report about 0.43 and 0.48.
    """
    prepared = add_sqft_price(clean(raw_data))
    X_train, X_test, y_train, y_test = split_dataset(prepared)

    grade_only = build_baseline_pipeline(["grade"]).fit(X_train, y_train)
    assert evaluate(grade_only, X_test, y_test)["adjusted_r2"] == pytest.approx(
        0.43, abs=0.02
    )

    with_age = build_baseline_pipeline(["grade", "last_known_change"]).fit(
        X_train, y_train
    )
    assert evaluate(with_age, X_test, y_test)["adjusted_r2"] == pytest.approx(
        0.48, abs=0.02
    )


# modeling helpers


def test_adjusted_r2_penalises_extra_features() -> None:
    assert adjusted_r2(0.8, 1000, 50) < 0.8
    assert adjusted_r2(0.8, 1000, 1) == pytest.approx(0.8, abs=1e-3)


def test_adjusted_r2_is_nan_when_undefined() -> None:
    assert np.isnan(adjusted_r2(0.8, 10, 20))


def test_split_is_reproducible(prepared: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        split_dataset(prepared)[0], split_dataset(prepared)[0]
    )


def test_split_rejects_a_missing_target(prepared: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        split_dataset(prepared.drop(columns=["price"]))


def test_error_table_percentages(split) -> None:
    X_train, X_test, y_train, y_test = split
    model = build_model_pipeline().fit(X_train, y_train)
    errors = error_table(model, X_test, y_test)

    assert len(errors) == len(X_test)
    recomputed = (errors["price_difference"] / errors["price"] * 100).round(2)
    pd.testing.assert_series_equal(
        errors["price_difference_percent"], recomputed, check_names=False
    )


def test_model_survives_a_save_and_load_round_trip(split, tmp_path) -> None:
    X_train, X_test, y_train, _ = split
    model = build_model_pipeline().fit(X_train, y_train)
    path = save_model(model, tmp_path / "model.bin")

    assert np.allclose(model.predict(X_test), load_model(path).predict(X_test))


def test_load_model_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="kc.train"):
        load_model(tmp_path / "absent.bin")
