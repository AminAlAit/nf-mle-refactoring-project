"""
Tests for cleaning.
"""

import pandas as pd

from kc.cleaning import (
    add_last_known_change,
    clean,
    clean_rowwise,
    drop_invalid_rows,
    fill_zero_categories,
    missing_value_report,
    rebuild_sqft_basement,
)


def test_drops_the_impossible_house(messy_frame: pd.DataFrame) -> None:
    result = drop_invalid_rows(messy_frame)
    assert 33 not in result["bedrooms"].to_numpy()
    assert len(result) == len(messy_frame) - 1


def test_keeps_expensive_houses(messy_frame: pd.DataFrame) -> None:
    """
    Rare isn't the same as wrong, so the 1.5M house stays.
    """
    assert clean(messy_frame)["price"].max() == 1_500_000.0


def test_does_not_mutate_its_input(messy_frame: pd.DataFrame) -> None:
    """
    This is what the notebook's inplace=True gets wrong.
    """
    before = messy_frame.copy(deep=True)
    clean(messy_frame)
    pd.testing.assert_frame_equal(messy_frame, before)


def test_basement_is_numeric_and_consistent(messy_frame: pd.DataFrame) -> None:
    result = rebuild_sqft_basement(messy_frame)
    assert result["sqft_basement"].dtype.kind == "i"
    expected = messy_frame["sqft_living"] - messy_frame["sqft_above"]
    assert result["sqft_basement"].equals(expected.astype("int64"))
    assert result["sqft_basement"].notna().all()


def test_zero_fill_only_touches_missing_values(messy_frame: pd.DataFrame) -> None:
    result = fill_zero_categories(messy_frame)
    assert result["view"].isna().sum() == 0
    assert result["waterfront"].isna().sum() == 0
    assert result.loc[1, "view"] == 2  # real value survives
    assert result.loc[4, "waterfront"] == 1
    assert result.loc[2, "view"] == 0  # missing became 0
    assert result.loc[3, "waterfront"] == 0


def test_last_known_change_prefers_renovation_year(messy_frame: pd.DataFrame) -> None:
    result = add_last_known_change(messy_frame)
    assert result.loc[1, "last_known_change"] == 2003  # renovated
    assert result.loc[0, "last_known_change"] == 1955  # yr_renovated == 0
    assert result.loc[2, "last_known_change"] == 1930  # yr_renovated missing
    assert "yr_built" not in result.columns
    assert "yr_renovated" not in result.columns


def test_clean_is_idempotent(messy_frame: pd.DataFrame) -> None:
    once = clean(messy_frame)
    pd.testing.assert_frame_equal(once, clean(once))


def test_clean_leaves_no_missing_values(messy_frame: pd.DataFrame) -> None:
    assert clean(messy_frame).isna().sum().sum() == 0


def test_rowwise_cleaning_preserves_row_count(messy_frame: pd.DataFrame) -> None:
    """
    Why it's safe inside a sklearn pipeline.
    """
    assert len(clean_rowwise(messy_frame)) == len(messy_frame)


def test_clean_on_the_real_dataset(raw_data: pd.DataFrame) -> None:
    result = clean(raw_data)
    assert len(result) == len(raw_data) - 1
    assert result.isna().sum().sum() == 0
    assert result["sqft_basement"].dtype.kind == "i"
    assert result["last_known_change"].between(1900, 2015).all()
    assert missing_value_report(result).empty


def test_drop_invalid_rows_tolerates_a_missing_column() -> None:
    """
    A prediction request has no bedrooms guarantee, so don't explode.
    """
    assert len(drop_invalid_rows(pd.DataFrame({"price": [1.0]}))) == 1
