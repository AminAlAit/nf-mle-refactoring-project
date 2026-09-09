"""
Tests for features_amin.

The one that matters is test_fast_water_distance_matches_the_notebook_loop.
Swapping a nested loop for a broadcast is the biggest change to the notebook's
logic, so "same answer" should be checked, not claimed in a comment.
"""

import numpy as np
import pandas as pd
import pytest

from kc.cleaning_amin import clean
from kc.config_amin import WEALTH_CENTER_LAT, WEALTH_CENTER_LONG
from kc.features_amin import (
    add_center_distance,
    add_features,
    add_sqft_price,
    add_water_distance,
    dist,
    water_distance,
    water_distance_naive,
    waterfront_reference,
)


def test_distance_to_itself_is_zero() -> None:
    assert dist(-122.3, 47.6, -122.3, 47.6) == pytest.approx(0.0)


def test_one_degree_of_latitude_is_about_111_km() -> None:
    """
    Units check: km, not degrees.
    """
    assert dist(-122.3, 48.6, -122.3, 47.6) == pytest.approx(111.3, abs=0.5)


def test_longitude_is_compressed_up_north() -> None:
    east_west = dist(-121.3, 47.6, -122.3, 47.6)
    north_south = dist(-122.3, 48.6, -122.3, 47.6)
    assert east_west < north_south


def test_center_distance_is_zero_at_the_center() -> None:
    frame = pd.DataFrame({"lat": [WEALTH_CENTER_LAT], "long": [WEALTH_CENTER_LONG]})
    assert add_center_distance(frame).loc[0, "center_distance"] == pytest.approx(
        0.0, abs=1e-9
    )


def test_center_distance_matches_the_notebook_average(raw_data: pd.DataFrame) -> None:
    """
    Cell 72 says houses sit about 17 km from the wealth centre.
    """
    result = add_center_distance(clean(raw_data))
    assert result["center_distance"].mean() == pytest.approx(17.5, abs=0.5)


def test_sqft_price_uses_living_area_plus_lot(messy_frame: pd.DataFrame) -> None:
    result = add_sqft_price(messy_frame)
    assert result.loc[0, "sqft_price"] == pytest.approx(round(300000.0 / 6500, 2))


def test_sqft_price_is_skipped_without_a_price_column() -> None:
    """
    A prediction request has no price, so this mustn't raise.
    """
    frame = pd.DataFrame({"sqft_living": [1500], "sqft_lot": [5000]})
    assert "sqft_price" not in add_sqft_price(frame).columns


def test_waterfront_house_has_zero_water_distance(messy_frame: pd.DataFrame) -> None:
    result = add_water_distance(clean(messy_frame))
    on_the_water = result.loc[result["waterfront"] == 1, "water_distance"]
    assert on_the_water.iloc[0] == pytest.approx(0.0)


def test_water_distance_needs_a_reference() -> None:
    with pytest.raises(ValueError, match="No waterfront"):
        water_distance([-122.3], [47.6], np.empty((0, 2)))


def test_fast_water_distance_matches_the_notebook_loop(raw_data: pd.DataFrame) -> None:
    """
    200 real rows against all 146 waterfront houses. Identical, not just close.

    Only 200 rows because the naive version is far too slow for the whole dataset,
    which is the reason it got replaced.
    """
    cleaned = clean(raw_data)
    reference = waterfront_reference(cleaned)
    sample = cleaned.sample(200, random_state=42)

    fast = water_distance(sample["long"], sample["lat"], reference)
    slow = np.array(water_distance_naive(sample, reference))

    assert np.array_equal(fast, slow)


def test_supplied_reference_beats_the_frames_own(raw_data: pd.DataFrame) -> None:
    """
    A single house has no waterfront neighbours, so the reference has to come in.
    This is what makes serving one prediction possible at all.
    """
    cleaned = clean(raw_data)
    reference = waterfront_reference(cleaned)
    result = add_water_distance(cleaned.iloc[[0]], reference=reference)

    assert result["water_distance"].notna().all()
    assert result["water_distance"].iloc[0] > 0


def test_add_features_adds_all_three(raw_data: pd.DataFrame) -> None:
    result = add_features(clean(raw_data))
    for column in ("sqft_price", "center_distance", "water_distance", "delta_lat"):
        assert column in result.columns
    assert result[["center_distance", "water_distance"]].notna().all().all()
