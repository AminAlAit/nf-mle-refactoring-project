"""
Feature engineering. Notebook cells 56-70.
"""

import numpy as np
import pandas as pd

from kc.config import (
    EARTH_RADIUS_KM,
    LON_CORRECTION_LAT,
    WEALTH_CENTER_LAT,
    WEALTH_CENTER_LONG,
)

# One degree of latitude in km. A degree of longitude is this times cos(lat),
# because the meridians squeeze together as you go north.
KM_PER_DEGREE = 2 * np.pi * EARTH_RADIUS_KM / 360


def dist(long, lat, ref_long, ref_lat):
    """
    Distance in km to a reference point. Cell 67.

    Works on scalars or arrays. Column vectors for the houses and row vectors for
    the references give you the whole distance matrix by broadcasting.
    """
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corrected = delta_long * np.cos(np.radians(ref_lat))
    return np.sqrt(delta_long_corrected**2 + delta_lat**2) * KM_PER_DEGREE


def add_sqft_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price per square foot. Cell 56.

    Derived from price, so it can't go into the model. It would leak the answer.
    Skipped when there's no price column, i.e. a prediction request.
    """
    if "price" not in df.columns:
        return df.copy()
    out = df.copy()
    out["sqft_price"] = (out["price"] / (out["sqft_living"] + out["sqft_lot"])).round(2)
    return out


def add_center_distance(
    df: pd.DataFrame,
    center_lat: float = WEALTH_CENTER_LAT,
    center_long: float = WEALTH_CENTER_LONG,
) -> pd.DataFrame:
    """
    Distance in km to the "centre of wealth". Cell 60.
    """
    out = df.copy()
    out["delta_lat"] = np.absolute(center_lat - out["lat"])
    out["delta_long"] = np.absolute(center_long - out["long"])
    # The notebook uses a fixed cos(LON_CORRECTION_LAT) here instead of calling
    # dist(); see the note in config about the two latitudes.
    out["center_distance"] = (
        np.sqrt(
            (out["delta_long"] * np.cos(np.radians(LON_CORRECTION_LAT))) ** 2
            + out["delta_lat"] ** 2
        )
        * KM_PER_DEGREE
    )
    return out


def waterfront_reference(df: pd.DataFrame) -> np.ndarray:
    """
    Coordinates of the waterfront houses, as an (n, 2) array. Cell 68.
    """
    return df.loc[df["waterfront"] == 1, ["long", "lat"]].to_numpy(dtype=float)


def water_distance(long, lat, reference: np.ndarray) -> np.ndarray:
    """
    Distance in km to the closest waterfront house. Cell 69.

    The notebook loops over 21,596 houses x 146 waterfront houses and takes minutes.
    Same arithmetic, one broadcast, milliseconds. test_features checks they match.
    """
    if reference.size == 0:
        raise ValueError("No waterfront houses to measure against.")

    house_long = np.asarray(long, dtype=float).reshape(-1, 1)
    house_lat = np.asarray(lat, dtype=float).reshape(-1, 1)
    ref_long = reference[:, 0].reshape(1, -1)
    ref_lat = reference[:, 1].reshape(1, -1)

    return dist(house_long, house_lat, ref_long, ref_lat).min(axis=1)


def add_water_distance(
    df: pd.DataFrame, reference: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Add water_distance. Cells 69-70.

    Without a reference it takes the waterfront houses from df, like the notebook.
    Inside a pipeline the reference comes from the training split only.
    """
    out = df.copy()
    if reference is None:
        reference = waterfront_reference(out)
    out["water_distance"] = water_distance(out["long"], out["lat"], reference)
    return out


def add_features(
    df: pd.DataFrame, water_reference: np.ndarray | None = None
) -> pd.DataFrame:
    """
    All three derived columns.
    """
    out = add_sqft_price(df)
    out = add_center_distance(out)
    out = add_water_distance(out, reference=water_reference)
    return out


def water_distance_naive(df: pd.DataFrame, reference: np.ndarray) -> list[float]:
    """
    The notebook's original loop. Only used by the test that proves the fast
    version gives the same answer. Don't call this for real.
    """
    distances = []
    for idx in df.index:
        per_reference = []
        for ref_long, ref_lat in reference:
            per_reference.append(
                dist(df["long"][idx], df["lat"][idx], ref_long, ref_lat).min()
            )
        distances.append(min(per_reference))
    return distances
