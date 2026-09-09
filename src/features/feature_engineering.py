# utils for feature engineering

import pandas as pd 
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class WaterfrontDistanceAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer that calculates the distance from each house to the nearest waterfront house.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "WaterfrontDistanceAdder":
        waterfront_houses = X.loc[X["waterfront"] == 1, ["long", "lat"]]
        self.reference_long_ = waterfront_houses["long"].to_numpy()
        self.reference_lat_ = waterfront_houses["lat"].to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.reference_long_.size == 0:
            X["water_distance"] = np.nan
            return X

        # calculate geographic distance between two points using Haversine formula
        delta_long = np.abs(X["long"].to_numpy()[:, None] - self.reference_long_[None, :])
        delta_lat = np.abs(X["lat"].to_numpy()[:, None] - self.reference_lat_[None, :])
        delta_long_corr = delta_long * np.cos(np.radians(self.reference_lat_[None, :]))
        distances = (
            (delta_long_corr**2 + delta_lat**2) ** 0.5 * 2 * np.pi * 6378 / 360
        )
        X["water_distance"] = distances.min(axis=1)
        
        return X

def add_sqft_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new feature column 'sqft_price' to the DataFrame, which is calculated as:
    
    price_per_square_foot = total_price / (living_area + lot_area)
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    
    df['sqft_price'] = (df['price'] / (df['sqft_living'] + df['sqft_lot'])).round(2)
    return df

def add_distance_to_center_of_wealth(df: pd.DataFrame, target_lat: float, target_long: float) -> pd.DataFrame:
    """
    Add a new feature column that describes the geographic distance to a reference point (center of wealth).
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_lat (float): The latitude of the reference point.
        target_long (float): The longitude of the reference point.
        
    Returns:
        pd.DataFrame: The DataFrame with the new feature column added.
    """
    
    # determine deltas in latitude and longitude
    df['delta_lat'] = np.absolute(target_lat - df['lat'])
    df['delta_long'] = np.absolute(target_long - df['long'])
    
    # calculate distance to center of wealth using the Haversine formula
    df["center_distance"] = (
        (
            (df["delta_long"] * np.cos(np.radians(47.6219))) ** 2
            + df["delta_lat"] ** 2
        )
        ** (1 / 2)
        * 2
        * np.pi
        * 6378
        / 360
    )

    return df
