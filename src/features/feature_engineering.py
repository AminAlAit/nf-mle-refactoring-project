# utils for feature engineering

import pandas as pd 
import numpy as np

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

def calc_haversine_distance(long, lat, ref_long, ref_lat):
    """
    Calculate the geographic distance between two points using the Haversine formula.
    
    Args:
        long (float): Longitude of the first point.
        lat (float): Latitude of the first point.
        ref_long (float): Longitude of the reference point.
        ref_lat (float): Latitude of the reference point.
        
    Returns:
        float: The geographic distance between the two points in kilometers.
    """
    
    delta_long = np.absolute(long - ref_long)
    delta_lat = np.absolute(lat - ref_lat)
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    
    distance = (
        ((delta_long_corr) ** 2 + delta_lat ** 2) ** (1 / 2)
        * 2
        * np.pi
        * 6378
        / 360
    )
    
    return distance

def add_distance_to_waterfront_proxy(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Add a new feature column that describes the geographic distance to the nearest waterfront house.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with the new feature column added.
    """
    
    df_waterfront_houses = df.query('waterfront == 1')
    
    distances_to_waterfront_houses = []
    
    # TODO: vectorize!
    for idx in df.index:
        ref_list = []
        for x, y in zip(list(df_waterfront_houses['long']), list(df_waterfront_houses['lat'])):
            ref_list.append(calc_haversine_distance(df['long'][idx], df['lat'][idx], x, y).min())
        distances_to_waterfront_houses.append(min(ref_list))
    
    df['water_distance'] = distances_to_waterfront_houses
    return df