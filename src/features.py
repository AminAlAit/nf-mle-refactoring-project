import numpy as np
import pandas as pd
from numpy.typing import NDArray

center_point_lat = 47.62774
center_point_long = -122.24194
center_point_ref_lat = 47.6219

### This is code for feature engineering where water_front_house and center_distance labelled data is added to to the king_county data

def calculate_dist(delta_long: NDArray, delta_lat: NDArray, ref_lat: NDArray) -> NDArray:

     delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
     
     return (np.sqrt(delta_long_corr**2 + delta_lat**2)* 2 * np.pi * 6378 / 360)


def feature_engi_water_house_dist(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy()

    water_list = df.query("waterfront == 1")

    house_long = df["long"].to_numpy()
    house_lat = df["lat"].to_numpy()

    water_long = water_list["long"].to_numpy()
    water_lat = water_list["lat"].to_numpy()

    delta_long = house_long[:, None] - water_long[None, :]
    delta_lat = house_lat[:, None] - water_lat[None, :]

    distance = calculate_dist(delta_long, delta_lat, water_lat[None, :])

    df["water_distance"] = distance.min(axis=1)
    
    return df

def feature_engi_center_dist(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy()
    
    df["sqft_price"] = (df.price / (df.sqft_living + df.sqft_lot)).round(3)
    
    df["delta_lat"] = np.absolute(center_point_lat - df["lat"])
    
    df["delta_long"] = np.absolute(center_point_long - df["long"])

    df["unity"] = 1.0

    ref_lat = df["unity"].to_numpy()*center_point_ref_lat

    delta_long = df["delta_long"].to_numpy() 
    delta_lat =  df["delta_lat"].to_numpy()

    distance = calculate_dist(delta_long, delta_lat, ref_lat)

    df["center_distance"] = distance

    df.drop(columns=["sqft_price","delta_lat","delta_long","unity"],inplace=True)

    return df
        
