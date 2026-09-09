import numpy as np
import pandas as pd 
from pathlib import Path
from features import feature_engi_center_dist, feature_engi_water_house_dist



MAX_PLAUSIBLE_BEDROOMS = 33

## This code cleanup the data of King_county_house through refactering 

def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "King_County_House_prices_dataset.csv"
    return pd.read_csv(data_path)



def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame: 
     df = dataframe.copy()
     df.fillna({"view":0, "waterfront":0},inplace=True)
     df = df[df["bedrooms"] < MAX_PLAUSIBLE_BEDROOMS].copy()
     df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]

     renovated = df["yr_renovated"]
     last_known_change = renovated.where(renovated.notna()
            & (renovated != 0),df["yr_built"]).astype(int)
     df["last_known_change"] = last_known_change
     df.drop(columns=["yr_renovated","yr_built"],inplace=True)
  
     return df  

## Here we check the whether data is cleaned up properly or not? For this, I print featured data_

data = load_data()

preprocessed_data = preprocess(data)

featured_data_for_center = feature_engi_center_dist(preprocessed_data)

featured_data_for_water_font = feature_engi_water_house_dist(featured_data_for_center)

#print(featured_data_for_water_font.info())



    


