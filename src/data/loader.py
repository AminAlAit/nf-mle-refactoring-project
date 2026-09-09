# utils for loading dataset

import pandas as pd


# load_raw_data(path) -> pd.DataFrame
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.

    Args:
        path (str): The path to the CSV file.   
    """
    return pd.read_csv(path)