# utils for data cleaning
import pandas as pd


def drop_bad_bedroom_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with invalid bedroom values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with invalid bedroom rows dropped.
    """
    
    df = df[df['bedrooms'] <= 5]    # threshold based on perc75% (4.0)  # Keep rows with non-negative bedroom values
    return df

def fix_sqft_basement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the 'sqft_basement' column by recalculating basement area as the diff. between living area and area above ground.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'sqft_basement' column fixed.
    """
    df['sqft_basement'] = df['sqft_living'] - df['sqft_above']
    return df

def derive_last_renovation_year(renovation_years: pd.Series, built_years: pd.Series) -> list[int]:
    """
    Derive the last renovation year from the 'yr_renovated' column.

    Args:
        renovation_years (pd.Series): The input Series containing renovation years.

    Returns:
        list[int]: A list of last renovation years.
    """
    
    years = []
    for idx, yr_re in renovation_years.items():
        if str(yr_re) == "nan" or yr_re == 0:
            years.append(built_years[idx])
        else:
            years.append(int(yr_re))
    return years



def fix_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing values in the DataFrame by filling them with appropriate value strategies.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values fixed.
    """
    
    missing_vals = df.isnull().sum().to_frame(name="count")
    missing_vals["percentage"] = (missing_vals["count"] / df.shape[0] * 100).round(
        2
    )
    missing_vals.query("count != 0")
    
    # fill missing 'view' values with 0
    df['view'] = df['view'].fillna(0)
    
    # fill missing 'waterfront' values with 0
    df['waterfront'] = df['waterfront'].fillna(0)
    
    # fill missing 'yr_renovated' values (future work: should be pulled into helper function)
    years_last_renovated = derive_last_renovation_year(df['yr_renovated'], df['yr_built'])
    df['last_known_change'] = years_last_renovated
    df.drop(columns=['yr_renovated', 'yr_built'], inplace=True)
    
    return df