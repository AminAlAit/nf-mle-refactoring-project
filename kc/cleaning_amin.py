"""
Data cleaning. Notebook cells 26-43.

Nothing here modifies its input, and clean(clean(df)) == clean(df).
The notebook uses inplace=True, which means you can't re-run its cells.
"""

import numpy as np
import pandas as pd

from kc.config_amin import MAX_PLAUSIBLE_BEDROOMS, ZERO_FILLED_COLUMNS


def drop_invalid_rows(
    df: pd.DataFrame, max_bedrooms: int = MAX_PLAUSIBLE_BEDROOMS
) -> pd.DataFrame:
    """
    Drop the 33-bedroom house. Cells 26-28.
    """
    if "bedrooms" not in df.columns:
        return df.copy()
    return df.loc[df["bedrooms"] <= max_bedrooms].copy()


def rebuild_sqft_basement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the basement as sqft_living - sqft_above. Cells 29-30.
    """
    out = df.copy()
    out["sqft_basement"] = (out["sqft_living"] - out["sqft_above"]).astype("int64")
    return out


def fill_zero_categories(
    df: pd.DataFrame, columns: tuple[str, ...] = ZERO_FILLED_COLUMNS
) -> pd.DataFrame:
    """
    Fill missing view and waterfront with 0. Cells 34-38.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].fillna(0).astype("int64")
    return out


def add_last_known_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge yr_built and yr_renovated into one column. Cells 40-43.

    The notebook does this with a row-by-row loop; this is the same thing vectorised.
    """
    out = df.copy()

    # Already done, so running clean() twice shouldn't blow up.
    if "last_known_change" in out.columns and "yr_built" not in out.columns:
        return out

    renovated = out["yr_renovated"].fillna(0)
    out["last_known_change"] = np.where(
        renovated > 0, renovated, out["yr_built"]
    ).astype("int64")
    return out.drop(columns=["yr_renovated", "yr_built"])


def clean_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    The cleaning steps that keep the row count.

    Split out because dropping rows inside a sklearn pipeline would misalign X and y.
    """
    out = rebuild_sqft_basement(df)
    out = fill_zero_categories(out)
    out = add_last_known_change(out)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning: drop bad rows, then fix the rest.
    """
    return clean_rowwise(drop_invalid_rows(df))


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing values per column, with percentages. Cells 32 and 39.
    """
    report = df.isnull().sum().to_frame(name="count")
    report["percentage"] = (report["count"] / len(df) * 100).round(2)
    return report.loc[report["count"] != 0]
