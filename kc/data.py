"""
Loading the CSV. Notebook cell 7.
"""

from pathlib import Path

import pandas as pd

from kc.config import RAW_DATA_PATH

# 454 rows have "?" in sqft_basement, which is why pandas reads it as text. Read it
# as text on purpose and rebuild the numbers in cleaning.
RAW_DTYPES = {"sqft_basement": "string"}


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Read the CSV, no cleaning.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}")
    return pd.read_csv(path, dtype=RAW_DTYPES)
