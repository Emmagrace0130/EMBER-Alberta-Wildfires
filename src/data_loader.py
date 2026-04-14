"""
data_loader.py
--------------
Load the raw Alberta wildfire Excel data into a pandas DataFrame.
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the wildfire Excel file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Excel file (e.g. fp-historical-wildfire-data-2006-2024.xlsx).

    Returns
    -------
    pd.DataFrame
        Raw data as loaded from disk.
    """
    df = pd.read_excel(filepath)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df
