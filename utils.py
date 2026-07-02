"""Data utility functions for the Bike Rental Demand Prediction project.

This module provides reusable, stateless helpers for loading, cleaning,
inspecting, and analysing the bike rental dataset. No Streamlit or ML
logic lives here — it is intentionally kept as pure data utilities so
it can be imported by both train_model.py and app.py without side-effects.
"""
from __future__ import annotations

import pandas as pd

NUMERIC_COLUMN: list[str] = ["temp", "humidity", "windspeed", "count"]


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If no file exists at ``file_path``.
        Exception: For any other I/O or parsing error.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to load the dataset from '{file_path}': {e}") from e


def get_dataset_info(df: pd.DataFrame) -> dict[str, object]:
    """Return a summary dictionary describing a DataFrame's structure.

    Args:
        df: The input DataFrame to inspect.

    Returns:
        A dict with keys:
            - ``shape``: (rows, columns) tuple.
            - ``columns``: list of column names.
            - ``dtypes``: mapping of column name → dtype.
            - ``missing_values``: mapping of column name → missing count.
            - ``duplicates``: number of duplicate rows.
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and fill missing numeric values with column medians.

    Missing values are filled rather than dropped because the dataset is small
    (100 rows) and losing any row would noticeably reduce training data.

    Args:
        df: The raw input DataFrame.

    Returns:
        A cleaned copy of the DataFrame with a reset index.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    for column in NUMERIC_COLUMN:
        if column in cleaned.columns and cleaned[column].isnull().any():
            median_value = cleaned[column].median()
            # Use assignment instead of inplace=True — deprecated in pandas 2.x
            cleaned[column] = cleaned[column].fillna(median_value)

    return cleaned.reset_index(drop=True)


def get_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute the pairwise Pearson correlation matrix for selected columns.

    Args:
        df: The input DataFrame.
        columns: List of column names to include in the correlation matrix.

    Returns:
        A square DataFrame of correlation coefficients.
    """
    return df[columns].corr()
