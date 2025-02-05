"""Functions to support query-like syntax when finding the matches."""

import itertools
import re
from typing import List, Tuple

import numpy as np
import pandas as pd


def validate_pipeline_input(
    meta: pd.DataFrame, feats: np.ndarray, columns: List[str]
) -> None:
    """Validate the metadata and features for consistency and completeness.

    Parameters
    ----------
    meta : pd.DataFrame
        The metadata DataFrame describing the profiles.
    feats : np.ndarray
        The feature matrix where rows correspond to profiles in the metadata.
    columns : List[str]
        List of column names in the metadata to validate for null values.

    Raises
    ------
    ValueError:
        - If any of the specified metadata columns contain null values.
        - If the number of rows in the metadata and features are not equal.
        - If the feature matrix contains null values.
    """
    # Check for null values in the specified metadata columns
    if meta[columns].isna().any(axis=None):
        raise ValueError("metadata columns should not have null values.")

    # Check if the number of rows in metadata matches the feature matrix
    if len(meta) != len(feats):
        raise ValueError("Metadata and features must have the same number of rows.")

    # Check for null values in the feature matrix
    if np.isnan(feats).any():
        raise ValueError("features should not have null values.")


def flatten_str_list(*args):
    """Create a single list with all the params given."""
    columns = set()
    for col in args:
        if isinstance(col, str):
            columns.add(col)
        elif isinstance(col, dict):
            columns.update(itertools.chain.from_iterable(col.values()))
        else:
            columns.update(col)
    columns = list(columns)
    return columns


def evaluate_and_filter(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Evaluate query filters and filter the metadata DataFrame based on specified columns.

    This function processes column specifications, extracts any filter conditions,
    applies these conditions to the metadata DataFrame, and returns the filtered metadata
    along with the updated list of columns.

    Parameters
    ----------
    df : pd.DataFrame
        The metadata DataFrame containing information about profiles to be filtered.
    columns : List[str]
        A list of metadata column names.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - The filtered metadata DataFrame.
        - The updated list of columns after processing any filter specifications.
    """
    # Extract query filters from the column specifications
    query_list, columns = extract_filters(columns, df.columns)

    # Apply the extracted filters to the metadata DataFrame
    df = apply_filters(df, query_list)

    # Return the filtered metadata DataFrame and the updated list of columns
    return df, columns


def extract_filters(
    columns: List[str], df_columns: List[str]
) -> Tuple[List[str], List[str]]:
    """Extract and validate query filters from selected metadata columns.

    Parameters
    ----------
    columns : List[str]
        A list of selected metadata column names or query expressions. Query expressions
        should follow a valid syntax (e.g., "metadata_column > 5" or "metadata_column == 'value'").
    df_columns : List[str]
        All available metadata column names to validate against.

    Returns
    -------
    Tuple[List[str], List[str]]
        - `queries_to_eval`: A list of valid query expressions to evaluate.
        - `parsed_cols`: A list of valid metadata column names extracted from the input `columns`.

    Raises
    ------
    ValueError:
        - If a metadata column or query expression is invalid (e.g., references a non-existent column).
        - If duplicate queries are found for the same metadata column.
    """
    # Initialize lists to store parsed metadata column names and query expressions
    parsed_cols = []
    queries_to_eval = []

    # Iterate through each entry in the selected metadata columns
    for col in columns:
        if col in df_columns:
            # If the entry is a valid metadata column name, add it to parsed_cols
            parsed_cols.append(col)
            continue

        # Use regex to extract metadata column names from query expressions
        column_names = re.findall(r"(\w+)\s*[=<>!]+", col)

        # Validate the extracted metadata column names against all available metadata columns
        valid_column_names = [col for col in column_names if col in df_columns]
        if not valid_column_names:
            raise ValueError(f"Invalid query or metadata column name: {col}")

        # Add valid query expressions and associated metadata columns
        queries_to_eval.append(col)
        parsed_cols.extend(valid_column_names)

        # Check for duplicate metadata columns in the parsed list
        if len(parsed_cols) != len(set(parsed_cols)):
            raise ValueError(f"Duplicate queries for column: {col}")

    # Return the queries to evaluate and the parsed metadata column names
    return queries_to_eval, parsed_cols


def apply_filters(df: pd.DataFrame, query_list: List[str]) -> pd.DataFrame:
    """Combine and apply query filters to a DataFrame.

    This function takes a list of query expressions and applies them to a DataFrame
    to filter its rows. If no query expressions are provided, the original DataFrame
    is returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to which the filters will be applied.
    query_list : List[str]
        A list of query expressions (e.g., "column_name > 5"). These expressions
        should follow the syntax supported by `pd.DataFrame.query`.

    Returns
    -------
    pd.DataFrame
        The DataFrame filtered based on the provided query expressions.

    Raises
    ------
    ValueError:
        - If the combined query results in an empty DataFrame.
        - If the combined query expression is invalid.
    """
    # If no queries are provided, return the original DataFrame unchanged
    if not query_list:
        return df

    # Combine the query expressions into a single string using logical AND (&)
    combined_query = " & ".join(f"({query})" for query in query_list)

    try:
        # Apply the combined query to filter the DataFrame
        df_filtered = df.query(combined_query)

        # Raise an error if the filtered DataFrame is empty
        if df_filtered.empty:
            raise ValueError(f"No data matched the query: {combined_query}")
    except Exception as e:
        # Handle any issues with the query expression and provide feedback
        raise ValueError(
            f"Invalid combined query expression: {combined_query}. Error: {e}"
        )

    # Return the filtered DataFrame
    return df_filtered
