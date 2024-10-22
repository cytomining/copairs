import itertools
import re
from typing import List, Tuple

import numpy as np
import pandas as pd


def validate_pipeline_input(meta, feats, columns):
    if meta[columns].isna().any(axis=None):
        raise ValueError("metadata columns should not have null values.")
    if len(meta) != len(feats):
        raise ValueError("meta and feats have different number of rows")
    if np.isnan(feats).any():
        raise ValueError("features should not have null values.")


def flatten_str_list(*args):
    """create a single list with all the params given"""
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


def evaluate_and_filter(df, columns) -> Tuple[pd.DataFrame, List[str]]:
    """Evaluate queries and filter the dataframe"""
    query_list, columns = extract_filters(columns, df.columns)
    df = apply_filters(df, query_list)
    return df, columns


def extract_filters(columns, df_columns) -> Tuple[List[str], List[str]]:
    """Extract and validate filters from columns"""
    parsed_cols = []
    queries_to_eval = []

    for col in columns:
        if col in df_columns:
            parsed_cols.append(col)
            continue
        column_names = re.findall(r"(\w+)\s*[=<>!]+", col)

        valid_column_names = [col for col in column_names if col in df_columns]
        if not valid_column_names:
            raise ValueError(f"Invalid query or column name: {col}")

        queries_to_eval.append(col)
        parsed_cols.extend(valid_column_names)

        if len(parsed_cols) != len(set(parsed_cols)):
            raise ValueError(f"Duplicate queries for column: {col}")

    return queries_to_eval, parsed_cols


def apply_filters(df, query_list):
    """Combine and apply filters to dataframe"""
    if not query_list:
        return df

    combined_query = " & ".join(f"({query})" for query in query_list)
    try:
        df_filtered = df.query(combined_query)
        if df_filtered.empty:
            raise ValueError(f"No data matched the query: {combined_query}")
    except Exception as e:
        raise ValueError(
            f"Invalid combined query expression: {combined_query}. Error: {e}"
        )

    return df_filtered
