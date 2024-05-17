import itertools
import re
from typing import List, Tuple

import pandas as pd
import numpy as np


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
    """Evaluate the query and filter the dataframe"""
    parsed_cols = []
    for col in columns:
        if col in df.columns:
            parsed_cols.append(col)
            continue

        column_names = re.findall(r"(\w+)\s*[=<>!]+", col)
        valid_column_names = [col for col in column_names if col in df.columns]
        if not valid_column_names:
            raise ValueError(f"Invalid query or column name: {col}")

        try:
            df = df.query(col)
            parsed_cols.extend(valid_column_names)
        except:
            raise ValueError(f"Invalid query expression: {col}")

    return df, parsed_cols
