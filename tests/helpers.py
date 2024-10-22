from itertools import product
from typing import Dict

import numpy as np
import pandas as pd

from copairs.matching import ColumnList

SEED = 0


def simulate_plates(n_compounds, n_replicates, plate_size):
    """Round robin creation of platemaps"""
    total = n_compounds * n_replicates

    compounds = []
    plates = []
    wells = []
    for i in range(total):
        compound_id = i % n_compounds
        well_id = i % plate_size
        plate_id = i // plate_size
        compounds.append(f"c{compound_id}")
        plates.append(f"p{plate_id}")
        wells.append(f"w{well_id}")

    dframe = pd.DataFrame({"c": compounds, "p": plates, "w": wells})
    return dframe


def simulate_random_plates(
    n_compounds: int,
    n_replicates: int,
    plate_size: int,
    sameby=ColumnList,
    diffby=ColumnList,
):
    rng = np.random.default_rng(SEED)
    dframe = simulate_plates(n_compounds, n_replicates, plate_size)
    # Shuffle values
    for col in dframe.columns:
        rng.shuffle(dframe[col].values)
    sort_cols = list(diffby) + list(sameby)
    dframe = dframe.sort_values(sort_cols).reset_index(drop=True)
    return dframe


def simulate_random_dframe(
    length: int,
    vocab_size: Dict[str, int],
    sameby: ColumnList,
    diffby: ColumnList,
    rng: np.random.Generator,
):
    dframe = pd.DataFrame(columns=list(vocab_size.keys()), index=range(length))
    for col, size in vocab_size.items():
        dframe[col] = rng.integers(1, size + 1, size=length)
        dframe[col] = col + dframe[col].astype(str)

    sort_cols = list(diffby) + list(sameby)
    dframe = dframe.sort_values(sort_cols)
    dframe = dframe.drop_duplicates().reset_index(drop=True)
    return dframe


def create_dframe(n_options, n_rows):
    """
    Random permutation of a fix number of elements per column
    """
    if isinstance(n_options, int):
        n_options = [n_options] * 3
    colc = list(f"c{i}" for i in range(n_options[0]))
    colp = list(f"p{i}" for i in range(n_options[1]))
    colw = list(f"w{i}" for i in range(n_options[2]))
    dframe = pd.DataFrame((product(colc, colp, colw)), columns=list("cpw"))
    dframe = dframe.sample(n_rows, random_state=SEED).reset_index(drop=True)
    return dframe
