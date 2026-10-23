"""Helper functions for testing."""

import os
import sys
import subprocess
from typing import Dict
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from copairs import nulls
from copairs.matching import ColumnList

SEED = 0
PROJECT_ROOT = Path(__file__).parents[1]


def run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter on this checkout's sources.

    Keeps a crash in compiled code, or a module loaded by the code, out of
    the test process.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


_WITHOUT_MODULES = """
import sys
import importlib.util
from importlib.abc import MetaPathFinder

BLOCKED = {names!r}


class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)


_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: (
    None if name.split(".")[0] in BLOCKED else _find_spec(name, *args)
)
sys.meta_path.insert(0, Block())
"""


def without_modules(*names: str) -> str:
    """Code that makes ``names`` unimportable in a child, as if not installed.

    Their imports raise ModuleNotFoundError and ``importlib.util.find_spec``
    returns None for them.
    """
    return _WITHOUT_MODULES.format(names=names)


def simulate_plates(n_compounds, n_replicates, plate_size):
    """Round robin creation of platemaps."""
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
    """Simulate random platemaps."""
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
    """Simulate random dataframe."""
    dframe = pd.DataFrame(columns=list(vocab_size.keys()), index=range(length))
    for col, size in vocab_size.items():
        dframe[col] = rng.integers(1, size + 1, size=length)
        dframe[col] = col + dframe[col].astype(str)

    sort_cols = list(diffby) + list(sameby)
    dframe = dframe.sort_values(sort_cols)
    dframe = dframe.drop_duplicates().reset_index(drop=True)
    return dframe


def create_dframe(n_options, n_rows):
    """Create a dataframe with predefined number of plates, wells, and compounds."""
    if isinstance(n_options, int):
        n_options = [n_options] * 3
    colc = list(f"c{i}" for i in range(n_options[0]))
    colp = list(f"p{i}" for i in range(n_options[1]))
    colw = list(f"w{i}" for i in range(n_options[2]))
    dframe = pd.DataFrame((product(colc, colp, colw)), columns=list("cpw"))
    dframe = dframe.sample(n_rows, random_state=SEED).reset_index(drop=True)
    return dframe


def brute_ap_pvalues(scores, conf_ix, confs, null_size, seed):
    """Count null values >= each score over the materialized nulls."""
    null = nulls.ap_nulls(confs, null_size, seed, dtype=np.float64)
    num = (null[conf_ix] >= nulls.tie_thresholds(scores)[:, None]).sum(axis=1)
    return (num + 1) / (null_size + 1)


def brute_group_nulls(ptr, conf_ix, conf_cnt, confs, null_size, seed):
    """Materialized mAP nulls: each group's mean of its members' AP nulls.

    Group ``g`` has ``conf_cnt[e]`` members with configuration
    ``confs[conf_ix[e]]`` for ``e`` in ``ptr[g]:ptr[g + 1]``; members sharing a
    configuration share null samples.
    """
    null = nulls.ap_nulls(confs, null_size, seed, dtype=np.float64)
    group_null = np.empty((len(ptr) - 1, null_size))
    for g in range(len(ptr) - 1):
        acc = np.zeros(null_size)
        for e in range(ptr[g], ptr[g + 1]):
            acc += conf_cnt[e] * null[conf_ix[e]]
        group_null[g] = acc / np.sum(conf_cnt[ptr[g] : ptr[g + 1]])
    return group_null


def brute_map_pvalues(map_scores, ptr, conf_ix, conf_cnt, confs, null_size, seed):
    """Count group null values >= each mAP over :func:`brute_group_nulls`."""
    group_null = brute_group_nulls(ptr, conf_ix, conf_cnt, confs, null_size, seed)
    num = (group_null >= nulls.tie_thresholds(map_scores)[:, None]).sum(axis=1)
    return (num + 1) / (null_size + 1)
