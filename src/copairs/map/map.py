import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm.contrib.concurrent import thread_map

from copairs import compute

logger = logging.getLogger("copairs")


def mean_average_precision(
    ap_scores: pd.DataFrame, sameby, null_size: int, threshold: float, seed: int
) -> pd.DataFrame:
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    ap_scores = ap_scores.reset_index(drop=True).copy()

    logger.info("Computing null_dist...")
    null_confs = ap_scores[["n_pos_pairs", "n_total_pairs"]].values
    null_confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = compute.get_null_dists(null_confs, null_size, seed=seed)
    ap_scores["null_ix"] = rev_ix

    def get_p_value(params):
        map_score, indices = params
        null_dist = null_dists[rev_ix[indices]].mean(axis=0)
        num = (null_dist > map_score).sum()
        p_value = (num + 1) / (null_size + 1)
        return p_value

    logger.info("Computing p-values...")

    map_scores = ap_scores.groupby(sameby, observed=True).agg(
        {
            "average_precision": ["mean", lambda x: list(x.index)],
        }
    )
    map_scores.columns = ["mean_average_precision", "indices"]

    params = map_scores[["mean_average_precision", "indices"]]
    map_scores["p_value"] = thread_map(get_p_value, params.values, leave=False)
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        map_scores["p_value"], method="fdr_bh"
    )
    map_scores["corrected_p_value"] = pvals_corrected
    map_scores["below_p"] = map_scores["p_value"] < threshold
    map_scores["below_corrected_p"] = map_scores["corrected_p_value"] < threshold
    map_scores.drop(columns=["indices"], inplace=True)
    map_scores.reset_index(inplace=True)
    return map_scores
