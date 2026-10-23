"""Tests for the choice of mAP null in copairs' mAP functions."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from copairs import compute
from copairs.map import (
    get_map_pvalue,
    average_precision,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from tests.helpers import ap_frame, query_meta
from copairs.methods import NULLS


def test_copied_null_averages_configuration_nulls(tmp_path):
    """null="copied" averages each member's configuration null, as copairs 0.5.5."""
    rows = [("a", 0.4, 2, 30)] * 3 + [("b", 0.3, 4, 40), ("b", 0.5, 5, 41)]
    frame = ap_frame(rows)
    got = get_map_pvalue(
        frame, ["g"], 500, 0, progress_bar=False, cache_dir=tmp_path, null="copied"
    )
    confs, rev_ix = np.unique(
        frame[["n_pos_pairs", "n_total_pairs"]].values, axis=0, return_inverse=True
    )
    null_dists = compute.get_null_dists(
        confs, 500, seed=0, cache_dir=tmp_path / "ref", progress_bar=False
    )
    for g, p in zip(got["g"], got["p_value"]):
        rows = np.flatnonzero(frame["g"] == g)
        null = null_dists[rev_ix.ravel()[rows]].mean(axis=0, dtype=np.float64)
        score = frame["average_precision"].iloc[rows].mean()
        assert p == ((null >= score - 1e-6 * score).sum() + 1) / 501


def perfect_retrieval_scores():
    """Return two groups of two replicates (one positive each) among 100 profiles."""
    return ap_frame([("a", 1.0, 1, 100)] * 2 + [("b", 0.5, 1, 100)] * 2)


def test_map_pvalue_counts_ties(tmp_path):
    """An mAP equal to a null atom counts the atom.

    Under the copied null P(mAP >= 1) = P(AP >= 1) = 1/100. Under the joint
    null, given the shared similarity u each member's rank - 1 is
    Binomial(99, 1 - u): both APs are 1 with probability int u^198 = 1/199, and
    mAP >= 0.5 needs one rank of 1 or both of 2.
    """
    kwargs = dict(null_size=100_000, threshold=0.05, seed=0, progress_bar=False)
    both_two = 99**2 * 2 / (197 * 198 * 199)  # int (99 (1 - u) u^98)^2 du
    for null, expected in (
        ("copied", np.array([0.01, 0.02])),
        ("joint", np.array([1 / 199, 2 / 100 - 1 / 199 + both_two])),
    ):
        got = mean_average_precision(
            perfect_retrieval_scores(), ["g"], null=null, cache_dir=tmp_path, **kwargs
        )
        se = np.sqrt(expected * (1 - expected) / kwargs["null_size"])
        assert (np.abs(got["p_value"].to_numpy() - expected) < 5 * se).all()


@pytest.mark.parametrize("null", ["copied", "joint"])
def test_float32_scores_count_ties(null, tmp_path):
    """float32-rounded scores still tie with the null atoms they round from.

    With one positive among 3, AP is 1, 1/2 or 1/3, so every null value is
    ``>= 1/3`` and the p-value of an mAP of 1/3 is exactly 1. pandas 2 averages
    float32 APs to float32 and pandas 3 to float64; ties hold under both.
    """
    scores = pd.DataFrame(
        {
            "g": ["a", "a"],
            "average_precision": np.float32([1 / 3, 1 / 3]),
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 3,
        }
    )
    got = mean_average_precision(
        scores, ["g"], 1000, 0.05, 0, progress_bar=False, null=null, cache_dir=tmp_path
    )
    assert got["p_value"].iloc[0] == 1.0


def test_nulls_share_the_pair_count_check():
    """Positive pairs that miss the AP table's counts raise alike under both nulls."""
    meta = query_meta()
    feats = np.random.default_rng(0).standard_normal((len(meta), 10))
    ap, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    short = dataclasses.replace(pairs, pos=pairs.pos[1:])
    kwargs = dict(null_size=10, threshold=0.05, seed=0, progress_bar=False)
    for null in ("joint", "permutation"):
        with pytest.raises(ValueError, match="2 positive pairs but 3 in the AP table"):
            mean_average_precision(ap, ["g"], null=null, pairs=short, **kwargs)


def test_unknown_null():
    """Unknown nulls, and pos_pairs outside the joint null, raise."""
    frame = ap_frame([("a", 1.0, 1, 100)] * 2)
    kwargs = dict(null_size=10, threshold=0.05, seed=0, progress_bar=False)
    with pytest.raises(ValueError, match="unknown null"):
        mean_average_precision(frame, ["g"], null="exact", **kwargs)
    with pytest.raises(ValueError, match="joint null only"):
        mean_average_precision(
            frame, ["g"], null="copied", pos_pairs=np.array([[0, 1]]), **kwargs
        )


def test_hierarchical_takes_each_null():
    """The hierarchical entry point validates null= and passes it to the p-values."""
    rows = [(g, d) for g in "abc" for d in (1, 2) for _ in range(3)] + [("ctrl", 0)] * 8
    meta = pd.DataFrame(rows, columns=["g", "dose"])
    meta["ref"] = np.where(meta["g"] == "ctrl", meta.index, -1)
    feats = np.random.default_rng(5).standard_normal((len(meta), 6))
    feats[meta["g"].eq("a").to_numpy()] += 2.0
    keys = ["g", "dose", "ref"]
    ap = average_precision(meta, feats, keys, [], [], keys, progress_bar=False)
    ap = ap[ap["g"] != "ctrl"]
    kwargs = dict(null_size=300, threshold=0.05, seed=0, progress_bar=False)
    sameby = ["g", "dose"]
    p = {}
    for null in NULLS:
        flat = mean_average_precision(ap, sameby, null=null, **kwargs)
        hier = mean_average_precision_hierarchical(
            ap, sameby, hierarchical_by=["g"], null=null, **kwargs
        )
        assert hier["p_value"].tolist() == flat["p_value"].tolist()
        p[null] = flat["p_value"].tolist()
    assert p["permutation"] != p["joint"] != p["copied"]
    default = mean_average_precision_hierarchical(
        ap, sameby, hierarchical_by=["g"], **kwargs
    )
    assert default["p_value"].tolist() == p["permutation"]
    with pytest.raises(ValueError, match="unknown null"):
        mean_average_precision_hierarchical(
            ap, sameby, hierarchical_by=["g"], null="exact", **kwargs
        )
