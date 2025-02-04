"""Tests for assign reference index helper function."""

import numpy as np
import pandas as pd

from copairs.map import average_precision
from copairs.matching import assign_reference_index
from tests.helpers import simulate_random_dframe


def test_assign_reference_index():
    SEED = 42
    length = 20
    vocab_size = {"p": 5, "w": 3, "l": 2}
    n_feats = 5
    pos_sameby = ["l"]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = ["l"]
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))

    ap = average_precision(meta, feats, pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    ap_ri = average_precision(
        assign_reference_index(meta, "l=='l1'"),
        feats,
        pos_sameby + ["Metadata_Reference_Index"],
        pos_diffby,
        neg_sameby,
        neg_diffby + ["Metadata_Reference_Index"],
    )

    # Check no AP values were computed for the reference samples.
    assert ap_ri.query("l=='l1'").average_precision.isna().all()

    # Check AP values for all other samples are equal
    pd.testing.assert_frame_equal(
        ap_ri.query("l!='l1'").drop(columns="Metadata_Reference_Index"),
        ap.query("l!='l1'"),
    )
