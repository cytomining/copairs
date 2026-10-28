"""Check global reference reapplication against the public metadata/AP pipeline."""

import numpy as np
import pandas as pd
import pytest
from study import Scenario, naive_map, assignments, make_layout

from copairs.map import average_precision


@pytest.mark.parametrize("rule", ["none", "same", "different"])
def test_global_matches_relabelled_public_average_precision(rule):
    """Rebuild real group/control columns, not the study's reference-list helper."""
    data = make_layout(Scenario("public_oracle", (1, 1), (1, 1), rule), 19)
    features = np.arange(len(data["plate"]), dtype=np.int64)[:, None]

    def similarity(left, right):
        return np.float32(1) - data["keys"][left[:, 0], right[:, 0]]

    for chosen in assignments(data):
        meta = pd.DataFrame(
            {
                "group": [f"control_{i}" for i in range(len(features))],
                "is_control": True,
                "plate": data["plate"],
            }
        )
        meta.loc[chosen, "group"] = "query"
        meta.loc[chosen, "is_control"] = False
        ap = average_precision(
            meta,
            features,
            pos_sameby=["group"],
            pos_diffby=[],
            neg_sameby=["plate"] if rule == "same" else [],
            neg_diffby=["is_control"] + (["plate"] if rule == "different" else []),
            distance=similarity,
            progress_bar=False,
        )
        query = ap.loc[ap["group"] == "query", "average_precision"]
        assert len(query) == len(chosen)
        assert query.mean() == pytest.approx(
            naive_map(data, chosen, "B_global")[0], abs=1e-7
        )
