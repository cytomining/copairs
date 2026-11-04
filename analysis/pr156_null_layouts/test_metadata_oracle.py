"""Check global reference reapplication against the public metadata/AP pipeline."""

import numpy as np
import pandas as pd
import pytest
from study import (
    Scenario,
    naive_map,
    assignments,
    make_layout,
    matched_plate_guard_diagnostic,
)

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


def test_matched_plate_rules_document_current_production_guard_asymmetry():
    """Identical observed graphs are accepted for negative-only rules, rejected with pos_diffby."""
    diagnostic = matched_plate_guard_diagnostic()
    negative_only, both = diagnostic["cases"]
    assert negative_only["status"] == "accepted_by_production"
    assert both["status"] == "rejected_by_production_guard"
    assert "pos_diffby values do not fix" in both["reason"]
    assert negative_only["positive_pairs"] == both["positive_pairs"] == [[0, 1]]
    assert (
        negative_only["negative_pairs"]
        == both["negative_pairs"]
        == [
            [0, 4],
            [0, 5],
            [1, 2],
            [1, 3],
        ]
    )
    assert negative_only["observed_map"] == both["observed_map"]
    assert not diagnostic["guard_bypassed"]
    metadata = diagnostic["pool_assignment_metadata"]
    assert len(metadata) == 9
    assert sum(row["fixed_positive_edges"] == 0 for row in metadata) == 4
    assert sum(row["forbidden_pool_negative_edges"] > 0 for row in metadata) == 8
    assert "not computed" in diagnostic["positive_reapplication_maps"]
