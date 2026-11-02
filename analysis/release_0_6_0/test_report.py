"""Report validation must compare both scores and group identities across seeds."""

import json

import pandas as pd
import pytest
from report import NULLS, report


@pytest.fixture
def inputs(tmp_path):
    """Build complete tiny runs with the perfect-pair oracle used by the report."""
    case = {"key": "lincs_targets", "title": "Targets", "groups_expected": 2}
    (tmp_path / "cases.json").write_text(json.dumps([case]))
    source = tmp_path / "runs"
    directory = source / case["key"]
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "members": [2, 2],
            "positive_counts": ["1", "1"],
            "total_counts": ["49,50", "49,50"],
        },
        index=["PSMB1", "other"],
    ).to_csv(directory / "pair-counts.csv")
    records = []
    for seed in (0, 19, 41):
        for null in NULLS:
            p = {"permutation": 0.1, "joint": 1 / 98, "copied": 1 / (49 * 50)}[null]
            flagged = int(null != "permutation")
            pd.DataFrame(
                {
                    "mean_average_precision": [1.0, 0.5],
                    "mean_normalized_average_precision": [1.0, 0.4],
                    "p_value": [p, 0.9],
                    "corrected_p_value": [2 * p, 0.9],
                    "below_corrected_p": [bool(flagged), False],
                },
                index=["PSMB1", "other"],
            ).to_csv(directory / f"{seed}-{null}.csv")
            records.append(
                case
                | {
                    "seed": seed,
                    "null": null,
                    "draws": 100_000,
                    "status": "ok",
                    "flagged": flagged,
                    "changed_vs_permutation": flagged,
                    "gained_vs_permutation": flagged,
                    "lost_vs_permutation": 0,
                }
            )
    (source / "summary.json").write_text(json.dumps(records))
    return tmp_path, source


def test_report_accepts_matching_scores_and_membership(inputs):
    """Accept all three methods and seeds when both scores and identities match."""
    work, source = inputs
    output = work / "report"
    report(work, [source], output)
    assert len(pd.read_csv(output / "group-comparison.csv")) == 6


@pytest.mark.parametrize("changed", ["normalized_score", "membership"])
def test_report_rejects_changed_seed_inputs(inputs, changed):
    """Reject drift even when raw mAP and all discovery counts remain unchanged."""
    work, source = inputs
    for null in NULLS:
        path = source / "lincs_targets" / f"19-{null}.csv"
        frame = pd.read_csv(path, index_col=0)
        if changed == "normalized_score":
            frame.loc["other", "mean_normalized_average_precision"] += 0.01
        else:
            frame = frame.rename(index={"other": "another"})
        frame.to_csv(path)
    with pytest.raises(AssertionError):
        report(work, [source], work / "report")
