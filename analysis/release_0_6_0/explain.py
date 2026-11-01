"""Check six actual LINCS wells against independently calculated exact nulls."""

import json
import math
import argparse
import tempfile
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from compare import RESOURCES, build_ap
from prepare import feature_columns
from scipy.spatial.distance import cdist

from copairs import map as cmap


def explain(workdir, output):
    """Enumerate all label assignments in a prespecified six-well subset."""
    frame = pd.read_parquet(workdir / "prepared/lincs_activity.parquet")
    compound = sorted(set(frame.Metadata_group) - {"DMSO"})[0]
    treatment = (
        frame.loc[frame.Metadata_group.eq(compound)]
        .sort_values("Metadata_Well")
        .head(2)
    )
    control = frame.loc[frame.Metadata_control].sort_values("Metadata_Well").head(4)
    small = pd.concat([treatment, control], ignore_index=True)
    wells = small.Metadata_Well.tolist()
    similarities = (
        1
        - cdist(
            small[feature_columns(small)],
            small[feature_columns(small)],
            metric="cosine",
        )
    ).astype(np.float32)
    ranks = np.zeros((6, 6), dtype=int)
    for row in range(6):
        others = np.array([j for j in range(6) if j != row])
        assert len(np.unique(similarities[row, others])) == 5
        order = others[np.argsort(-similarities[row, others])]
        ranks[row, order] = np.arange(1, 6)
    assignments = []
    for i, j in itertools.combinations(range(6), 2):
        score = float(
            np.float32(
                np.mean(
                    [np.float32(1 / ranks[i, j]), np.float32(1 / ranks[j, i])],
                    dtype=np.float64,
                )
            )
        )
        assignments.append(
            {
                "well_1": wells[i],
                "well_2": wells[j],
                "rank_1": int(ranks[i, j]),
                "rank_2": int(ranks[j, i]),
                "mAP": score,
            }
        )
    observed = assignments[0]["mAP"]
    assert observed == 1
    hits = sum(row["mAP"] >= observed for row in assignments)
    exact = {"permutation": hits / 15, "joint": 1 / 9, "copied": 1 / 5}
    ap, by = build_ap(small, {"kind": "activity", "blocked": False})
    sampled = {}
    draws = 100_000
    for null, probability in exact.items():
        with tempfile.TemporaryDirectory(prefix="copairs-six-well-") as cache:
            result = cmap.mean_average_precision(
                ap,
                by,
                null_size=draws,
                threshold=0.05,
                null=null,
                seed=0,
                progress_bar=False,
                cache_dir=cache,
                **RESOURCES,
            )
        assert float(result.mean_average_precision.iloc[0]) == observed
        sampled[null] = float(result.p_value.iloc[0])
        assert abs(sampled[null] - probability) < 6 * math.sqrt(
            probability * (1 - probability) / draws
        ) + 2 / (draws + 1)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(assignments).to_csv(output / "six-well-assignments.csv", index=False)
    summary = {
        "compound": compound,
        "wells": wells,
        "selection": "First compound alphabetically; first two treatment wells and first four DMSO wells by well ID",
        "features": len(feature_columns(small)),
        "observed_mAP": observed,
        "exact_p": exact,
        "sampled_p": sampled,
        "draws": draws,
        "seed": 0,
        "perfect_pairs": [
            [row["well_1"], row["well_2"]] for row in assignments if row["mAP"] == 1
        ],
    }
    (output / "six-well-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    explain(args.workdir, args.output)
