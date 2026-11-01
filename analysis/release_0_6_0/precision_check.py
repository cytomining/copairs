"""Fixed-draw follow-up for observed seed sensitivity, not a new model."""

import json
import argparse
import tempfile
from pathlib import Path

import pandas as pd
from compare import RESOURCES, build_ap, check_candidate

from copairs import map as cmap


def precision_check(workdir, output):
    """Repeat the threshold-sensitive ES2 copied comparison at one million draws."""
    output.mkdir(parents=True, exist_ok=True)
    check_candidate(output)
    case = next(
        case
        for case in json.loads((workdir / "cases.json").read_text())
        if case["key"] == "cellhealth_es2"
    )
    frame = pd.read_parquet(workdir / "prepared/cellhealth_es2.parquet")
    ap, by = build_ap(frame, case)
    summaries = []
    for seed in (0, 19, 41):
        with tempfile.TemporaryDirectory(prefix="copairs-precision-") as cache:
            result = cmap.mean_average_precision(
                ap,
                by,
                null_size=1_000_000,
                seed=seed,
                threshold=0.05,
                null="copied",
                cache_dir=cache,
                progress_bar=False,
                **RESOURCES,
            )
        result.to_csv(output / f"{seed}-copied.csv", index=False)
        summaries.append(
            {
                "case": "cellhealth_es2",
                "null": "copied",
                "draws": 1_000_000,
                "seed": seed,
                "tested": len(result),
                "flagged": int(result.below_corrected_p.sum()),
            }
        )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precision_check(args.workdir, args.output)
