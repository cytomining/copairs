"""Compare all three nulls on each fixed AP table; no fallback or model tuning."""

import json
import hashlib
import argparse
import platform
import tempfile
from time import perf_counter
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd
from prepare import feature_columns

import copairs
from copairs import map as cmap

HERE = Path(__file__).resolve().parent
NULLS = ("permutation", "joint", "copied")
RESOURCES = dict(
    max_workers=4,
    max_reference_profiles=512,
    pool_memory_budget=256 * 1024**2,
    key_table_memory_budget=32 * 1024**2,
)


def check_candidate(output):
    """Verify installed library bytes and record the numerical environment."""
    expected = json.loads((HERE / "candidate-library.json").read_text())
    assert metadata.version("copairs") == expected["version"]
    site = Path(copairs.__file__).resolve().parent.parent
    for path, digest in expected["files"].items():
        assert hashlib.sha256((site / path).read_bytes()).hexdigest() == digest, path
    versions = {
        name: metadata.version(name)
        for name in (
            "copairs",
            "numpy",
            "pandas",
            "scipy",
            "duckdb",
            "pyarrow",
            "pycytominer",
            "scikit-learn",
            "statsmodels",
        )
    }
    record = {
        "candidate": expected["commit"],
        "matching_library_files": len(expected["files"]),
        "python": platform.python_version(),
        "packages": versions,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resources": RESOURCES,
    }
    (output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")


def build_ap(frame, case):
    """Compute AP and preserve complete groups and their attached pair context."""
    features = feature_columns(frame)
    meta = frame.drop(columns=features).copy()
    if case["kind"] == "multilabel":
        meta["Metadata_target"] = meta.Metadata_target.str.split("|", regex=False)
        ap = cmap.multilabel.average_precision(
            meta,
            frame[features].to_numpy(),
            pos_sameby=["Metadata_target"],
            pos_diffby=["Metadata_compound"],
            neg_sameby=[],
            neg_diffby=["Metadata_target", "Metadata_compound"],
            multilabel_col="Metadata_target",
            progress_bar=False,
        )
        return ap, ["Metadata_target"]
    meta["Metadata_reference_index"] = np.where(
        meta.Metadata_control, np.arange(len(meta)), -1
    )
    ap = cmap.average_precision(
        meta,
        frame[features].to_numpy(),
        pos_sameby=["Metadata_group", "Metadata_reference_index"],
        pos_diffby=[],
        neg_sameby=["Metadata_Plate"] if case["blocked"] else [],
        neg_diffby=["Metadata_group", "Metadata_reference_index", "Metadata_control"],
        progress_bar=False,
    )
    return ap.loc[~ap.Metadata_control], ["Metadata_group"]


def run(workdir, output, draws, seeds, only):
    """Run fixed-draw comparisons and record successes or explicit rejections."""
    output.mkdir(parents=True, exist_ok=True)
    check_candidate(output)
    cases = json.loads((workdir / "cases.json").read_text())
    records = []
    for case in cases:
        if only and case["key"] not in only:
            continue
        directory = output / case["key"]
        directory.mkdir(exist_ok=True)
        frame = pd.read_parquet(workdir / "prepared" / f"{case['key']}.parquet")
        start = perf_counter()
        ap, by = build_ap(frame, case)
        ap_seconds = perf_counter() - start
        ap.to_csv(directory / "ap.csv", index_label="ap_row")
        counts = ap.groupby(by[0]).agg(
            members=("n_pos_pairs", "size"),
            positive_counts=(
                "n_pos_pairs",
                lambda x: ",".join(map(str, sorted(set(x)))),
            ),
            total_counts=(
                "n_total_pairs",
                lambda x: ",".join(map(str, sorted(set(x)))),
            ),
        )
        counts.to_csv(directory / "pair-counts.csv")
        baseline = None
        for seed in seeds:
            results = {}
            for null in NULLS:
                print(
                    f"{case['key']}: {null}, seed={seed}, draws={draws:,}", flush=True
                )
                start = perf_counter()
                with tempfile.TemporaryDirectory(prefix="copairs-real-null-") as cache:
                    try:
                        result = cmap.mean_average_precision(
                            ap,
                            by,
                            null_size=draws,
                            threshold=0.05,
                            seed=seed,
                            null=null,
                            progress_bar=False,
                            cache_dir=cache,
                            **RESOURCES,
                        )
                    except ValueError as error:
                        record = case | {
                            "null": null,
                            "seed": seed,
                            "draws": draws,
                            "status": "rejected",
                            "error": str(error),
                        }
                        records.append(record)
                        print(record, flush=True)
                        (directory / f"{seed}-{null}-error.json").write_text(
                            json.dumps(record, indent=2) + "\n"
                        )
                        continue
                seconds = perf_counter() - start
                assert len(result) == case["groups_expected"]
                assert result.p_value.between(1 / (draws + 1), 1).all()
                assert result.corrected_p_value.between(0, 1).all()
                assert not result.below_corrected_p.isna().any()
                scores = result[
                    by + ["mean_average_precision", "mean_normalized_average_precision"]
                ]
                if baseline is None:
                    baseline = scores.copy()
                pd.testing.assert_frame_equal(scores, baseline)
                result.to_csv(directory / f"{seed}-{null}.csv", index=False)
                results[null] = result.set_index(by[0])
                record = case | {
                    "null": null,
                    "seed": seed,
                    "draws": draws,
                    "status": "ok",
                    "tested": len(result),
                    "flagged": int(result.below_corrected_p.sum()),
                    "raw_p_below_05": int(result.below_p.sum()),
                    "seconds": seconds,
                    "ap_seconds": ap_seconds,
                    "mAP_identical": True,
                }
                records.append(record)
                print(
                    f"  {record['flagged']}/{record['tested']} flagged; {seconds:.2f}s",
                    flush=True,
                )
            if "permutation" in results:
                for record in records:
                    if (
                        record["key"] == case["key"]
                        and record["seed"] == seed
                        and record["status"] == "ok"
                    ):
                        alternative = results[record["null"]].below_corrected_p
                        primary = results["permutation"].below_corrected_p
                        record["changed_vs_permutation"] = int(
                            alternative.ne(primary).sum()
                        )
                        record["gained_vs_permutation"] = int(
                            (alternative & ~primary).sum()
                        )
                        record["lost_vs_permutation"] = int(
                            (~alternative & primary).sum()
                        )
        # Checkpoint summaries after every complete case.
        pd.DataFrame(records).to_csv(output / "summary.csv", index=False)
        (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 19, 41])
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    run(args.workdir, args.output, args.draws, args.seeds, args.only)
