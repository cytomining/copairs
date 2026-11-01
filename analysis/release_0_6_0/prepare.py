"""Prepare ten fixed real-data comparisons; selection never uses null p-values."""

import json
import hashlib
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pycytominer import normalize, feature_select

HERE = Path(__file__).resolve().parent
OPS = [
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
]


def feature_columns(frame):
    """Return numerical feature names, excluding metadata."""
    return [c for c in frame if not c.startswith("Metadata")]


def normalize_plate(frame, plate_column, select):
    """Apply the paper's plate normalization and optional selection recipe."""
    features = feature_columns(frame)
    meta = [c for c in frame if c not in features]
    parts = [
        normalize(
            group,
            features=features,
            meta_features=meta,
            method="mad_robustize",
            samples="all",
        )
        for _, group in frame.groupby(plate_column, sort=True)
    ]
    result = pd.concat(parts, ignore_index=True)
    if select:
        result = feature_select(result, operation=OPS, na_cutoff=0)
    return result


def consensus(frame, compound, annotation):
    """Aggregate every annotated noncontrol compound without activity filtering."""
    valid = frame.loc[
        frame[annotation].notna() & frame[annotation].ne("") & ~frame.Metadata_control
    ]
    result = valid.groupby([compound, annotation], as_index=False)[
        feature_columns(valid)
    ].median()
    result = result.rename(
        columns={compound: "Metadata_compound", annotation: "Metadata_target"}
    )
    return result


def prepare(workdir):
    """Verify inputs and build fixed, explicitly described comparison datasets."""
    data = workdir / "data"
    prepared = workdir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for asset in json.loads((HERE / "data-sources.json").read_text()):
        path = data / asset["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(path.name + ".part")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--max-time",
                    "180",
                    asset["url"],
                    "-o",
                    str(temporary),
                ],
                check=True,
            )
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == asset["sha256"]
            temporary.replace(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"], path
    cases = []
    preprocessing = {}

    def save(key, title, frame, kind, *, blocked=False):
        frame = frame.reset_index(drop=True)
        features = feature_columns(frame)
        values = frame[features].to_numpy()
        assert np.isfinite(values).all(), key
        assert np.all(np.linalg.norm(values, axis=1) > 0), key
        frame.to_parquet(prepared / f"{key}.parquet", index=False)
        record = {
            "key": key,
            "title": title,
            "kind": kind,
            "blocked": blocked,
            "profiles": len(frame),
            "features": len(features),
        }
        if kind == "activity":
            record["controls"] = int(frame.Metadata_control.sum())
            sizes = frame.loc[~frame.Metadata_control].groupby("Metadata_group").size()
        else:
            sizes = (
                frame.Metadata_target.str.split("|", regex=False)
                .explode()
                .value_counts()
            )
            record["singleton_labels"] = int(sizes.eq(1).sum())
            record["profiles_with_multiple_labels"] = int(
                frame.Metadata_target.str.contains("|", regex=False).sum()
            )
            sizes = sizes.loc[sizes > 1]
        record["members_min"] = int(sizes.min())
        record["members_max"] = int(sizes.max())
        record["groups_expected"] = len(sizes)
        cases.append(record)
        print(record, flush=True)

    lincs = pd.read_csv(data / "lincs/SQ00014812_normalized_feature_select.csv.gz")
    lincs = lincs.loc[:, lincs.nunique() > 1].copy()
    lincs["Metadata_group"] = lincs.Metadata_broad_sample
    lincs["Metadata_control"] = lincs.Metadata_group.eq("DMSO")
    save("lincs_activity", "LINCS: compound activity", lincs, "activity")
    # Unlike the earlier six-well note's companion workflow, this comparison
    # uses every annotated compound, with no activity-based selection.
    lincs_consensus = consensus(lincs, "Metadata_broad_sample", "Metadata_target")
    save("lincs_targets", "LINCS: target consistency", lincs_consensus, "multilabel")
    preprocessing["lincs"] = {
        "recipe": "Supplied normalized/feature-selected plate; remove constant columns; all annotated compounds for target consistency",
        "annotated_compounds": len(lincs_consensus),
    }

    raw = []
    for path in sorted((data / "cellhealth").glob("SQ*_augmented.csv.gz")):
        frame = pd.read_csv(path).rename(
            columns={
                "Image_Metadata_Plate": "Metadata_Plate",
                "Image_Metadata_Well": "Metadata_Well",
            }
        )
        frame = frame.drop(columns="Metadata_broad_sample")
        raw.append(frame)
    cellhealth = normalize_plate(
        pd.concat(raw, ignore_index=True), "Metadata_Plate", True
    )
    cellhealth = cellhealth.drop(
        columns=[c for c in cellhealth if "costes" in c.lower()]
    )
    preprocessing["cellhealth"] = {
        "recipe": "Whole-plate MAD robustization; pooled feature selection using the paper's operations; remove Costes features",
        "raw_profiles": sum(len(x) for x in raw),
        "features": len(feature_columns(cellhealth)),
        "selection": "Exactly six wells per noncontrol guide within a cell line; retain all Chr2/Luc/LacZ cutting controls",
    }
    for cell in ("A549", "ES2", "HCC44"):
        frame = cellhealth.loc[cellhealth.Metadata_cell_line.eq(cell)].copy()
        sizes = frame.groupby("Metadata_pert_name").size()
        six = sizes.index[sizes.eq(6)]
        controls = frame.Metadata_gene_name.isin(["Chr2", "Luc", "LacZ"])
        frame = frame.loc[frame.Metadata_pert_name.isin(six) | controls].copy()
        frame["Metadata_group"] = frame.Metadata_pert_name
        frame["Metadata_control"] = frame.Metadata_gene_name.isin(
            ["Chr2", "Luc", "LacZ"]
        )
        save(
            f"cellhealth_{cell.lower()}",
            f"Cell Health {cell}: guide activity",
            frame,
            "activity",
        )
        if cell == "A549":
            save(
                "cellhealth_a549_blocked",
                "Cell Health A549: plate-matched controls",
                frame,
                "activity",
                blocked=True,
            )

    matched = {}
    for name, plate, select in (
        ("cellpainting", "Metadata_Plate", True),
        ("nelisa", "Metadata_nelisa_plate_id", False),
    ):
        frame = pd.read_parquet(data / f"nelisa/{name}_profiles.parquet")
        raw_shape = list(frame.shape)
        frame = normalize_plate(frame, plate, select)
        frame["Metadata_broad_sample"] = frame.Metadata_broad_sample.fillna("NA")
        frame["Metadata_group"] = frame.Metadata_broad_sample
        frame["Metadata_control"] = frame.Metadata_control_type.eq("negcon")
        assert frame.Metadata_control.equals(frame.Metadata_group.eq("NA"))
        matched[name] = frame
        preprocessing[name] = {
            "raw_shape": raw_shape,
            "recipe": "Whole-plate MAD robustization; paper feature selection operations for Cell Painting, no feature selection for nELISA",
            "features": len(feature_columns(frame)),
        }
    sizes = pd.concat(
        {
            name: frame.groupby("Metadata_broad_sample").size()
            for name, frame in matched.items()
        },
        axis=1,
    )
    keep = sizes.index[sizes.notna().all(axis=1) & sizes.nunique(axis=1).eq(1)]
    # This matches the paper's ID/count harmonization, not an asserted well-level join.
    for name, title in (
        ("cellpainting", "Matched Cell Painting"),
        ("nelisa", "nELISA secretome"),
    ):
        frame = matched[name]
        preprocessing[name]["excluded_compound_ids"] = sorted(
            set(frame.Metadata_broad_sample) - set(keep)
        )
        frame = frame.loc[frame.Metadata_broad_sample.isin(keep)].copy()
        save(f"{name}_activity", f"{title}: compound activity", frame, "activity")
        aggregated = consensus(frame, "Metadata_broad_sample", "Metadata_target_list")
        save(
            f"{name}_targets", f"{title}: target consistency", aggregated, "multilabel"
        )
    (workdir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (workdir / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    prepare(parser.parse_args().workdir)
