"""Assemble checked PR tables from complete primary and repeat runs."""

import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NULLS = ("permutation", "joint", "copied")


def report(workdir, inputs, output):
    """Check per-group decisions, summarize seed variation, and emit a table."""
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((workdir / "cases.json").read_text())
    records = []
    locations = {}
    for source in inputs:
        rows = json.loads((source / "summary.json").read_text())
        for row in rows:
            key = (row["key"], row["seed"], row["null"])
            assert key not in locations, key
            assert row["status"] == "ok", row
            locations[key] = source / row["key"]
            records.append(row)
    summary = pd.DataFrame(records)
    seeds = sorted(summary.seed.unique().tolist())
    assert seeds == [0, 19, 41]
    assert set(summary.draws) == {100_000}
    assert len(summary) == len(cases) * len(seeds) * len(NULLS)
    groups = []
    for case in cases:
        for seed in seeds:
            directory = locations[(case["key"], seed, "permutation")]
            counts = pd.read_csv(
                directory / "pair-counts.csv",
                index_col=0,
                dtype={"positive_counts": str, "total_counts": str},
            )
            joined = None
            for null in NULLS:
                directory = locations[(case["key"], seed, null)]
                frame = pd.read_csv(directory / f"{seed}-{null}.csv", index_col=0)
                assert not frame.index.duplicated().any()
                if joined is None:
                    joined = frame[["mean_average_precision"]].join(counts)
                np.testing.assert_array_equal(
                    joined.mean_average_precision, frame.mean_average_precision
                )
                for column in ("p_value", "corrected_p_value", "below_corrected_p"):
                    joined[f"{null}_{column}"] = frame[column]
                assert frame.below_corrected_p.equals(frame.corrected_p_value.le(0.05))
            primary = joined.permutation_below_corrected_p
            for null in NULLS:
                row = summary.loc[
                    (summary.key == case["key"])
                    & (summary.seed == seed)
                    & (summary.null == null)
                ].iloc[0]
                alternative = joined[f"{null}_below_corrected_p"]
                gained = int((alternative & ~primary).sum())
                lost = int((~alternative & primary).sum())
                assert int(alternative.sum()) == row.flagged
                assert gained + lost == row.changed_vs_permutation
                assert (
                    gained == row.gained_vs_permutation
                    and lost == row.lost_vs_permutation
                )
            joined.index.name = "group"
            groups.append(joined.reset_index().assign(case=case["key"], seed=seed))
    wide = pd.concat(groups, ignore_index=True)
    # Repeated seeds must change sampling only, never the observed scores.
    assert wide.groupby(["case", "group"]).mean_average_precision.nunique().eq(1).all()
    wide.to_csv(output / "group-comparison.csv", index=False)
    summary.to_csv(output / "case-summary.csv", index=False)
    stable = (
        wide.groupby(["case", "group"])[[f"{n}_below_corrected_p" for n in NULLS]]
        .nunique()
        .gt(1)
    )
    stability = stable.groupby("case").sum().astype(int)
    stability.columns = [f"{n}_groups_changing_call_across_seeds" for n in NULLS]
    stability.to_csv(output / "seed-stability.csv")
    table = [
        "| Dataset / comparison | Groups | Permutation | Joint | Copied | Changed vs permutation (J / C) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        current = summary.loc[summary.key.eq(case["key"])]
        primary = current.loc[current.seed.eq(0)].set_index("null")
        cells = []
        for null in NULLS:
            values = current.loc[current.null.eq(null), "flagged"]
            text = str(int(primary.loc[null, "flagged"]))
            if values.min() != values.max():
                text += f" [{int(values.min())}–{int(values.max())}]"
            cells.append(text)
        changed = " / ".join(
            str(int(primary.loc[n, "changed_vs_permutation"]))
            for n in ("joint", "copied")
        )
        table.append(
            f"| {case['title']} | {case['groups_expected']} | "
            + " | ".join(cells)
            + f" | {changed} |"
        )
    note = "Counts flagged at BH-FDR 0.05, with 100,000 draws per group. Main counts and changed-call counts use seed 0; brackets show the count range over seeds 0, 19 and 41 where it varies. Changed counts compare group identities, not just totals."
    (output / "table.md").write_text("\n".join(table) + "\n\n" + note + "\n")
    selected = wide.loc[
        wide.seed.eq(0)
        & (
            ((wide.case == "lincs_activity") & (wide.group == "BRD-K68132782-003-13-8"))
            | ((wide.case == "lincs_targets") & (wide.group == "PSMB1"))
        )
    ]
    selected.to_csv(output / "illustrative-groups.csv", index=False)
    # Independent perfect-retrieval calculation for the real PSMB1 pair.
    psmb1 = selected.loc[selected.group.eq("PSMB1")].iloc[0]
    counts = [int(x) for x in psmb1.total_counts.split(",")]
    assert psmb1.members == 2 and psmb1.positive_counts == "1"
    assert psmb1.mean_average_precision == 1 and len(counts) == 2
    exact = {"joint": 1 / (sum(counts) - 1), "copied": 1 / math.prod(counts)}
    for null, probability in exact.items():
        for seed in seeds:
            row = wide.loc[
                (wide.case == "lincs_targets")
                & (wide.group == "PSMB1")
                & (wide.seed == seed)
            ].iloc[0]
            sampled = row[f"{null}_p_value"]
            assert (
                abs(sampled - probability)
                < 6 * math.sqrt(probability * (1 - probability) / 100_000) + 2 / 100_001
            )
    oracle = {
        "case": "lincs_targets",
        "label": "PSMB1",
        "members": 2,
        "positive_counts": [1, 1],
        "total_counts": counts,
        "exact_p": exact,
        "seeds_checked": seeds,
    }
    (output / "perfect-pair-oracle.json").write_text(
        json.dumps(oracle, indent=2) + "\n"
    )
    print("\n".join(table))
    print("\n" + note)
    print("\nGroup/method calls varying across seeds:", int(stable.to_numpy().sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.workdir, args.inputs, args.output)
