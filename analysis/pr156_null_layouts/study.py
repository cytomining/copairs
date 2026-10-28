"""Compare pool, local-filter and global-reapplication permutation statistics."""

import sys
import json
import math
import argparse
import platform
import itertools
from pathlib import Path
from dataclasses import asdict, dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd
from source import PIN, load_permutation
from scipy.stats import wasserstein_distance

production = load_permutation()
DRAW_SEED = 156
MODES = ("A", "B_local", "B_global")


@dataclass(frozen=True)
class Scenario:
    """Declare fixed plates, observed roles, and feature-generation parameters."""

    name: str
    members_per_plate: tuple[int, ...]
    controls_per_plate: tuple[int, ...]
    fixed_rule: str
    plate_effect: float = 0.0
    cluster_sd: float = 0.0
    dimensions: int = 8


SCENARIOS = [
    Scenario("shared_exchangeable", (3,), (5,), "none"),
    Scenario("same_plate_zero", (2, 1), (3, 4), "same"),
    Scenario("same_plate_strong", (2, 1), (3, 4), "same", 4.0),
    *[
        Scenario(f"cross_balanced_{name}", (2, 2), (3, 3), "different", effect)
        for name, effect in [("zero", 0.0), ("weak", 0.75), ("strong", 4.0)]
    ],
    *[
        Scenario(f"cross_imbalanced_{name}", (3, 1), (2, 5), "different", effect)
        for name, effect in [("zero", 0.0), ("weak", 0.75), ("strong", 4.0)]
    ],
    Scenario("cross_role_swap", (2, 0), (0, 2), "different", 4.0),
    Scenario("cross_reference_exhaustion", (3, 0), (0, 2), "different", 4.0),
    Scenario("cross_clustered_zero", (2, 2), (3, 3), "different", 0.0, 2.5),
    Scenario("cross_clustered_strong", (2, 2), (3, 3), "different", 4.0, 2.5),
    Scenario("overlapping_three_plate", (1, 1, 1), (2, 2, 2), "different", 4.0),
]


def fixed_allowed(plate, x, y, rule):
    """Evaluate only the declared fixed metadata restriction."""
    if rule == "none":
        return True
    if rule == "same":
        return plate[x] == plate[y]
    if rule == "different":
        return plate[x] != plate[y]
    raise ValueError(f"Unknown fixed rule: {rule}")


def make_data(scenario, seed):
    """Generate Gaussian profiles with optional plate shifts and paired units."""
    members_plate = np.repeat(
        np.arange(len(scenario.members_per_plate)), scenario.members_per_plate
    )
    controls_plate = np.repeat(
        np.arange(len(scenario.controls_per_plate)), scenario.controls_per_plate
    )
    plate = np.concatenate([members_plate, controls_plate])
    n = len(members_plate)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(len(plate), scenario.dimensions))
    features += scenario.plate_effect * plate[:, None]
    units = np.empty(len(plate), dtype=int)
    next_unit = 0
    for p in np.unique(plate):
        indices = np.flatnonzero(plate == p)
        units[indices] = next_unit + np.arange(len(indices)) // 2
        next_unit += math.ceil(len(indices) / 2)
    if scenario.cluster_sd:
        features += (
            scenario.cluster_sd
            * rng.normal(size=(next_unit, scenario.dimensions))[units]
        )
    # Supply a symmetric float32 similarity, matching production's rank-key contract.
    delta = features[:, None] - features[None, :]
    similarities = (-np.square(delta).sum(axis=2)).astype(np.float32)
    keys = np.float32(1) - similarities

    def similarity(pairs):
        return similarities[pairs[:, 0], pairs[:, 1]]

    positives = np.array(
        list(itertools.combinations(range(n), 2)), dtype=np.int64
    ).reshape(-1, 2)
    negatives = np.array(
        [
            (x, y)
            for x in range(n)
            for y in range(n, len(plate))
            if fixed_allowed(plate, x, y, scenario.fixed_rule)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    return plate, units, keys, similarity, positives, negatives


def make_layout(scenario, seed):
    """Build observed pairs from the declared rules, then use production grouping."""
    plate, units, keys, similarity, positives, negatives = make_data(scenario, seed)
    n = sum(scenario.members_per_plate)
    if n < 2:
        raise ValueError(
            "The study requires a complete positive graph with >=2 members"
        )
    counts = np.bincount(negatives[:, 0], minlength=n)
    group = production.group_layouts(
        [np.arange(n)], np.arange(n), np.full(n, n - 1), counts, positives, negatives
    )[0]
    members, edges, blocks, block_of = group
    pool = production._Pool(blocks, similarity)
    layout = production._Layout(
        members, edges, block_of, similarity, pool, draws=100_000
    )
    profiles = np.concatenate([members, pool.refs])
    # These scenarios put every profile in the group's original union.
    if set(profiles) != set(range(len(plate))):
        raise ValueError("Scenario includes profiles outside the original group union")
    pools = [
        np.concatenate(
            [np.flatnonzero(block_of == b), n + np.arange(pool.cr_lo[b], pool.cr_hi[b])]
        )
        for b in range(len(blocks))
    ]
    return dict(
        plate=plate[profiles],
        units=units[profiles],
        keys=keys[np.ix_(profiles, profiles)],
        profiles=profiles,
        pools=pools,
        layout=layout,
        group=group,
        similarity=similarity,
        scenario=scenario,
    )


def assignments(data):
    """Enumerate uniform subsets per block; complete positives make slot order irrelevant."""
    layout = data["layout"]
    slots = [np.flatnonzero(layout.slot_block == b) for b in range(len(data["pools"]))]
    choices = [
        list(itertools.combinations(pool, len(slot)))
        for pool, slot in zip(data["pools"], slots)
    ]
    out = []
    for parts in itertools.product(*choices):
        pm = np.empty(layout.n, dtype=np.int64)
        for slot, part in zip(slots, parts):
            pm[slot] = part
        out.append(pm)
    return np.asarray(out)


def references(data, pm, slot, mode):
    """Move group/control roles with labels, then choose the specified reference universe."""
    if mode not in MODES:
        raise ValueError(mode)
    universe = (
        range(len(data["plate"]))
        if mode == "B_global"
        else data["pools"][data["layout"].slot_block[slot]]
    )
    chosen = set(pm)
    refs = [int(y) for y in universe if y not in chosen]
    if mode != "A":
        refs = [
            y
            for y in refs
            if fixed_allowed(data["plate"], pm[slot], y, data["scenario"].fixed_rule)
        ]
    return refs


def naive_map(data, pm, mode):
    """Compute AP directly by sorting positive/negative lists, independently of rank tables.

    Positives precede negatives at equal float32 rank keys, as in the pinned
    production statistic. With positives but no negatives, AP is defined as 1;
    zero-reference slots are recorded, never dropped from the group mean.
    """
    if len(pm) < 2:
        raise ValueError("AP undefined without positive neighbours")
    scores, counts = [], []
    for slot, x in enumerate(pm):
        positives = [int(y) for t, y in enumerate(pm) if t != slot]
        refs = references(data, pm, slot, mode)
        entries = [(data["keys"][x, y], 0) for y in positives]
        entries += [(data["keys"][x, y], 1) for y in refs]
        entries.sort()
        hits, total = 0, 0.0
        for rank, (_, negative) in enumerate(entries, start=1):
            if not negative:
                hits += 1
                total += hits / rank
        scores.append(total / len(positives))
        counts.append(len(refs))
    return float(np.mean(scores)), counts


def evaluate(data, pm, mode):
    """Evaluate all assignments without conditioning or deleting zero-reference slots."""
    results = [naive_map(data, row, mode) for row in pm]
    values = np.array([r[0] for r in results])
    counts = np.array([r[1] for r in results])
    return values, counts


def summary(values, observed, sampled=False):
    """Report a finite distribution and its upper-tail probability at the identity."""
    if not len(values):
        return {"status": "empty_support", "n": 0}
    hit = values >= production.tie_thresholds(observed)
    n = len(values)
    return dict(
        status="ok",
        n=n,
        mean=float(values.mean()),
        sd=float(values.std()),
        q90=float(np.quantile(values, 0.9)),
        q95=float(np.quantile(values, 0.95)),
        q99=float(np.quantile(values, 0.99)),
        p=float((hit.sum() + 1) / (n + 1) if sampled else hit.mean()),
        p_mc_se=float(np.sqrt(hit.mean() * (1 - hit.mean()) / n)) if sampled else 0.0,
    )


def compare(a, b, observed, sampled=False):
    """Compare paired assignments, including paired Monte Carlo tail uncertainty."""
    locations = np.unique(np.concatenate([a, b]))
    ecdf_a = np.searchsorted(np.sort(a), locations, side="right") / len(a)
    ecdf_b = np.searchsorted(np.sort(b), locations, side="right") / len(b)
    threshold = production.tie_thresholds(observed)
    difference = (b >= threshold).astype(float) - (a >= threshold)
    scale = len(a) / (len(a) + 1) if sampled else 1
    return dict(
        wasserstein=float(wasserstein_distance(a, b)),
        max_ecdf_difference=float(np.max(np.abs(ecdf_a - ecdf_b))),
        mean_difference=float(np.mean(b - a)),
        p_difference=float(difference.mean() * scale),
        paired_p_difference_mc_se=float(
            difference.std(ddof=1) / np.sqrt(len(a)) * scale
        )
        if sampled
        else 0.0,
    )


def reference_summary(counts, base):
    """Report how often the statistic's negative-list sizes change."""
    return dict(
        minimum=int(counts.min()),
        maximum=int(counts.max()),
        mean=float(counts.mean()),
        zero_reference_slot_fraction=float(np.mean(counts == 0)),
        assignments_with_zero_reference_fraction=float(
            np.mean((counts == 0).any(axis=1))
        ),
        changed_slot_fraction=float(np.mean(counts != base)),
    )


def run_one(scenario, seed, draws):
    """Compare exact finite supports and a production sampler using common assignments."""
    record = {"scenario": asdict(scenario), "data_seed": seed}
    try:
        data = make_layout(scenario, seed)
    except ValueError as error:
        if scenario.name != "overlapping_three_plate":
            raise
        return dict(record, status="unsupported_by_A", reason=str(error))
    layout = data["layout"]
    pm = assignments(data)
    exact_a = production._group_maps_np(layout, pm)
    naive_a, counts_a = evaluate(data, pm, "A")
    np.testing.assert_allclose(exact_a, naive_a, rtol=0, atol=1e-14)
    identity = np.arange(layout.n)
    observed = naive_map(data, identity, "A")[0]
    for mode in MODES:
        np.testing.assert_allclose(
            naive_map(data, identity, mode)[0], observed, rtol=0, atol=1e-14
        )
    exact = {"A": summary(exact_a, observed)}
    comparisons, counts = {}, {"A": reference_summary(counts_a, counts_a)}
    local_counts = None
    for mode in MODES[1:]:
        values, refs = evaluate(data, pm, mode)
        exact[mode] = summary(values, observed)
        comparisons[mode] = compare(exact_a, values, observed)
        counts[mode] = reference_summary(refs, counts_a)
        if mode == "B_local":
            local_counts = refs
    # C is a conditional assignment law, not a filtering statistic.
    valid = (local_counts == counts_a).all(axis=1)
    exact["C_conditioned_A"] = summary(exact_a[valid], observed)
    exact["C_conditioned_A"]["support_fraction"] = float(valid.mean())
    exact["C_conditioned_A"]["identity_only"] = bool(valid.sum() == 1)
    out = np.empty(draws)
    hits, sampler_identity, _ = production.permutation_null(
        *data["group"], data["similarity"], draws, DRAW_SEED, out=out
    )
    sampled_pm = production._draws_np(
        layout, 0, draws, *production.group_key(DRAW_SEED, data["group"][0])
    )
    np.testing.assert_array_equal(out, production._group_maps_np(layout, sampled_pm))
    np.testing.assert_allclose(sampler_identity, observed, rtol=0, atol=1e-14)
    sampled = {"A": summary(out, observed, sampled=True)}
    assert sampled["A"]["p"] == (hits + 1) / (draws + 1)
    sampled_comparisons = {}
    for mode in MODES[1:]:
        values, _ = evaluate(data, sampled_pm, mode)
        sampled[mode] = summary(values, observed, sampled=True)
        sampled_comparisons[mode] = compare(out, values, observed, sampled=True)
    return dict(
        record,
        status="ok",
        observed_map=observed,
        exact_support=len(pm),
        profile_plates=data["plate"].tolist(),
        profile_units=data["units"].tolist(),
        original_profile_indices=data["profiles"].tolist(),
        original_members=data["group"][0].tolist(),
        block_pool_codes=[p.tolist() for p in data["pools"]],
        slot_blocks=layout.slot_block.tolist(),
        exact=exact,
        comparisons=comparisons,
        references=counts,
        sampler=sampled,
        sampler_comparisons=sampled_comparisons,
        max_production_naive_absolute_error=float(np.max(np.abs(exact_a - naive_a))),
    )


def matched_plate_guard_diagnostic():
    """Exercise public guards on matched observed graphs, without bypassing a rejection."""
    from copairs.map import average_precision, mean_average_precision

    plate = np.array([0, 1, 0, 0, 1, 1])
    meta = pd.DataFrame(
        {
            "group": [
                "query",
                "query",
                "control_0",
                "control_1",
                "control_2",
                "control_3",
            ],
            "is_control": [False, False, True, True, True, True],
            "plate": plate,
        }
    )
    features = np.array([[1, 0], [0, 1], [1, 0.1], [1, 0.2], [0.1, 1], [0.2, 1]])

    def similarity(left, right):
        return -np.square(left - right).sum(axis=1)

    cases = []
    for positive_plate_rule in (False, True):
        rules = dict(
            pos_sameby=["group"],
            pos_diffby=["plate"] if positive_plate_rule else [],
            neg_sameby=[],
            neg_diffby=["is_control", "plate"],
        )
        ap, pairs = average_precision(
            meta,
            features,
            **rules,
            distance=similarity,
            progress_bar=False,
            return_pairs=True,
        )
        query = ap.loc[ap["group"] == "query"]
        case = dict(
            name="positive_and_negative_plate"
            if positive_plate_rule
            else "negative_plate_only",
            rules=rules,
            positive_pairs=sorted(pairs.pos.tolist()),
            negative_pairs=sorted(pairs.neg.tolist()),
            observed_map=float(query["average_precision"].mean()),
        )
        try:
            mean_average_precision(
                query,
                ["group"],
                null_size=32,
                threshold=0.05,
                seed=DRAW_SEED,
                progress_bar=False,
                max_workers=1,
                null="permutation",
                pairs=pairs,
            )
        except ValueError as error:
            if not positive_plate_rule or "pos_diffby" not in str(error):
                raise
            case.update(status="rejected_by_production_guard", reason=str(error))
        else:
            case["status"] = "accepted_by_production"
        cases.append(case)
    # Metadata-only enumeration, not a null computation or lower-level kernel bypass.
    pools = [[0, 4, 5], [1, 2, 3]]
    metadata_assignments = []
    for chosen in itertools.product(*pools):
        forbidden_negatives = sum(
            plate[x] == plate[y]
            for x, pool in zip(chosen, pools)
            for y in pool
            if y != x
        )
        metadata_assignments.append(
            dict(
                chosen=list(chosen),
                fixed_positive_edges=int(plate[chosen[0]] != plate[chosen[1]]),
                forbidden_pool_negative_edges=int(forbidden_negatives),
            )
        )
    return dict(
        profiles=meta.to_dict(orient="records"),
        features=features.tolist(),
        similarity="negative squared Euclidean",
        cases=cases,
        pool_assignment_metadata=metadata_assignments,
        guard_bypassed=False,
        positive_reapplication_maps="not computed; zero-positive assignments are undefined, not dropped",
    )


def render_report(result):
    """Render exact ranges across datasets; preserve per-dataset detail in JSON."""
    lines = [
        "# PR #156 synthetic null-layout results",
        "",
        f"Pinned production source: `{PIN}`. Data seeds: {result['data_seeds']}; "
        f"Philox sampler seed: {DRAW_SEED}; {result['draws_per_dataset']} draws/dataset.",
        "",
        "Exact statistics below enumerate equally weighted blockwise subsets. Complete "
        "positive graphs make within-block slot order immaterial. Ranges span dataset "
        "seeds, **not confidence intervals**. Exact comparisons have no Monte Carlo "
        "error conditional on the generated features; they do not remove dataset uncertainty.",
        "",
        "A = production pools; B_local = filter A references by fixed plate rule; "
        "B_global = reapply the rule to all unlabelled profiles in the original union. "
        "Group/control roles move with labels in all models. Δp = p(B) − p(A).",
        "",
        "| Scenario | Exact support | B_local W1 | B_global W1 | B_local Δp | B_global Δp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        rows = [r for r in result["results"] if r["scenario"]["name"] == scenario.name]
        if rows[0]["status"] != "ok":
            lines.append(f"| {scenario.name} | unsupported | — | — | — | — |")
            continue

        def span(mode, key):
            values = [r["comparisons"][mode][key] for r in rows]
            return f"{min(values):.4f}–{max(values):.4f}"

        lines.append(
            f"| {scenario.name} | {rows[0]['exact_support']} | "
            + " | ".join(
                [
                    span("B_local", "wasserstein"),
                    span("B_global", "wasserstein"),
                    span("B_local", "p_difference"),
                    span("B_global", "p_difference"),
                ]
            )
            + " |"
        )
    supported = [r for r in result["results"] if r["status"] == "ok"]
    seed = result["data_seeds"][0]
    selected = [
        r
        for r in supported
        if r["data_seed"] == seed
        and r["scenario"]["name"]
        in {
            "cross_balanced_zero",
            "cross_balanced_strong",
            "cross_imbalanced_strong",
            "cross_clustered_strong",
        }
    ]
    lines += [
        "",
        f"## Distribution detail (data seed {seed})",
        "",
        "All seeds are in JSON; these rows illustrate both a zero-effect and stress cases.",
        "",
        "| Scenario | Model | Mean | SD | q95 | Exact p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        for mode in MODES:
            s = row["exact"][mode]
            lines.append(
                f"| {row['scenario']['name']} | {mode} | {s['mean']:.4f} | {s['sd']:.4f} | {s['q95']:.4f} | {s['p']:.4f} |"
            )
    lines += [
        "",
        "## Reference-count and conditioning diagnostics",
        "",
        "Counts depend on layout, not feature seed/effect. Means are over slots and exact assignments. "
        "The zero-reference fraction is over slots, not whole draws.",
        "",
        "| Layout example | Mean references A / local / global | Local / global zero-reference fraction | C support / A support |",
        "|---|---:|---:|---:|",
    ]
    for row in supported:
        if row["data_seed"] != seed or row["scenario"]["name"] not in {
            "cross_balanced_strong",
            "cross_imbalanced_strong",
            "cross_role_swap",
            "cross_reference_exhaustion",
        }:
            continue
        refs = row["references"]
        means = " / ".join(f"{refs[m]['mean']:.3f}" for m in MODES)
        zeros = " / ".join(
            f"{refs[m]['zero_reference_slot_fraction']:.3f}" for m in MODES[1:]
        )
        c = row["exact"]["C_conditioned_A"]
        label = " (identity only)" if c["identity_only"] else ""
        lines.append(
            f"| {row['scenario']['name']} | {means} | {zeros} | {c['n']} / {row['exact_support']}{label} |"
        )
    max_error = max(r["max_production_naive_absolute_error"] for r in supported)
    max_se = max(
        r["sampler_comparisons"][m]["paired_p_difference_mc_se"]
        for r in supported
        for m in MODES[1:]
    )
    max_mc_error = max(
        abs(
            r["sampler_comparisons"][m]["p_difference"]
            - r["comparisons"][m]["p_difference"]
        )
        for r in supported
        for m in MODES[1:]
    )
    lines += [
        "",
        "## Sampler / independent-calculation checks",
        "",
        f"{len(supported)} supported datasets; {sum(r['exact_support'] for r in supported):,} "
        f"enumerated A values compared against independent list sorting; maximum absolute error {max_error:.3g}. "
        "Production sampler output also equals evaluation on the extracted Philox assignments, draw for draw.",
        "",
        f"Across all paired sampler comparisons, maximum approximate SE(Δp) = {max_se:.4f}; "
        f"maximum |sampled Δp − exact Δp| = {max_mc_error:.4f}. "
        "These are descriptive diagnostics, not simultaneous uncertainty bounds. The main tables use exact values.",
        "",
    ]
    guard = result["matched_plate_guard"]
    metadata = guard["pool_assignment_metadata"]
    undefined = sum(row["fixed_positive_edges"] == 0 for row in metadata)
    negative_violations = sum(
        row["forbidden_pool_negative_edges"] > 0 for row in metadata
    )
    lines += [
        "",
        "## Positive/negative policy consistency and current guard asymmetry",
        "",
        "The owner requires one pool-based relabelling policy for positive and negative "
        "pair restrictions; fixed metadata handling on both sides is future follow-up. "
        "The current positive-only `pos_diffby` rejection needs review for alignment and "
        "has not been separately approved as the final policy. This report documents "
        "current behavior; it does not claim that policy alignment has been implemented.",
        "",
        "A matched six-profile example has one query member and two controls on each "
        "of two plates. Both specifications have the same observed positive edge (0,1) "
        "and negatives (0,4), (0,5), (1,2), (1,3). Public `average_precision` followed by "
        "`mean_average_precision(null='permutation')` gives:",
        "",
        "| Positive plate rule | Negative plate rule | Pinned production outcome |",
        "|---|---|---|",
        f"| none | `neg_diffby=['is_control','plate']` | {guard['cases'][0]['status']} |",
        f"| `pos_diffby=['plate']` | same negative rule | {guard['cases'][1]['status']} |",
        "",
        "The rejection says `pos_diffby values do not fix which relabelled profiles are "
        "positives`; its full text and explicit pair lists are saved in results.json. "
        "No rejected specification is assigned a production p-value, and **no guard is bypassed**.",
        "",
        f"Metadata-only enumeration of the {len(metadata)} blockwise assignments finds "
        f"{negative_violations} with forbidden A negative edges under a reapplied fixed plate "
        f"rule, and {undefined} with zero positive edges if the fixed positive plate rule "
        "were reapplied. This is not a lower-level-kernel null run: no relabelled mAP "
        "or permutation p-value is computed for that rejected specification, and undefined assignments are not "
        "silently dropped. A full positive-graph/reference comparison, including a stated "
        "policy for zero-positive statistics, remains a coverage gap. The main bounded "
        "A/B_local/B_global comparisons above remain negative-reference comparisons "
        "with complete positive graphs and no `pos_diffby` rule.",
    ]
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- Shared-reference and same-plate scenarios are invariance checks, not evidence "
        "that cross-plate restrictions are harmless. Differences can exist even at zero nuisance effect.",
        "- B_local only removes negatives: its AP and upper-tail p cannot decrease on a "
        "common assignment. B_global can also add negatives from other original blocks, so "
        "neither its distribution nor its p-value has that ordering.",
        "- Filtering changes reference counts and thus the statistic. Zero-negative slots "
        "retain all positives and have AP=1; they are reported, not omitted. Neither B "
        "is asserted to be a better-calibrated test or to test the same hypothesis as A.",
        "- C_conditioned_A keeps A's statistic but accepts only assignments whose every A "
        "negative edge respects the fixed rule. Its separate support/p-values are in JSON; "
        "identity-only support is uninformative, not a substitute for A or B.",
        "- Three-plate cross-reference sets overlap and production rejects them. No p-values "
        "or replacement nulls are reported for those cases.",
        "- Clustered profiles share a Gaussian unit offset within each adjacent pair on a "
        "plate. Profilewise assignments need not be exchangeable; retaining geometry "
        "does not prove validity. No calibration/rejection-rate claim is made.",
        "- Small complete-positive graphs, three synthetic seeds and one group per dataset "
        "are a bounded first slice, not an exhaustive design survey. Effects, dimensions, "
        "rules, identities, blocks and all per-seed summaries are in results.json.",
        "- JSON includes means, population SD, q90/q95/q99, max ECDF distance, W1, exact "
        "upper-tail p-values, changed/zero reference counts, plus the production-sampler "
        "comparison on common assignments with approximate paired Monte Carlo SEs. "
        "Sampler p-values use (1+hits)/(1+draws); exact p-values use hits/support.",
        "",
        "This analysis informs future consideration only; it does not change the "
        "decision to retain A or production code.",
        "",
    ]
    return "\n".join(lines)


def main():
    """Run the bounded exact study and a modest production-sampler cross-check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 41])
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("--draws must be >=2")
    rows = []
    for scenario in SCENARIOS:
        for seed in args.seeds:
            print(f"{scenario.name}: seed {seed}", flush=True)
            rows.append(run_one(scenario, seed, args.draws))
    result = dict(
        production_commit=PIN,
        baseline_commit="310e1ce",
        data_seeds=args.seeds,
        draw_seed=DRAW_SEED,
        draws_per_dataset=args.draws,
        python=platform.python_version(),
        dependencies={
            name: version(name)
            for name in ["numpy", "scipy", "pandas", "pytest", "ruff"]
        },
        invocation=["study.py", *sys.argv[1:]],
        matched_plate_guard=matched_plate_guard_diagnostic(),
        results=rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "REPORT.md").write_text(render_report(result))


if __name__ == "__main__":
    main()
