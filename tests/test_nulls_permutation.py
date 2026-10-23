"""Tests for the label-permutation mAP null (copairs.nulls.permutation)."""

import json
import pickle
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from copairs.map import average_precision, mean_average_precision
from copairs.nulls import permutation, tie_thresholds

SPEC = dict(pos_sameby=["g"], pos_diffby=[], neg_sameby=[], neg_diffby=["g"])
LAYOUTS = [
    (1, 3, 5, SPEC),
    (2, 2, 4, dict(SPEC, neg_sameby=["plate"])),
    (2, 2, 3, dict(SPEC, pos_diffby=["plate"], neg_sameby=["plate"])),
    (3, 3, 20, dict(SPEC, neg_sameby=["plate"])),
]
LAYOUT_IDS = ["one-pool", "plate-blocks", "plate-blocks-diffby", "three-plates"]


def layout_meta(plates, reps, ctrls):
    """``reps`` replicates of group "q" and ``ctrls`` controls on each plate."""
    rows = []
    for p in range(plates):
        rows += [("q", p)] * reps + [(f"c{p}_{i}", p) for i in range(ctrls)]
    return pd.DataFrame(rows, columns=["g", "plate"])


def relabel(meta, chosen):
    """Give group "q" to the ``chosen`` rows and a unique control label to the rest."""
    g = np.array([f"c{i}" for i in range(len(meta))], dtype=object)
    g[list(chosen)] = "q"
    return meta.assign(g=g)


def enumerate_maps(meta, feats, spec, blocks):
    """MAP of group "q" under every relabelling, computed by average_precision."""
    k = [int((meta["g"].iloc[b] == "q").sum()) for b in blocks]
    values = []
    for choice in itertools.product(
        *(itertools.combinations(b, kb) for b, kb in zip(blocks, k))
    ):
        chosen = [i for part in choice for i in part]
        ap = average_precision(relabel(meta, chosen), feats, progress_bar=False, **spec)
        values.append(ap.loc[ap["g"] == "q", "average_precision"].mean())
    return np.array(values)


def group_of_q(meta, feats, spec):
    """Observed mAP, layout and pairs of group "q"."""
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **spec
    )
    ap = ap.query("n_pos_pairs > 0 and g == 'q'")
    layout = permutation.group_layouts(
        [np.arange(len(ap))],
        ap.index.to_numpy(),
        ap["n_pos_pairs"].to_numpy(),
        (ap["n_total_pairs"] - ap["n_pos_pairs"]).to_numpy(),
        pairs.pos,
        pairs.neg,
    )[0]
    return ap["average_precision"].mean(), layout, pairs


def draws_of(meta, feats, spec, null_size, seed=0):
    """Observed mAP and the null's draws for group "q"."""
    observed, layout, pairs = group_of_q(meta, feats, spec)
    out = np.empty(null_size)
    count, identity, _ = permutation.permutation_null(
        *layout, pairs.similarity, null_size, seed, out=out
    )
    return observed, identity, count, out


def scalar_draws(layout, start, size, k0, k1):
    """Return draws ``start, ...`` from the scalar functions, one at a time."""
    n = layout.n
    pm = np.empty(n, dtype=np.int64)
    chosen = np.empty(int(np.diff(layout.bm_ptr).max()), dtype=np.int64)
    scratch = (
        np.empty(n, np.int64),
        np.empty(n, np.int64),
        np.empty(n, np.uint64),
        np.empty(n, np.int32),
        np.empty(n, np.float64),
    )
    pms, maps = [], []
    for j in range(start, start + size):
        permutation._draw(j, *layout.draw_args(), k0, k1, pm, chosen)
        pms.append(pm.copy())
        maps.append(permutation._group_map(pm, *layout.map_args(), *scratch))
    return np.array(pms), np.array(maps)


@pytest.mark.parametrize("tables", ["full", "full-ranks", "ranked", "search"])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "tied"])
@pytest.mark.parametrize("plates, reps, ctrls, spec", LAYOUTS, ids=LAYOUT_IDS)
def test_vectorized_matches_scalar_draws(
    plates, reps, ctrls, spec, ties, tables, monkeypatch
):
    """The NumPy draws and mAPs equal the scalar functions', bit for bit.

    Small layouts gather keys from one (code, code) table, and ranks from one
    too ("full-ranks") or, given enough draws, the ranks plus the drawn block
    members before them ("full"); larger ones read the rank tables, and pools
    above RANK_POOL search sorted rows.
    """
    if tables not in ("full", "full-ranks"):
        monkeypatch.setattr(permutation, "_FULL_TABLE", 0)
    if tables == "search":
        monkeypatch.setattr(permutation, "RANK_POOL", 0)
    meta = layout_meta(plates, reps, ctrls)
    feats = np.random.default_rng(plates + ctrls).standard_normal((len(meta), 6))
    if ties:  # rounding ties many similarities and zeroes some profiles (NaN)
        feats = np.round(feats)
    observed, (members, edges, blocks, block_of), pairs = group_of_q(meta, feats, spec)
    pool = permutation._Pool(blocks, pairs.similarity)
    draws = 0 if tables == "full-ranks" else 300
    layout = permutation._Layout(
        members, edges, block_of, pairs.similarity, pool, draws=draws
    )
    assert pool.ranked == (tables != "search")
    assert (layout.full_keys is not None) == tables.startswith("full")
    assert (layout.full_ranks is not None) == (tables == "full-ranks")
    assert (layout.full_before is not None) == (tables == "full")
    k0, k1 = permutation.group_key(3, members)
    start = 2**32 - 50  # draws on both sides of the counter's word boundary
    pm, maps = scalar_draws(layout, start, 300, k0, k1)
    got_pm = permutation._draws_np(layout, start, 300, k0, k1)
    np.testing.assert_array_equal(got_pm, pm)
    np.testing.assert_array_equal(permutation._group_maps_np(layout, got_pm), maps)
    identity = np.arange(layout.n)
    scalar_identity = scalar_identity_map(layout, identity)
    assert permutation._group_maps_np(layout, identity[None])[0] == scalar_identity
    assert scalar_identity == pytest.approx(observed, abs=1e-12)


def scalar_identity_map(layout, pm):
    """MAP of the observed labels from the scalar function."""
    n = layout.n
    scratch = (
        np.empty(n, np.int64),
        np.empty(n, np.int64),
        np.empty(n, np.uint64),
        np.empty(n, np.int32),
        np.empty(n, np.float64),
    )
    return permutation._group_map(pm, *layout.map_args(), *scratch)


def test_hits_match_scalar_draws():
    """The count of draws reaching the observed mAP is the scalar draws' count."""
    meta = layout_meta(2, 3, 6)
    feats = np.random.default_rng(21).standard_normal((len(meta), 5))
    spec = dict(SPEC, neg_sameby=["plate"])
    observed, group, pairs = group_of_q(meta, feats, spec)
    count, identity, pool = permutation.permutation_null(
        *group, pairs.similarity, 700, 5
    )
    members, edges, _, block_of = group
    layout = permutation._Layout(members, edges, block_of, pairs.similarity, pool)
    _, maps = scalar_draws(layout, 0, 700, *permutation.group_key(5, members))
    assert count == (maps >= tie_thresholds(identity)).sum()
    assert 0 < count < 700


def test_draws_do_not_depend_on_chunks(monkeypatch):
    """Draw j is the same whatever the chunk sizes of draws and pool tables."""
    meta = layout_meta(2, 2, 6)
    feats = np.random.default_rng(22).standard_normal((len(meta), 5))
    spec = dict(SPEC, neg_sameby=["plate"])
    full = draws_of(meta, feats, spec, 1000)
    monkeypatch.setattr(permutation._Layout, "chunk", lambda self: 77)
    monkeypatch.setattr(permutation, "_PAIR_CHUNK", 7)
    monkeypatch.setattr(permutation, "_CHUNK_ELEMENTS", 5)
    chunked = draws_of(meta, feats, spec, 1000)
    assert full[1:3] == chunked[1:3]
    np.testing.assert_array_equal(full[3], chunked[3])


@pytest.mark.parametrize("plates, reps, ctrls, spec", LAYOUTS[:3], ids=LAYOUT_IDS[:3])
def test_matches_relabelled_average_precision(plates, reps, ctrls, spec):
    """Draws follow the mAP of uniformly random relabellings, as copairs computes them."""
    meta = layout_meta(plates, reps, ctrls)
    feats = np.random.default_rng(plates * 10 + ctrls).standard_normal((len(meta), 6))
    per_plate = reps + ctrls
    blocks = [list(range(p * per_plate, (p + 1) * per_plate)) for p in range(plates)]
    exact = enumerate_maps(meta, feats, spec, blocks)
    observed, identity, count, draws = draws_of(meta, feats, spec, 60_000)
    assert identity == pytest.approx(observed, abs=1e-12)
    # Every draw is one of the relabellings' mAPs, with their frequencies.
    values, freq = np.unique(np.round(exact, 9), return_counts=True)
    ix = np.searchsorted(values, np.round(draws, 9))
    assert (values[np.minimum(ix, len(values) - 1)] == np.round(draws, 9)).all()
    expected = freq / freq.sum() * len(draws)
    chi2 = ((np.bincount(ix, minlength=len(values)) - expected) ** 2 / expected).sum()
    assert stats.chi2.sf(chi2, len(values) - 1) > 1e-3
    assert count == (draws >= identity - 1e-10).sum()


def test_calibrated_on_exchangeable_profiles():
    """Groups of profiles drawn like their controls get uniform p-values."""
    rng = np.random.default_rng(5)
    n_groups, reps, ctrls = 300, 3, 20
    meta = pd.DataFrame(
        {"g": [f"q{i // reps}" for i in range(n_groups * reps)] + ["ctrl"] * ctrls}
    )
    meta["ref"] = np.where(meta["g"] == "ctrl", np.arange(len(meta)), -1)
    feats = rng.standard_normal((len(meta), 4))
    spec = dict(
        pos_sameby=["g", "ref"], pos_diffby=[], neg_sameby=[], neg_diffby=["g", "ref"]
    )
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **spec
    )
    ap = ap[meta["g"] != "ctrl"]
    res = mean_average_precision(
        ap, ["g"], 999, 0.05, 0, progress_bar=False, null="permutation", pairs=pairs
    )
    p = res["p_value"].to_numpy()
    assert stats.binom(n_groups, 0.05).ppf(0.0005) <= (p <= 0.05).sum()
    assert (p <= 0.05).sum() <= stats.binom(n_groups, 0.05).ppf(0.9995)


def test_seeded_and_reproducible():
    """The same seed gives the same p-values; another seed generally differs."""
    meta = layout_meta(1, 3, 12)
    feats = np.random.default_rng(1).standard_normal((len(meta), 5))
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **SPEC
    )
    kwargs = dict(progress_bar=False, null="permutation", pairs=pairs)
    p = [
        mean_average_precision(ap, ["g"], 500, 0.05, seed, **kwargs)["p_value"].tolist()
        for seed in (0, 0, 1)
    ]
    assert p[0] == p[1]
    assert p[0] != p[2]


def test_default_null_uses_attached_pairs():
    """AP frames carry their pairs, so the default permutation null needs no pairs=."""
    meta = layout_meta(2, 3, 6)
    feats = np.random.default_rng(10).standard_normal((len(meta), 5))
    spec = dict(SPEC, neg_sameby=["plate"])
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **spec
    )
    explicit = mean_average_precision(
        ap, ["g"], 500, 0.05, 0, progress_bar=False, null="permutation", pairs=pairs
    )
    filtered = ap[ap["g"] == "q"].copy()  # filtering and copies keep the pairs
    default = mean_average_precision(filtered, ["g"], 500, 0.05, 0, progress_bar=False)
    assert default["p_value"].tolist() == explicit["p_value"].tolist()
    halves = pd.concat([ap.iloc[:10], ap.iloc[10:]])  # pieces of one frame
    assert (
        mean_average_precision(halves, ["g"], 500, 0.05, 0, progress_bar=False)[
            "p_value"
        ].tolist()
        == explicit["p_value"].tolist()
    )


def test_frames_without_pairs_raise():
    """Saved, pickled or merged AP frames lose their pairs and say so."""
    meta = layout_meta(1, 3, 6)
    feats = np.random.default_rng(11).standard_normal((len(meta), 5))
    ap = average_precision(meta, feats, progress_bar=False, **SPEC)
    other = average_precision(
        meta.assign(g=meta["g"] + "x"), feats, progress_bar=False, **SPEC
    )
    assert json.loads(json.dumps(ap.attrs)) == {"copairs_pairs": {}}  # as to_parquet
    for frame in (pickle.loads(pickle.dumps(ap)), pd.concat([ap, other])):
        with pytest.raises(ValueError, match="needs the pairs"):
            mean_average_precision(frame, ["g"], 100, 0.05, 0, progress_bar=False)
        res = mean_average_precision(
            frame, ["g"], 100, 0.05, 0, progress_bar=False, null="copied"
        )
        assert len(res)


def test_rejects_unsupported_layouts():
    """Missing pairs, positives outside the group and mismatched pairs raise."""
    meta = layout_meta(1, 3, 5)
    meta["moa"] = ["m"] * 3 + ["c0", "c1", "c2", "c3", "c4"]
    feats = np.random.default_rng(2).standard_normal((len(meta), 4))
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **SPEC
    )
    args = (["g"], 100, 0.05, 0)
    kwargs = dict(progress_bar=False, null="permutation")
    bare = ap.copy()
    bare.attrs.clear()
    with pytest.raises(ValueError, match="needs the pairs.*null='joint'"):
        mean_average_precision(bare, *args, **kwargs)
    other = dict(SPEC, neg_diffby=["g", "plate"])
    _, mismatched = average_precision(
        meta.assign(plate=[0, 0, 1, 0, 0, 1, 1, 1]), feats, progress_bar=False,
        return_pairs=True, **other,
    )  # fmt: skip
    with pytest.raises(ValueError, match="in the AP table"):
        mean_average_precision(ap, *args, pairs=mismatched, **kwargs)
    spec = dict(pos_sameby=["moa"], pos_diffby=[], neg_sameby=[], neg_diffby=["moa"])
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **spec
    )
    meta_ap = ap.assign(half=[0, 0, 1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="positives outside the group.*null='joint'"):
        mean_average_precision(
            meta_ap, ["moa", "half"], 100, 0.05, 0, pairs=pairs, **kwargs
        )


def test_group_without_references():
    """Members with positives but no negatives have one relabelling: p = 1."""
    meta = pd.DataFrame(
        {
            "g": ["q"] * 3 + ["r"] * 2 + [f"c{i}" for i in range(5)],
            "plate": [0] * 3 + [1] * 7,
        }
    )
    feats = np.random.default_rng(4).standard_normal((len(meta), 4))
    spec = dict(SPEC, neg_sameby=["plate"])
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **spec
    )
    res = mean_average_precision(
        ap, ["g"], 100, 0.05, 0, progress_bar=False, null="permutation", pairs=pairs
    ).set_index("g")
    assert res.loc["q", "mean_average_precision"] == 1.0
    assert res.loc["q", "p_value"] == 1.0
    assert res.loc["r", "p_value"] < 1.0


def test_rank_table_matches_binary_search(monkeypatch):
    """Pools above RANK_POOL search sorted rows; both paths give the same draws."""
    meta = layout_meta(2, 2, 4)
    feats = np.random.default_rng(3).standard_normal((len(meta), 5))
    spec = dict(SPEC, neg_sameby=["plate"])
    ranked = draws_of(meta, feats, spec, 5000)
    monkeypatch.setattr(permutation, "RANK_POOL", 0)
    searched = draws_of(meta, feats, spec, 5000)
    assert ranked[1:3] == searched[1:3]
    np.testing.assert_array_equal(ranked[3], searched[3])


def test_pvalues_do_not_depend_on_workers(monkeypatch):
    """Groups sharing pools give the same p-values on 1 or 4 threads, or alone."""
    rows = []
    for p in range(3):
        rows += [(f"c{p}_{i}", p) for i in range(6)]
    for g in range(6):  # group g on plates g % 3 and (g + 1) % 3: 3 pools of 2 groups
        rows += [(f"q{g}", g % 3), (f"q{g}", (g + 1) % 3)]
    meta = pd.DataFrame(rows, columns=["g", "plate"])
    meta["ref"] = np.where(meta["g"].str.startswith("c"), meta.index, -1)
    feats = np.random.default_rng(6).standard_normal((len(meta), 5))
    spec = dict(
        pos_sameby=["g", "ref"], pos_diffby=[], neg_sameby=["plate"],
        neg_diffby=["g", "ref"],
    )  # fmt: skip
    ap = average_precision(meta, feats, progress_bar=False, **spec)
    ap = ap[ap["ref"] < 0]
    args = (["g"], 300, 0.05, 0)
    built, workers = [], []

    class CountedPool(permutation._Pool):
        def __init__(self, blocks, similarity):
            built.append(len(blocks))
            super().__init__(blocks, similarity)

    class RecordedExecutor(permutation.ThreadPoolExecutor):
        def __init__(self, max_workers):
            workers.append(max_workers)
            super().__init__(max_workers)

    monkeypatch.setattr(permutation, "_Pool", CountedPool)
    monkeypatch.setattr(permutation, "ThreadPoolExecutor", RecordedExecutor)
    p = [
        mean_average_precision(ap, *args, progress_bar=False, max_workers=w)[
            "p_value"
        ].tolist()
        for w in (1, 4)
    ]
    assert built == [2] * 6  # each run builds each of the 3 pools once
    assert workers == [1, 4]
    alone = [
        mean_average_precision(ap[ap["g"] == g], *args, progress_bar=False)[
            "p_value"
        ].item()
        for g in sorted(ap["g"].unique())
    ]
    assert p[0] == p[1] == alone
    assert len(set(alone)) > 1
