"""Tests for the label-permutation mAP null (copairs.nulls.permutation)."""

import json
import time
import pickle
import itertools
import threading
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from copairs import nulls, compute
from copairs.map import multilabel, average_precision, mean_average_precision
from copairs.nulls import permutation, tie_thresholds
from tests.helpers import ML_SPEC, label_meta
from copairs.nulls.common import stream_key

SPEC = dict(pos_sameby=["g"], pos_diffby=[], neg_sameby=[], neg_diffby=["g"])
LAYOUTS = [
    (1, 3, 5, SPEC),
    (2, 2, 4, dict(SPEC, neg_sameby=["plate"])),
    (2, 2, 3, dict(SPEC, pos_diffby=["plate"], neg_sameby=["plate"])),
    (3, 3, 20, dict(SPEC, neg_sameby=["plate"])),
]
LAYOUT_IDS = ["one-pool", "plate-blocks", "plate-blocks-diffby", "three-plates"]
BACKENDS = nulls.available_backends()


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


def draws_of(meta, feats, spec, null_size, seed=0, backend="auto"):
    """Observed mAP and the null's draws for group "q"."""
    observed, layout, pairs = group_of_q(meta, feats, spec)
    out = np.empty(null_size)
    count, identity, _ = permutation.permutation_null(
        *layout, pairs.similarity, null_size, seed, out=out, backend=backend
    )
    return observed, identity, count, out


def map_scratch(n):
    """Scratch arrays of the scalar ``group_map`` for a group of ``n`` members."""
    return (
        np.empty(n, np.int64),
        np.empty(n, np.int64),
        np.empty(n, np.uint64),
        np.empty(n, np.int32),
        np.empty(n, np.float64),
    )


def permutation_similarity(feats):
    """Cosine ``pairs -> similarities`` of ``feats``, as average_precision's."""
    cosine = compute.get_similarity_fn("cosine", progress_bar=False)
    return lambda pairs: cosine(feats, np.asarray(pairs), 20000)


def scalar_draws(layout, start, size, k0, k1, label=False, kernels=None):
    """Return draws ``start, ...`` from the scalar functions, one at a time.

    ``label`` takes the label null's (``draw_pool``, ``label_group_map``).
    ``kernels`` replaces the Python ``(draw, group_map)``, e.g. by Numba's
    compiled ones, which run the same way.
    """
    n = layout.n
    pm = np.empty(n, dtype=np.int64)
    chosen = np.empty(int(np.diff(layout.bm_ptr).max()), dtype=np.int64)
    if label:
        draw, group_map = permutation._draw_pool, permutation._label_group_map
        scratch = label_scratch(n)
    else:
        draw, group_map = permutation._draw, permutation._group_map
        scratch = map_scratch(n)
    if kernels is not None:
        draw, group_map = kernels
    pms, maps = [], []
    for j in range(start, start + size):
        draw(j, *layout.draw_args(), k0, k1, pm, chosen)
        pms.append(pm.copy())
        maps.append(group_map(pm, *layout.map_args(), *scratch))
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
    layout = permutation._Layout(members, edges, block_of, pairs.similarity, pool)
    np_tables = permutation._NumpyTables(layout, draws)
    assert pool.ranked == (tables != "search")
    assert (np_tables.full_keys is not None) == tables.startswith("full")
    assert (np_tables.full_ranks is not None) == (tables == "full-ranks")
    assert (np_tables.full_before is not None) == (tables == "full")
    k0, k1 = permutation.group_key(3, members)
    start = 2**32 - 50  # draws on both sides of the counter's word boundary
    pm, maps = scalar_draws(layout, start, 300, k0, k1)
    got_pm = permutation._draws_np(layout, start, 300, k0, k1)
    np.testing.assert_array_equal(got_pm, pm)
    np.testing.assert_array_equal(permutation._group_maps_np(np_tables, got_pm), maps)
    identity = np.arange(layout.n)
    scalar_identity = scalar_identity_map(layout, identity)
    assert permutation._group_maps_np(np_tables, identity[None])[0] == scalar_identity
    assert scalar_identity == pytest.approx(observed, abs=1e-12)


@pytest.mark.parametrize("ranked", [True, False], ids=["ranked", "search"])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "tied"])
@pytest.mark.parametrize("plates, reps, ctrls, spec", LAYOUTS, ids=LAYOUT_IDS)
def test_numba_matches_numpy_draws(
    plates, reps, ctrls, spec, ties, ranked, monkeypatch
):
    """Numba's draws, mAPs and hits equal NumPy's, bit for bit.

    Pools up to RANK_POOL read rank tables, larger ones search sorted rows.
    """
    pytest.importorskip("numba")
    from copairs.nulls import _numba

    if not ranked:
        monkeypatch.setattr(permutation, "RANK_POOL", 0)
    meta = layout_meta(plates, reps, ctrls)
    feats = np.random.default_rng(plates + ctrls).standard_normal((len(meta), 6))
    if ties:  # rounding ties many similarities and zeroes some profiles (NaN)
        feats = np.round(feats)
    _, group, pairs = group_of_q(meta, feats, spec)
    members, edges, blocks, block_of = group
    pool = permutation._Pool(blocks, pairs.similarity)
    layout = permutation._Layout(members, edges, block_of, pairs.similarity, pool)
    np_tables = permutation._NumpyTables(layout, 300)
    assert pool.ranked == ranked
    k0, k1 = permutation.group_key(3, members)
    start = 2**32 - 50  # draws on both sides of the counter's word boundary
    pm = permutation._draws_np(layout, start, 300, k0, k1)
    expected = permutation._group_maps_np(np_tables, pm)
    kernels = (_numba._draw_nb, _numba._group_map_nb)
    got_pm, got_maps = scalar_draws(layout, start, 300, k0, k1, kernels=kernels)
    np.testing.assert_array_equal(got_pm, pm)
    np.testing.assert_array_equal(got_maps, expected)
    for parallel in (True, False):  # on Numba's threads or the caller's
        maps = _numba.permutation_maps(layout, start, 300, k0, k1, parallel)
        np.testing.assert_array_equal(maps, expected)
    identity = permutation._group_maps_np(np_tables, np.arange(layout.n)[None])[0]
    assert _numba.permutation_identity(layout) == identity
    out = {b: np.empty(2000) for b in ("numpy", "numba")}
    runs = {
        b: permutation.permutation_null(
            *group, pairs.similarity, 2000, 3, pool, out=out[b], backend=b
        )[:2]
        for b in out
    }
    assert runs["numba"] == runs["numpy"]  # hits and identity
    np.testing.assert_array_equal(out["numba"], out["numpy"])


def scalar_identity_map(layout, pm):
    """MAP of the observed labels from the scalar function."""
    return permutation._group_map(pm, *layout.map_args(), *map_scratch(layout.n))


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_draws_do_not_depend_on_chunks(backend, monkeypatch):
    """Draw j is the same whatever the chunk sizes of draws and pool tables."""
    meta = layout_meta(2, 2, 6)
    feats = np.random.default_rng(22).standard_normal((len(meta), 5))
    spec = dict(SPEC, neg_sameby=["plate"])
    full = draws_of(meta, feats, spec, 1000, backend=backend)
    monkeypatch.setattr(permutation, "_NUMBA_CHUNK", 77)  # caps every chunk
    monkeypatch.setattr(permutation, "_PAIR_CHUNK", 7)
    monkeypatch.setattr(permutation, "_CHUNK_ELEMENTS", 5)
    chunked = draws_of(meta, feats, spec, 1000, backend=backend)
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


@pytest.mark.parametrize("stop_after", [None, 10])
def test_calibrated_on_exchangeable_profiles(stop_after):
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
        ap, ["g"], 999, 0.05, 0, progress_bar=False, null="permutation",
        pairs=pairs, stop_after=stop_after,
    )  # fmt: skip
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


def test_dropped_members_say_why_and_what_to_do():
    """Dropping some of a group's members raises, naming the cause and the remedies."""
    meta = layout_meta(1, 3, 5)
    feats = np.random.default_rng(2).standard_normal((len(meta), 4))
    ap = average_precision(meta, feats, progress_bar=False, **SPEC)
    args = (["g"], 100, 0.05, 0)
    with pytest.raises(
        ValueError,
        match=r"1 group\(s\) .* profiles the AP scores no longer have.*"
        r"computed with them as positives.*recompute average_precision.*"
        r"null='joint' or null='copied'",
    ):
        mean_average_precision(ap.drop(index=[0]), *args, progress_bar=False)
    kept = ap[ap["g"] == "q"]  # whole groups (and controls) may be dropped
    assert len(mean_average_precision(kept, *args, progress_bar=False))


@pytest.mark.parametrize("distance", ["seuclidean", "mahalanobis"])
def test_rejects_batch_dependent_metrics(distance):
    """Metrics that fit their scale to each batch cannot be recomputed per pool."""
    meta = pd.DataFrame({"g": ["q", "q", "c0", "c1"]})
    feats = np.array([[0.0], [1.0], [10.0], [1000.0]])
    spec = dict(SPEC, distance=distance)
    ap = average_precision(meta, feats, progress_bar=False, **spec)
    args = (["g"], 100, 0.05, 0)
    with pytest.raises(
        ValueError, match=f"distance='{distance}'.*each batch.*null='joint'"
    ):
        mean_average_precision(ap, *args, progress_bar=False)
    for null in ("joint", "copied"):
        assert len(mean_average_precision(ap, *args, progress_bar=False, null=null))


def test_rejects_pools_that_break_pos_diffby():
    """Relabelling must keep pos_diffby's pairs; pools that could break them raise."""
    meta = pd.DataFrame(
        {"g": ["q"] * 3 + ["c0", "c1"], "plate": ["A", "B", "C", "A", "A"]}
    )
    feats = np.random.default_rng(5).standard_normal((len(meta), 4))
    spec = dict(SPEC, pos_diffby=["plate"])
    ap = average_precision(meta, feats, progress_bar=False, **spec)
    args = (["g"], 100, 0.05, 0)
    with pytest.raises(ValueError, match="pos_diffby.*null='joint'"):
        mean_average_precision(ap, *args, progress_bar=False)
    assert len(mean_average_precision(ap, *args, progress_bar=False, null="joint"))
    # One plate value per block, or a distinct value per profile, keeps the rules.
    distinct = meta.assign(plate=list("ABCDE"), well=range(5))
    cases = [(distinct, d) for d in (["plate"], ["plate", "well"], ["well"])]
    cases.append((meta.assign(plate="A", well=range(5)), ["well"]))
    plates, reps, ctrls, blocked = LAYOUTS[LAYOUT_IDS.index("plate-blocks-diffby")]
    frame = layout_meta(plates, reps, ctrls)  # each plate a block
    x = np.random.default_rng(6).standard_normal((len(frame), 4))
    ap = average_precision(frame, x, progress_bar=False, **blocked)
    assert len(mean_average_precision(ap, *args, progress_bar=False))
    for frame, diffby in cases:
        ap = average_precision(
            frame, feats, progress_bar=False, **dict(SPEC, pos_diffby=diffby)
        )
        assert len(mean_average_precision(ap, *args, progress_bar=False))


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


def test_whole_pool_pos_diffby_check_matches_member_split():
    """Checking whole pools decides as checking members and references apart.

    The label null passes its components whole. Random layouts on 1-3 pools,
    each column constant, distinct or repeating per pool, raise in both forms
    or in neither, and both outcomes occur.
    """
    rng = np.random.default_rng(5)

    def raises(layout, **kwargs):
        try:
            permutation.check_pos_diffby([layout], codes, **kwargs)
        except ValueError:  # the check's refusal is the outcome
            return True
        return False

    outcomes = []
    while len(outcomes) < 300:
        cut = rng.choice(np.arange(1, 12), rng.integers(0, 3), replace=False)
        pools = np.split(np.arange(12), np.sort(cut))
        codes = np.empty((12, 2), dtype=np.int64)
        for c, pool in itertools.product(range(2), pools):
            kind = rng.integers(3)
            if kind == 0:
                codes[pool, c] = rng.integers(3)
            elif kind == 1:
                codes[pool, c] = rng.choice(30, len(pool), replace=False)
            else:
                codes[pool, c] = rng.integers(3, size=len(pool))
        chosen = [pool[rng.random(len(pool)) < 0.5] for pool in pools]
        used = [b for b, m in enumerate(chosen) if len(m)]
        members = np.concatenate([chosen[b] for b in used])
        if len(members) < 2:
            continue
        block_of = np.repeat(np.arange(len(used)), [len(chosen[b]) for b in used])
        refs = [np.setdiff1d(pools[b], members) for b in used]
        split = raises((members, None, refs, block_of))
        whole = raises(
            (members, None, [pools[b] for b in used], block_of), whole_pools=True
        )
        assert split == whole
        outcomes.append(split)
    assert 0 < sum(outcomes) < len(outcomes)


def test_interleaved_pools_are_built_once(monkeypatch):
    """Groups ranking pools A, B, A build two pools, also when A and B share a prefix."""
    refs = {0: [10, 11, 12], 1: [10, 11, 13], 2: [10, 11, 12]}  # A, B, A
    groups = [np.array([2 * g, 2 * g + 1]) for g in refs]
    pos = np.array([[2 * g, 2 * g + 1] for g in refs])
    neg = np.array([[m, r] for g in refs for m in groups[g] for r in refs[g]])
    feats = np.random.default_rng(8).standard_normal((14, 5))
    similarity = permutation_similarity(feats)
    built = []

    class CountedPool(permutation._Pool):
        def __init__(self, blocks, *args):
            built.append([b.tolist() for b in blocks])
            super().__init__(blocks, *args)

    monkeypatch.setattr(permutation, "_Pool", CountedPool)
    args = (groups, np.arange(6), np.ones(6), np.full(6, 4), pos, neg, similarity)
    p = permutation.permutation_pvalues(*args, 200, 0)
    assert sorted(built) == [[[10, 11, 12]], [[10, 11, 13]]]
    monkeypatch.setattr(permutation, "_Pool", CountedPool.__base__)
    alone = [
        permutation.permutation_pvalues(
            [np.array([0, 1])], groups[g], np.ones(2), np.full(2, 4), pos[[g]],
            neg[neg[:, 0] // 2 == g], similarity, 200, 0,
        )[0]
        for g in refs
    ]  # fmt: skip
    assert p.tolist() == alone


def shared_pool_aps():
    """APs of 6 groups on 3 plates, two per pool, without the controls."""
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
    return ap[ap["ref"] < 0]


def test_pool_memory_is_bounded(monkeypatch):
    """Under POOL_BUDGET, a group with a new pool waits until others are freed."""
    ap = shared_pool_aps()
    args = (["g"], 300, 0.05, 0)
    expected = mean_average_precision(ap, *args, progress_bar=False, max_workers=1)
    lock, alive, peak = threading.Lock(), [0], [0]

    class SlowPool(permutation._Pool):
        def __init__(self, blocks, *args):
            with lock:
                alive[0] += 1
                peak[0] = max(peak[0], alive[0])
            time.sleep(0.2)  # let other threads reach their pools
            super().__init__(blocks, *args)

    release = permutation._SharedPools.release

    def counted_release(self, g):
        held = self.key[g] in self.pools
        release(self, g)
        if held and self.key[g] not in self.pools:
            with lock:
                alive[0] -= 1

    monkeypatch.setattr(permutation, "_Pool", SlowPool)
    monkeypatch.setattr(permutation._SharedPools, "release", counted_release)
    monkeypatch.setattr(permutation, "POOL_BUDGET", 1, raising=False)
    got = mean_average_precision(ap, *args, progress_bar=False, max_workers=4)
    assert peak[0] == 1 and alive[0] == 0  # one pool at a time, all freed
    assert got["p_value"].tolist() == expected["p_value"].tolist()


def test_overlapping_pools_slice_one_key_table(monkeypatch):
    """Distinct pools over shared references take their keys from one table."""
    rng = np.random.default_rng(9)
    rows = [(f"c{p}_{i}", p) for p in range(6) for i in range(5)]
    for g in range(10):  # each group on 3 of 6 plates: overlapping pools
        rows += [(f"q{g}", int(p)) for p in rng.choice(6, 3, replace=False)]
    meta = pd.DataFrame(rows, columns=["g", "plate"])
    meta["ref"] = np.where(meta["g"].str.startswith("c"), meta.index, -1)
    feats = rng.standard_normal((len(meta), 5))
    spec = dict(
        pos_sameby=["g", "ref"], pos_diffby=[], neg_sameby=["plate"],
        neg_diffby=["g", "ref"],
    )  # fmt: skip
    ap = average_precision(meta, feats, progress_bar=False, **spec)
    ap = ap[ap["ref"] < 0]
    args = (["g"], 300, 0.05, 0)
    monkeypatch.setattr(permutation, "KEY_TABLE_BUDGET", 0, raising=False)
    expected = mean_average_precision(ap, *args, progress_bar=False)
    monkeypatch.undo()
    sizes = []
    key_matrix = permutation._key_matrix

    def counted(similarity, profiles, *args):
        sizes.append(len(profiles))
        return key_matrix(similarity, profiles, *args)

    monkeypatch.setattr(permutation, "_key_matrix", counted)
    got = mean_average_precision(ap, *args, progress_bar=False)
    assert sizes == [30]  # one table over the 6 plates' controls
    assert got["p_value"].tolist() == expected["p_value"].tolist()
    _, layout, pairs = group_of_q(layout_meta(2, 2, 4), feats[:12], SPEC)
    blocks = layout[2]
    table = permutation._KeyTable(np.sort(np.concatenate(blocks)), pairs.similarity)
    np.testing.assert_array_equal(
        permutation._Pool(blocks, pairs.similarity, table).krr,
        key_matrix(pairs.similarity, np.concatenate(blocks)),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_pvalues_do_not_depend_on_workers(backend, monkeypatch):
    """Groups sharing pools give the same p-values on 1 or 4 threads, or alone.

    Under the Numba backend, given 2 Numba threads, one thread runs each
    group's draws on Numba's threads (the parallel kernel) and 4 threads
    run them on their own (see test_concurrent_calls_on_the_workqueue_layer
    for user threads).
    """
    ap = shared_pool_aps()
    args = (["g"], 300, 0.05, 0)
    built, workers, parallel, launched = [], [], [], []

    class CountedPool(permutation._Pool):
        def __init__(self, blocks, *args):
            built.append(len(blocks))
            super().__init__(blocks, *args)

    class RecordedExecutor(permutation.ThreadPoolExecutor):
        def __init__(self, max_workers):
            workers.append(max_workers)
            super().__init__(max_workers)

    monkeypatch.setattr(permutation, "_Pool", CountedPool)
    monkeypatch.setattr(permutation, "ThreadPoolExecutor", RecordedExecutor)
    if backend == "numba":
        from copairs.nulls import _numba

        maps, launch = _numba.permutation_maps, _numba._launch

        def recorded(*args):
            parallel.append(args[-1])
            return maps(*args)

        def recorded_launch(kernel, *args):
            launched.append(kernel is _numba._permutation_maps_parallel)
            return launch(kernel, *args)

        monkeypatch.setattr(_numba, "threads", lambda: 2)
        monkeypatch.setattr(_numba, "permutation_maps", recorded)
        monkeypatch.setattr(_numba, "_launch", recorded_launch)
    p, kernels = [], []
    for w in (1, 4):
        parallel.clear()
        launched.clear()
        p.append(
            mean_average_precision(
                ap, *args, progress_bar=False, max_workers=w, backend=backend
            )["p_value"].tolist()
        )
        kernels.append((set(parallel), set(launched)))
    assert built == [2] * 6  # each run builds each of the 3 pools once
    assert workers == [1, 4]
    if backend == "numba":  # (draws in parallel, parallel kernels launched)
        assert kernels == [({True}, {True}), ({False}, set())]
    alone = [
        mean_average_precision(
            ap[ap["g"] == g], *args, progress_bar=False, backend=backend
        )["p_value"].item()
        for g in sorted(ap["g"].unique())
    ]
    assert p[0] == p[1] == alone
    assert len(set(alone)) > 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stop_after", [None, 20])
def test_split_draws_do_not_change_pvalues(stop_after, backend, monkeypatch):
    """A group's draws run as several tasks, on any threads, with the same p-values.

    Chunks of 32 split each group's 300 draws into 10 tasks (rounds of 32
    with stop_after, which here stops after 1 to 9 rounds), on 1 or 4
    threads and with POOL_BUDGET = 1; the p-values match a NumPy run with
    the default chunks on one thread.
    """
    meta = plate_label_meta()
    feats = np.random.default_rng(15).standard_normal((len(meta), 5))
    frames = [
        (shared_pool_aps(), ["g"]),
        (multilabel_ap(meta, feats, PLATE_SPEC), ["target"]),
    ]

    def pvalues(workers, backend=backend):
        return [
            mean_average_precision(
                ap, by, 300, 0.05, 0, progress_bar=False, max_workers=workers,
                stop_after=stop_after, backend=backend,
            )["p_value"].tolist()
            for ap, by in frames
        ]  # fmt: skip

    expected = pvalues(1, "numpy")
    calls, maps_of = [], permutation._Run.maps_of

    def counted(self, start, size):
        calls.append(self.key)  # one key per group
        return maps_of(self, start, size)

    monkeypatch.setattr(permutation, "_NUMBA_CHUNK", 32)  # chunks and rounds
    monkeypatch.setattr(permutation._Run, "maps_of", counted)
    for workers, budget in ((1, 2**32), (4, 2**32), (4, 1)):
        monkeypatch.setattr(permutation, "POOL_BUDGET", budget)
        calls.clear()
        assert pvalues(workers) == expected
        tasks = Counter(calls)
        assert len(tasks) == 9  # 6 groups and 3 labels
        if stop_after is None:
            assert set(tasks.values()) == {10}
        else:
            assert min(tasks.values()) == 1 and max(tasks.values()) > 2


@pytest.mark.parametrize(
    "parallel", [False, True], ids=["pool-threads", "numba-threads"]
)
def test_large_groups_spread_their_numba_draws(parallel, monkeypatch):
    """On pool threads, a large group's Numba draws run as several tasks.

    The tasks are as large as NumPy's chunks (2**22 // 30**2 = 4660 draws
    for 30 members), so threads share them; on Numba's threads each launch
    keeps up to _NUMBA_CHUNK draws. Rounds of stop_after double up to
    _NUMBA_CHUNK either way and run in such tasks. P-values match NumPy's.
    """
    pytest.importorskip("numba")
    from copairs.nulls import _numba

    labels = ["big"] * 30 + ["s1"] * 3 + ["s2"] * 3 + [f"c{i}" for i in range(40)]
    meta = pd.DataFrame({"g": labels})
    feats = np.random.default_rng(16).standard_normal((len(meta), 5))
    ap = average_precision(meta, feats, progress_bar=False, **SPEC)
    args = (ap.query("n_pos_pairs > 0"), ["g"], 10000, 0.05, 0)
    kwargs = dict(progress_bar=False, max_workers=4)
    expected = mean_average_precision(*args, backend="numpy", **kwargs)
    calls, maps_of = Counter(), permutation._Run.maps_of

    def counted(self, start, size):
        calls[self.layout.n] += 1
        return maps_of(self, start, size)

    monkeypatch.setattr(permutation._Run, "maps_of", counted)
    # 3 groups on 4 threads run on their own if Numba has 2 threads.
    monkeypatch.setattr(_numba, "threads", lambda: 8 if parallel else 2)
    got = mean_average_precision(*args, backend="numba", **kwargs)
    assert got["p_value"].tolist() == expected["p_value"].tolist()
    assert calls == {30: 1 if parallel else 3, 3: 2}
    # 20000 draws that never stop: rounds of 256, 512, ..., 8192, then 3872;
    # off Numba's threads the round of 8192 runs as 4660 + 3532.
    kwargs["stop_after"] = 10**6
    args = (args[0], ["g"], 20000, 0.05, 0)
    expected = mean_average_precision(*args, backend="numpy", **kwargs)
    calls.clear()
    got = mean_average_precision(*args, backend="numba", **kwargs)
    assert got["p_value"].tolist() == expected["p_value"].tolist()
    assert calls == {30: 7 if parallel else 8, 3: 14}


PLATE_SPEC = dict(ML_SPEC, neg_sameby=["plate"])


def plate_label_meta():
    """Two plates of compounds with targets; a compound on each plate has none.

    Negatives share the plate, so each plate is a component of the negative
    graph (its target-free compound is negative to all others there).
    """
    targets = [["A"], ["A", "B"], ["B"], ["C"], []]
    targets += [["A", "C"], ["B"], ["A"], ["C", "B"], []]
    return pd.DataFrame(
        {
            "target": [t + [f"own{i}"] for i, t in enumerate(targets)],
            "plate": [0] * 5 + [1] * 5,
        }
    )


LABEL_DESIGNS = {
    "one-component": (label_meta, ML_SPEC, lambda meta: [list(range(len(meta)))]),
    "plates": (
        plate_label_meta,
        PLATE_SPEC,
        lambda meta: [list(np.flatnonzero(meta["plate"] == p)) for p in (0, 1)],
    ),
}


def multilabel_ap(meta, feats, spec=ML_SPEC, **kwargs):
    """Multilabel APs of ``meta`` under ``spec``."""
    return multilabel.average_precision(
        meta, feats, **spec, multilabel_col="target", progress_bar=False, **kwargs
    )


def moved_maps(meta, feats, label, spec, blocks):
    """MAP of ``label`` after moving it to every set of as many profiles per block."""
    holders = [i for i, t in enumerate(meta["target"]) if label in t]
    k = [len(set(holders) & set(b)) for b in blocks]
    values = []
    for choice in itertools.product(
        *(itertools.combinations(b, kb) for b, kb in zip(blocks, k))
    ):
        chosen = {i for part in choice for i in part}
        target = [
            [t for t in labels if t != label] + ([label] if i in chosen else [])
            for i, labels in enumerate(meta["target"])
        ]
        ap = multilabel_ap(meta.assign(target=target), feats, spec)
        values.append(ap.loc[ap["target"] == label, "average_precision"].mean())
    return np.array(values)


def label_group(meta, feats, label, spec=ML_SPEC):
    """Observed mAP, layout, pool and pairs of ``label``'s group."""
    ap, pairs = multilabel_ap(meta, feats, spec, return_pairs=True)
    ap = ap[ap["target"] == label]
    layouts, comp_profiles, (nbr_ptr, nbr) = permutation._label_layouts(
        [np.arange(len(ap))],
        ap.index.to_numpy(),
        ap["n_pos_pairs"].to_numpy().astype(np.int64),
        (ap["n_total_pairs"] - ap["n_pos_pairs"]).to_numpy().astype(np.int64),
        pairs,
    )
    pool = permutation._LabelPool(
        layouts[0][3], comp_profiles, nbr_ptr, nbr, pairs.similarity
    )
    return ap["average_precision"].mean(), layouts[0], pool, pairs


def label_draws(meta, feats, label, null_size, spec=ML_SPEC, seed=0, backend="numpy"):
    """Observed mAP of ``label`` and the null's draws for it."""
    observed, layout, pool, _ = label_group(meta, feats, label, spec)
    out = np.empty(null_size)
    run = permutation._label_run(*layout, pool, seed, backend=backend)
    count, draws = run.count(null_size, out=out)
    assert draws == null_size
    return observed, run.identity, count, out


def label_scratch(n):
    """Scratch arrays of the scalar ``label_group_map`` for ``n`` members."""
    return (
        np.empty(n, np.uint64),
        np.empty(n, np.int32),
        np.empty(n, np.bool_),
        np.empty(n, np.float64),
    )


@pytest.mark.parametrize("lone_table", [False, True], ids=["lone-search", "lone-table"])
@pytest.mark.parametrize("tables", ["ranked", "search"])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "tied"])
@pytest.mark.parametrize("design, label", [
    ("one-component", "A"), ("one-component", "B"), ("plates", "A"), ("plates", "C"),
])  # fmt: skip
def test_label_vectorized_matches_scalar_draws(
    design, label, ties, tables, lone_table, monkeypatch
):
    """The NumPy label moves and mAPs equal the scalar functions', bit for bit.

    Pools up to RANK_POOL read a rank table, larger ones search sorted rows;
    layouts that run many draws read lone partners' counts from a table.
    """
    if tables == "search":
        monkeypatch.setattr(permutation, "RANK_POOL", 0)
    make_meta, spec, _ = LABEL_DESIGNS[design]
    meta = make_meta()
    feats = np.random.default_rng(len(meta) + ord(label)).standard_normal(
        (len(meta), 6)
    )
    if ties:  # rounding ties many similarities and zeroes some profiles (NaN)
        feats = np.round(feats)
    observed, (members, edges, lone, comps, block_of, key), pool, _ = label_group(
        meta, feats, label, spec
    )
    assert pool.ranked == (tables == "ranked")
    assert len(comps) == (2 if design == "plates" else 1)
    draws = 10**6 if lone_table else 0
    layout = permutation._LabelLayout(members, edges, lone, block_of, pool)
    np_tables = permutation._LabelNumpyTables(layout, draws)
    assert (np_tables.lone_before is not None) == lone_table
    k0, k1 = permutation._label_key(3, members, key)
    start = 2**32 - 50  # draws on both sides of the counter's word boundary
    pm, maps = scalar_draws(layout, start, 300, k0, k1, label=True)
    got_pm = permutation._draws_pool_np(layout, start, 300, k0, k1)
    np.testing.assert_array_equal(got_pm, pm)
    np.testing.assert_array_equal(permutation._label_maps_np(np_tables, got_pm), maps)
    identity = layout.member_code
    scalar = permutation._label_group_map(
        identity, *layout.map_args(), *label_scratch(layout.n)
    )
    assert permutation._label_maps_np(np_tables, identity[None])[0] == scalar
    assert scalar == pytest.approx(observed, abs=1e-12)


@pytest.mark.parametrize("ranked", [True, False], ids=["ranked", "search"])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "tied"])
@pytest.mark.parametrize("design, label", [
    ("one-component", "A"), ("one-component", "B"), ("plates", "A"), ("plates", "C"),
])  # fmt: skip
def test_label_numba_matches_numpy_draws(design, label, ties, ranked, monkeypatch):
    """Numba's label moves, mAPs and hits equal NumPy's, bit for bit, also stopped.

    Pools up to RANK_POOL read a rank table, larger ones search sorted rows.
    A run that stops at its stop_after-th hit ran the full run's first draws.
    """
    pytest.importorskip("numba")
    from copairs.nulls import _numba

    if not ranked:
        monkeypatch.setattr(permutation, "RANK_POOL", 0)
    make_meta, spec, _ = LABEL_DESIGNS[design]
    meta = make_meta()
    feats = np.random.default_rng(len(meta) + ord(label)).standard_normal(
        (len(meta), 6)
    )
    if ties:  # rounding ties many similarities and zeroes some profiles (NaN)
        feats = np.round(feats)
    _, group, pool, _ = label_group(meta, feats, label, spec)
    members, edges, lone, _, block_of, key = group
    assert pool.ranked == ranked
    layout = permutation._LabelLayout(members, edges, lone, block_of, pool)
    np_tables = permutation._LabelNumpyTables(layout, 300)
    k0, k1 = permutation._label_key(3, members, key)
    start = 2**32 - 50  # draws on both sides of the counter's word boundary
    pm = permutation._draws_pool_np(layout, start, 300, k0, k1)
    expected = permutation._label_maps_np(np_tables, pm)
    kernels = (_numba._draw_pool_nb, _numba._label_group_map_nb)
    got_pm, got_maps = scalar_draws(
        layout, start, 300, k0, k1, label=True, kernels=kernels
    )
    np.testing.assert_array_equal(got_pm, pm)
    np.testing.assert_array_equal(got_maps, expected)
    for parallel in (True, False):  # on Numba's threads or the caller's
        maps = _numba.label_maps(layout, start, 300, k0, k1, parallel)
        np.testing.assert_array_equal(maps, expected)
    identity = permutation._label_maps_np(np_tables, layout.member_code[None])[0]
    assert _numba.label_identity(layout) == identity
    full = np.empty(2000)
    expected = permutation._label_run(*group, pool, 3).count(2000, out=full)
    for backend, parallel in (("numpy", True), ("numba", True), ("numba", False)):
        run = permutation._label_run(*group, pool, 3, 2000, backend, parallel)
        assert run.identity == identity
        out = np.full(2000, np.nan)
        assert run.count(2000, out=out) == expected
        np.testing.assert_array_equal(out, full)
        out[:] = np.nan
        count, draws = run.count(2000, 3, out)
        hits = np.flatnonzero(full >= run.thr)
        assert (count, draws) == ((3, hits[2] + 1) if len(hits) >= 3 else expected)
        np.testing.assert_array_equal(out[:draws], full[:draws])


@pytest.mark.parametrize("design, label", [
    ("one-component", "A"), ("one-component", "B"), ("plates", "A"), ("plates", "C"),
])  # fmt: skip
def test_label_moves_match_multilabel_average_precision(design, label):
    """Draws follow the mAP of uniformly random label moves, as copairs computes them."""
    make_meta, spec, blocks_of = LABEL_DESIGNS[design]
    meta = make_meta()
    feats = np.random.default_rng(7).standard_normal((len(meta), 5))
    exact = moved_maps(meta, feats, label, spec, blocks_of(meta))
    observed, identity, count, draws = label_draws(meta, feats, label, 60_000, spec)
    assert identity == pytest.approx(observed, abs=1e-12)
    values, freq = np.unique(np.round(exact, 9), return_counts=True)
    ix = np.searchsorted(values, np.round(draws, 9))
    assert (values[np.minimum(ix, len(values) - 1)] == np.round(draws, 9)).all()
    expected = freq / freq.sum() * len(draws)
    chi2 = ((np.bincount(ix, minlength=len(values)) - expected) ** 2 / expected).sum()
    assert stats.chi2.sf(chi2, len(values) - 1) > 1e-3
    assert count == (draws >= tie_thresholds(identity)).sum()


def test_label_permutation_through_mean_average_precision():
    """Multilabel APs with their pairs get permutation p-values per label."""
    meta = label_meta()
    feats = np.random.default_rng(8).standard_normal((len(meta), 5))
    ap, pairs = multilabel_ap(meta, feats, return_pairs=True)
    res = mean_average_precision(
        ap, ["target"], 500, 0.05, 0, progress_bar=False, null="permutation",
        pairs=pairs,
    )  # fmt: skip
    assert set(res["target"]) == {"A", "B", "C", "D"}
    assert ((res["p_value"] > 0) & (res["p_value"] <= 1)).all()
    # Labels come from the pairs: the frame needs no label column.
    renamed = mean_average_precision(
        ap.drop(columns="target").assign(g=ap["target"]), ["g"], 500, 0.05, 0,
        progress_bar=False,
    )  # fmt: skip
    assert renamed["p_value"].tolist() == res["p_value"].tolist()


def test_label_permutation_calibrated():
    """Labels assigned independently of exchangeable profiles get uniform p-values."""
    rng = np.random.default_rng(9)
    n_cpd, n_labels = 300, 150
    has = rng.random((n_cpd, n_labels)) < 0.03
    meta = pd.DataFrame(
        {
            "target": [
                [f"L{j}" for j in np.flatnonzero(h)] + [f"own{i}"]
                for i, h in enumerate(has)
            ]
        }
    )
    feats = rng.standard_normal((n_cpd, 4))
    ap, pairs = multilabel_ap(meta, feats, return_pairs=True)
    ap = ap[ap["target"].str.startswith("L")]
    res = mean_average_precision(
        ap, ["target"], 999, 0.05, 0, progress_bar=False, null="permutation",
        pairs=pairs,
    )  # fmt: skip
    p = res["p_value"].to_numpy()
    n = len(p)
    assert stats.binom(n, 0.05).ppf(0.0005) <= (p <= 0.05).sum()
    assert (p <= 0.05).sum() <= stats.binom(n, 0.05).ppf(0.9995)
    assert stats.kstest(p, "uniform").pvalue > 1e-3


def test_default_null_on_multilabel_frames():
    """Multilabel AP frames carry their pairs, so the default null needs no pairs=."""
    meta = label_meta()
    feats = np.random.default_rng(12).standard_normal((len(meta), 5))
    ap, pairs = multilabel_ap(meta, feats, return_pairs=True)
    default = mean_average_precision(ap, ["target"], 300, 0.05, 0, progress_bar=False)
    explicit = mean_average_precision(
        ap, ["target"], 300, 0.05, 0, progress_bar=False, null="permutation",
        pairs=pairs,
    )  # fmt: skip
    assert default["p_value"].tolist() == explicit["p_value"].tolist()
    kept = ap[ap["target"].isin(["A", "C"])].copy()  # whole labels may be dropped
    filtered = mean_average_precision(
        kept, ["target"], 300, 0.05, 0, progress_bar=False
    )
    assert filtered["p_value"].tolist() == explicit["p_value"].iloc[[0, 2]].tolist()


def test_label_guards():
    """Dropped rows, batch-dependent distances, pair counts and pos_diffby breaks raise."""
    meta = plate_label_meta()
    feats = np.random.default_rng(13).standard_normal((len(meta), 5))
    args = (["target"], 100, 0.05, 0)
    ap = multilabel_ap(meta, feats, PLATE_SPEC)
    by_plate = (["plate"], *args[1:])  # groups holding several labels
    with pytest.raises(
        ValueError, match="one 'target' value; group by it, or pass null='joint'"
    ):
        mean_average_precision(ap, *by_plate, progress_bar=False)
    assert len(mean_average_precision(ap, *by_plate, progress_bar=False, null="joint"))
    dropped = ap.drop(index=ap.index[ap["target"] == "A"][:1])
    split = ap.assign(half=np.arange(len(ap)) % 2)  # a label's rows grouped apart
    for part, by in ((dropped, args), (split, (["target", "half"], *args[1:]))):
        with pytest.raises(
            ValueError, match="no row for it in the group.*null='joint'"
        ):
            mean_average_precision(part, *by, progress_bar=False)
    assert len(mean_average_precision(dropped, *args, progress_bar=False, null="joint"))
    scaled = multilabel_ap(meta, feats, PLATE_SPEC, distance="seuclidean")
    with pytest.raises(ValueError, match="distance='seuclidean'.*null='joint'"):
        mean_average_precision(scaled, *args, progress_bar=False)
    with pytest.raises(ValueError, match="index must hold the row numbers"):
        mean_average_precision(
            ap.set_axis(ap.index.astype(str)), *args, progress_bar=False
        )
    with pytest.raises(ValueError, match="index reaches row"):
        mean_average_precision(ap.set_axis(ap.index + 100), *args, progress_bar=False)
    for column, kind in (("n_pos_pairs", "positive"), ("n_total_pairs", "negative")):
        miscounted = ap.copy()
        miscounted.loc[miscounted.index[0], column] += 1
        label = miscounted["target"].iloc[0]
        with pytest.raises(ValueError, match=f"{kind} pairs for '{label}' but"):
            mean_average_precision(miscounted, *args, progress_bar=False)
    # Positives must differ in "well"; one well on each plate is shared by
    # two compounds, so moved labels could pair profiles in one well.
    wells = meta.assign(well=[0, 1, 2, 3, 3, 0, 1, 2, 3, 3])
    spec = dict(PLATE_SPEC, pos_diffby=["well"])
    with pytest.raises(ValueError, match="pos_diffby.*null='joint'"):
        mean_average_precision(
            multilabel_ap(wells, feats, spec), *args, progress_bar=False
        )
    unique = meta.assign(well=range(10))  # a distinct well per profile keeps the rule
    assert len(
        mean_average_precision(
            multilabel_ap(unique, feats, spec), *args, progress_bar=False
        )
    )


@pytest.mark.parametrize(
    "spec, suggest_blocks", [(PLATE_SPEC, False), (ML_SPEC, True)],
    ids=["plates", "one-component"],
)  # fmt: skip
def test_label_pool_limit_names_other_nulls(spec, suggest_blocks, monkeypatch):
    """A label on more than MAX_POOL profiles raises, naming joint and copied.

    Under neg_sameby=["plate"] each plate is a component and label A is on
    both, so finer blocks cannot shrink its pool and the message does not
    suggest them; on one component it does.
    """
    meta = plate_label_meta()
    feats = np.random.default_rng(13).standard_normal((len(meta), 5))
    ap = multilabel_ap(meta, feats, spec)
    args = (["target"], 100, 0.05, 0)
    monkeypatch.setattr(permutation, "MAX_POOL", 6)
    match = "label 'A' .*: (split .* or )?pass null='joint' or null='copied'"
    with pytest.raises(ValueError, match=match) as err:
        mean_average_precision(ap, *args, progress_bar=False)
    assert ("neg_sameby" in str(err.value)) == suggest_blocks
    assert len(mean_average_precision(ap, *args, progress_bar=False, null="joint"))


def test_labels_with_one_holder_set_draw_independently():
    """Two labels held by the same profiles get different draws (keys by label)."""
    meta = pd.DataFrame(
        {
            "target": [["A", "B"], ["A", "B"], ["A", "B"]]
            + [[f"own{i}"] for i in range(7)]
        }
    )
    feats = np.random.default_rng(14).standard_normal((len(meta), 5))
    _, _, _, a = label_draws(meta, feats, "A", 200)
    _, _, _, b = label_draws(meta, feats, "B", 200)
    assert not np.array_equal(a, b)
    members = [2, 0, 1]
    assert permutation.group_key(5, members) == stream_key(5, (0, 1, 2))
    # The label key keeps #150's text, "(seed, members)/label".
    assert permutation._label_key(3, [12, 4, 7], 2) == (3205283342, 3195927061)
    assert permutation._label_key(3, [12, 4, 7], 2) == stream_key(3, (4, 7, 12), "/2")
    assert permutation._label_key(5, members, 0) != permutation._label_key(
        5, members, 1
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_label_pools_are_shared_and_workers_do_not_matter(backend, monkeypatch):
    """Labels on the same components share one pool; threads do not change p-values.

    Under the Numba backend, given 2 Numba threads, one thread runs each
    label's draws on Numba's threads (the parallel kernel) and 4 threads
    run the 3 labels' on their own.
    """
    meta = plate_label_meta()
    feats = np.random.default_rng(15).standard_normal((len(meta), 5))
    ap = multilabel_ap(meta, feats, PLATE_SPEC)
    args = (["target"], 300, 0.05, 0)
    built, parallel, launched = [], [], []

    class CountedPool(permutation._LabelPool):
        def __init__(self, comps, *args):
            built.append(comps)
            super().__init__(comps, *args)

    monkeypatch.setattr(permutation, "_LabelPool", CountedPool)
    if backend == "numba":
        from copairs.nulls import _numba

        maps, launch = _numba.label_maps, _numba._launch

        def recorded(*args):
            parallel.append(args[-1])
            return maps(*args)

        def recorded_launch(kernel, *args):
            launched.append(kernel is _numba._label_maps_parallel)
            return launch(kernel, *args)

        monkeypatch.setattr(_numba, "threads", lambda: 2)
        monkeypatch.setattr(_numba, "label_maps", recorded)
        monkeypatch.setattr(_numba, "_launch", recorded_launch)
    p, kernels = [], []
    for w in (1, 4):
        parallel.clear()
        launched.clear()
        p.append(
            mean_average_precision(
                ap, *args, progress_bar=False, max_workers=w, backend=backend
            )["p_value"].tolist()
        )
        kernels.append((set(parallel), set(launched)))
    assert p[0] == p[1]
    assert built == [(0, 5)] * 2  # one pool of both plates per call
    if backend == "numba":  # (draws in parallel, parallel kernels launched)
        assert kernels == [({True}, {True}), ({False}, set())]
    alone = [
        mean_average_precision(
            ap[ap["target"] == t], *args, progress_bar=False, backend=backend
        )["p_value"].item()
        for t in ("A", "B", "C")
    ]
    assert alone == p[0]
    _, layout, pool, pairs = label_group(meta, feats, "A", PLATE_SPEC)
    table = permutation._KeyTable(np.sort(pool.profiles), pairs.similarity)
    np.testing.assert_array_equal(
        CountedPool(layout[3], *label_pool_args(meta, feats), table).keys, pool.keys
    )


def label_pool_args(meta, feats):
    """``(comp_profiles, nbr_ptr, nbr, similarity)`` of PLATE_SPEC's label pools."""
    ap, pairs = multilabel_ap(meta, feats, PLATE_SPEC, return_pairs=True)
    _, comp_profiles, (nbr_ptr, nbr) = permutation._label_layouts(
        [], ap.index.to_numpy(), np.zeros(0), np.zeros(0), pairs
    )
    return comp_profiles, nbr_ptr, nbr, pairs.similarity


def sequential_expected(draws, identity, null_size, stop_after):
    """Besag-Clifford p-value from the full, ordered sequence of draws."""
    hits = np.flatnonzero(draws >= tie_thresholds(identity))
    if len(hits) >= stop_after:
        return stop_after / (hits[stop_after - 1] + 1)
    return (len(hits) + 1) / (null_size + 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stop_after", [1, 10, 400, 10**6])
def test_stop_after_follows_the_draw_sequence(stop_after, backend, monkeypatch):
    """Stopped runs give stop_after / (draw of that hit), from the same draws.

    The draws run are a prefix of the full run's (on NumPy), whatever the
    rounds; a run that never reaches stop_after hits keeps the fixed-size
    p-value.
    """
    null_size = 5000
    meta = layout_meta(2, 2, 6)
    feats = np.random.default_rng(13).standard_normal((len(meta), 5))
    spec = dict(SPEC, neg_sameby=["plate"])
    _, identity, _, full = draws_of(meta, feats, spec, null_size, backend="numpy")
    _, (members, edges, blocks, block_of), pairs = group_of_q(meta, feats, spec)
    pool = permutation._Pool(blocks, pairs.similarity)
    layout = permutation._Layout(members, edges, block_of, pairs.similarity, pool)
    run = permutation._Run(layout, permutation.group_key(0, members), backend)
    assert run.identity == identity
    out = np.full(null_size, np.nan)
    count, draws = run.count(null_size, stop_after, out)
    got = permutation._sequential_pvalue(count, draws, null_size, stop_after)
    assert got == sequential_expected(full, identity, null_size, stop_after)
    np.testing.assert_array_equal(out[:draws], full[:draws])
    if stop_after == 400:  # stops past the first round
        assert permutation._FIRST_ROUND < draws < null_size
    if stop_after == 10**6:
        assert draws == null_size
    ap = average_precision(meta, feats, progress_bar=False, **spec)
    ap = ap.query("n_pos_pairs > 0 and g == 'q'")
    args = (ap, ["g"], null_size, 0.05, 0)
    kwargs = dict(progress_bar=False, stop_after=stop_after, backend=backend)
    res = mean_average_precision(*args, **kwargs)
    assert res["p_value"].iloc[0] == got
    # Rounds of 7, 14, ..., 77 draws, in tasks of 5 off Numba's threads.
    monkeypatch.setattr(permutation, "_FIRST_ROUND", 7)
    monkeypatch.setattr(permutation, "_NUMBA_CHUNK", 77)
    monkeypatch.setattr(permutation, "_draw_chunk", lambda n: 5)
    rounds = mean_average_precision(*args, **kwargs)
    assert rounds["p_value"].iloc[0] == got


@pytest.mark.parametrize("hits_at", [[], [0], [255], [255, 767], [0, 255, 256, 1999]])
def test_count_hits_at_round_edges(hits_at):
    """Runs stop at the stop_after-th hit of the sequence, wherever rounds end.

    With the default chunk, rounds end after draws 255, 767, 1791 and 1999.
    """
    null_size = 2000
    seq = np.zeros(null_size)
    seq[hits_at] = 1.0
    hits = np.flatnonzero(seq)
    for chunk in (77, 256, 2**16):
        assert permutation._count_hits(
            lambda start, size: seq[start : start + size], null_size, 0.5, chunk
        ) == (len(hits), null_size)
        for stop_after in (1, 2, 3, 4, 5):
            got = permutation._count_hits(
                lambda start, size: seq[start : start + size], null_size, 0.5,
                chunk, stop_after,
            )  # fmt: skip
            if len(hits) >= stop_after:
                assert got == (stop_after, hits[stop_after - 1] + 1)
            else:
                assert got == (len(hits), null_size)


def test_split_rounds_count_in_draw_order():
    """A round's tasks may arrive in any order; hits are counted in draw order.

    Rounds of 256, 512, ... draws run as tasks of 7, which arrive here last
    first; the run still stops at the stop_after-th hit of the sequence.
    """
    null_size = 3000
    seq = (np.random.default_rng(17).random(null_size) < 0.01).astype(float)
    hits_at = np.flatnonzero(seq)
    for stop_after in (1, 5, 10, len(hits_at), len(hits_at) + 1):
        hits = permutation._Hits(null_size, 0.5, 7, stop_after)
        rounds = []
        while not hits.done:
            assert hits.round_done
            todo = hits.tasks()
            rounds.append(sum(size for _, size in todo))
            for start, size in reversed(todo):
                if hits.add(start, seq[start : start + size]):
                    break
        if stop_after <= len(hits_at):
            assert (hits.count, hits.draws) == (stop_after, hits_at[stop_after - 1] + 1)
        else:
            assert (hits.count, hits.draws) == (len(hits_at), null_size)
            assert rounds == [256, 512, 1024, 1208]
    assert len(hits_at) > 10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stop_after", [None, 3])
def test_draw_tasks_in_flight_are_bounded(stop_after, backend, monkeypatch):
    """Groups split into many tasks keep few in flight, with the same p-values.

    Tasks of 7 draws split each group's 3000 draws (or stop_after's rounds
    of 256, 512, ...) into hundreds; at most _TASKS_PER_WORKER per thread
    wait or run at once, and the tasks of a group that stopped mid-round
    are dropped.
    """
    ap = shared_pool_aps()
    args = (ap, ["g"], 3000, 0.05, 0)
    kwargs = dict(progress_bar=False, max_workers=2, stop_after=stop_after)
    expected = mean_average_precision(*args, backend="numpy", **kwargs)
    lock, pending, peak = threading.Lock(), set(), [0]

    def done(future):
        with lock:
            pending.discard(future)

    class CountingExecutor(permutation.ThreadPoolExecutor):
        def submit(self, fn, *args):
            future = super().submit(fn, *args)
            if getattr(fn, "__name__", "") == "maps_of":
                with lock:
                    pending.add(future)
                    peak[0] = max(peak[0], len(pending))
                future.add_done_callback(done)
            return future

    monkeypatch.setattr(permutation, "ThreadPoolExecutor", CountingExecutor)
    monkeypatch.setattr(permutation, "_draw_chunk", lambda n: 7)
    if backend == "numba":
        from copairs.nulls import _numba

        monkeypatch.setattr(_numba, "threads", lambda: 1)  # draws on pool threads
    got = mean_average_precision(*args, backend=backend, **kwargs)
    assert got["p_value"].tolist() == expected["p_value"].tolist()
    assert 2 < peak[0] <= permutation._TASKS_PER_WORKER * 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_stop_after_on_multilabel_groups(backend, monkeypatch):
    """Multilabel groups stop the same way, whatever the rounds.

    The p-values follow the sequence of NumPy's draws.
    """
    meta = label_meta()
    feats = np.random.default_rng(14).standard_normal((len(meta), 5))
    ap = multilabel_ap(meta, feats)
    args = (["target"], 3000, 0.05, 0)
    kwargs = dict(progress_bar=False, stop_after=10, backend=backend)
    res = mean_average_precision(ap, *args, **kwargs)
    for label in ("A", "B"):
        _, identity, count, full = label_draws(meta, feats, label, 3000)
        assert count >= 10  # the run stopped
        expected = sequential_expected(full, identity, 3000, 10)
        assert res.set_index("target").loc[label, "p_value"] == expected
    # Rounds of 7, 14, ... draws (at most 77), in tasks of 5 off Numba's
    # threads: every stopped label stops past its first round.
    monkeypatch.setattr(permutation, "_FIRST_ROUND", 7)
    monkeypatch.setattr(permutation, "_NUMBA_CHUNK", 77)
    monkeypatch.setattr(permutation, "_draw_chunk", lambda n: 5)
    for workers in (1, 4):
        rounds = mean_average_precision(ap, *args, max_workers=workers, **kwargs)
        assert rounds["p_value"].tolist() == res["p_value"].tolist()


def test_stop_after_arguments():
    """stop_after is a positive integer, for the permutation null only."""
    meta = layout_meta(1, 3, 6)
    feats = np.random.default_rng(15).standard_normal((len(meta), 5))
    ap = average_precision(meta, feats, progress_bar=False, **SPEC)
    args = (["g"], 100, 0.05, 0)
    for null in ("joint", "copied"):
        with pytest.raises(ValueError, match="permutation null only"):
            mean_average_precision(
                ap, *args, progress_bar=False, null=null, stop_after=10
            )
    meta = label_meta()
    feats = np.random.default_rng(15).standard_normal((len(meta), 5))
    labelled = multilabel_ap(meta, feats)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            mean_average_precision(ap, *args, progress_bar=False, stop_after=bad)
        with pytest.raises(ValueError, match="positive integer"):
            mean_average_precision(
                labelled, ["target"], *args[1:], progress_bar=False, stop_after=bad
            )
    default = mean_average_precision(ap, *args, progress_bar=False)
    explicit = mean_average_precision(ap, *args, progress_bar=False, stop_after=None)
    assert default["p_value"].tolist() == explicit["p_value"].tolist()
