"""Tests for the label-permutation mAP null (copairs.nulls.permutation)."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from copairs.map import average_precision, mean_average_precision
from copairs.nulls import permutation

SPEC = dict(pos_sameby=["g"], pos_diffby=[], neg_sameby=[], neg_diffby=["g"])


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


def draws_of(meta, feats, spec, null_size, seed=0):
    """Observed mAP and the kernel's draws for group "q"."""
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
    out = np.empty(null_size)
    count, identity, _ = permutation.permutation_null(
        *layout, pairs.similarity, null_size, seed, out=out
    )
    return ap["average_precision"].mean(), identity, count, out


@pytest.mark.parametrize(
    "plates, reps, ctrls, spec",
    [
        (1, 3, 5, SPEC),
        (2, 2, 4, dict(SPEC, neg_sameby=["plate"])),
        (2, 2, 3, dict(SPEC, pos_diffby=["plate"], neg_sameby=["plate"])),
    ],
    ids=["one-pool", "plate-blocks", "plate-blocks-diffby"],
)
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


def test_rejects_unsupported_layouts():
    """Missing pairs, positives outside the group and mismatched pairs raise."""
    meta = layout_meta(1, 3, 5)
    meta["moa"] = ["m"] * 3 + ["c0", "c1", "c2", "c3", "c4"]
    feats = np.random.default_rng(2).standard_normal((len(meta), 4))
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **SPEC
    )
    args = (["g"], 100, 0.05, 0)
    with pytest.raises(ValueError, match="needs pairs"):
        mean_average_precision(ap, *args, progress_bar=False, null="permutation")
    other = dict(SPEC, neg_diffby=["g", "plate"])
    _, mismatched = average_precision(
        meta.assign(plate=[0, 0, 1, 0, 0, 1, 1, 1]), feats, progress_bar=False,
        return_pairs=True, **other,
    )  # fmt: skip
    with pytest.raises(ValueError, match="in the AP table"):
        mean_average_precision(
            ap, *args, progress_bar=False, null="permutation", pairs=mismatched
        )
    spec = dict(pos_sameby=["moa"], pos_diffby=[], neg_sameby=[], neg_diffby=["moa"])
    ap, pairs = average_precision(
        meta, feats, progress_bar=False, return_pairs=True, **spec
    )
    meta_ap = ap.assign(half=[0, 0, 1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="positives outside the group"):
        mean_average_precision(
            meta_ap, ["moa", "half"], 100, 0.05, 0, progress_bar=False,
            null="permutation", pairs=pairs,
        )  # fmt: skip


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
