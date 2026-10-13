"""Tests for the label-permutation mAP null (copairs.nulls.permutation)."""

import json
import pickle
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from copairs.map import multilabel, average_precision, mean_average_precision
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
    bare = ap.copy()
    bare.attrs.clear()
    with pytest.raises(ValueError, match="needs the pairs"):
        mean_average_precision(bare, *args, progress_bar=False)
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


ML_SPEC = dict(
    pos_sameby=["target"], pos_diffby=[], neg_sameby=[], neg_diffby=["target"]
)


def label_meta():
    """Compounds with overlapping targets; c1 and c8 share two of them.

    Each compound also has a label of its own, so none is left without one
    when a label moves away; such labels make no pairs.
    """
    targets = [["A"], ["A", "B"], ["B"], ["A", "C"], ["C"], ["B", "C"], ["D"]]
    targets += [["D", "A"], ["A", "B"]]
    return pd.DataFrame({"target": [t + [f"own{i}"] for i, t in enumerate(targets)]})


def multilabel_ap(meta, feats, **kwargs):
    """Multilabel APs of ``meta`` under ML_SPEC."""
    return multilabel.average_precision(
        meta, feats, **ML_SPEC, multilabel_col="target", progress_bar=False,
        **kwargs,
    )  # fmt: skip


def moved_maps(meta, feats, label):
    """MAP of ``label`` after moving it to every same-size set of profiles."""
    holders = [i for i, t in enumerate(meta["target"]) if label in t]
    values = []
    for chosen in itertools.combinations(range(len(meta)), len(holders)):
        target = [
            [t for t in labels if t != label] + ([label] if i in chosen else [])
            for i, labels in enumerate(meta["target"])
        ]
        ap = multilabel_ap(meta.assign(target=target), feats)
        values.append(ap.loc[ap["target"] == label, "average_precision"].mean())
    return np.array(values)


def label_draws(meta, feats, label, null_size, seed=0):
    """Observed mAP of ``label`` and the kernel's draws for it."""
    ap, pairs = multilabel_ap(meta, feats, return_pairs=True)
    ap = ap[ap["target"] == label]
    layouts, comp_profiles, (nbr_ptr, nbr) = permutation.label_layouts(
        [np.arange(len(ap))],
        ap.index.to_numpy(),
        ap["target"].to_numpy(),
        ap["n_pos_pairs"].to_numpy().astype(np.int64),
        (ap["n_total_pairs"] - ap["n_pos_pairs"]).to_numpy().astype(np.int64),
        pairs,
    )
    pool = permutation._LabelPool(
        layouts[0][3], comp_profiles, nbr_ptr, nbr, pairs.similarity
    )
    out = np.empty(null_size)
    count, identity = permutation.label_permutation_null(
        *layouts[0], pool, null_size, seed, out=out
    )
    return ap["average_precision"].mean(), identity, count, out


@pytest.mark.parametrize("label", ["A", "B"])
def test_label_moves_match_multilabel_average_precision(label):
    """Draws follow the mAP of uniformly random label moves, as copairs computes them."""
    meta = label_meta()
    feats = np.random.default_rng(7).standard_normal((len(meta), 5))
    exact = moved_maps(meta, feats, label)
    observed, identity, count, draws = label_draws(meta, feats, label, 60_000)
    assert identity == pytest.approx(observed, abs=1e-12)
    values, freq = np.unique(np.round(exact, 9), return_counts=True)
    ix = np.searchsorted(values, np.round(draws, 9))
    assert (values[np.minimum(ix, len(values) - 1)] == np.round(draws, 9)).all()
    expected = freq / freq.sum() * len(draws)
    chi2 = ((np.bincount(ix, minlength=len(values)) - expected) ** 2 / expected).sum()
    assert stats.chi2.sf(chi2, len(values) - 1) > 1e-3
    assert count == (draws >= identity - 1e-10).sum()


def test_label_permutation_through_mean_average_precision():
    """Multilabel APs with their pairs get permutation p-values per label."""
    meta = label_meta()
    feats = np.random.default_rng(8).standard_normal((len(meta), 5))
    for method in ("fast", "legacy"):
        ap, pairs = multilabel_ap(meta, feats, method=method, return_pairs=True)
        res = mean_average_precision(
            ap, ["target"], 500, 0.05, 0, progress_bar=False,
            null="permutation", pairs=pairs,
        )  # fmt: skip
        assert set(res["target"]) == {"A", "B", "C", "D"}
        assert ((res["p_value"] > 0) & (res["p_value"] <= 1)).all()
    with pytest.raises(ValueError, match="one 'target' value"):
        mean_average_precision(
            ap.assign(g=0), ["g"], 100, 0.05, 0, progress_bar=False,
            null="permutation", pairs=pairs,
        )  # fmt: skip


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
    assert pairs.similarity._fn is None  # the kernel is dropped after each call
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
    legacy = mean_average_precision(
        pickle.loads(pickle.dumps(ap)), ["g"], 100, 0.05, 0, progress_bar=False,
        method="legacy",
    )  # fmt: skip
    assert len(legacy)


def test_default_null_on_multilabel_frames():
    """Multilabel AP frames carry their pairs too."""
    meta = label_meta()
    feats = np.random.default_rng(12).standard_normal((len(meta), 5))
    ap, pairs = multilabel_ap(meta, feats, return_pairs=True)
    default = mean_average_precision(ap, ["target"], 300, 0.05, 0, progress_bar=False)
    explicit = mean_average_precision(
        ap, ["target"], 300, 0.05, 0, progress_bar=False, null="permutation",
        pairs=pairs,
    )  # fmt: skip
    assert default["p_value"].tolist() == explicit["p_value"].tolist()
