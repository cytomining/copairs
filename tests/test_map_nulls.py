"""Tests for the choice of mAP null in copairs' mAP functions."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import copairs.map.map as map_module
from copairs import compute
from copairs.map import (
    get_map_pvalue,
    average_precision,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from tests.helpers import ap_frame, query_meta
from copairs.methods import NULLS
from tests.test_nulls import run_python


def test_copied_null_averages_configuration_nulls(tmp_path):
    """null="copied" averages each member's configuration null, as copairs 0.5.5."""
    rows = [("a", 0.4, 2, 30)] * 3 + [("b", 0.3, 4, 40), ("b", 0.5, 5, 41)]
    frame = ap_frame(rows)
    got = get_map_pvalue(
        frame, ["g"], 500, 0, progress_bar=False, cache_dir=tmp_path, null="copied"
    )
    confs, rev_ix = np.unique(
        frame[["n_pos_pairs", "n_total_pairs"]].values, axis=0, return_inverse=True
    )
    null_dists = compute.get_null_dists(
        confs, 500, seed=0, cache_dir=tmp_path / "ref", progress_bar=False
    )
    for g, p in zip(got["g"], got["p_value"]):
        rows = np.flatnonzero(frame["g"] == g)
        null = null_dists[rev_ix.ravel()[rows]].mean(axis=0, dtype=np.float64)
        score = frame["average_precision"].iloc[rows].mean()
        assert p == ((null >= score - 1e-6 * score).sum() + 1) / 501


def perfect_retrieval_scores():
    """Return two groups of two replicates (one positive each) among 100 profiles."""
    return ap_frame([("a", 1.0, 1, 100)] * 2 + [("b", 0.5, 1, 100)] * 2)


def test_map_pvalue_counts_ties(tmp_path):
    """An mAP equal to a null atom counts the atom.

    Under the copied null P(mAP >= 1) = P(AP >= 1) = 1/100. Under the joint
    null, given the shared similarity u each member's rank - 1 is
    Binomial(99, 1 - u): both APs are 1 with probability int u^198 = 1/199, and
    mAP >= 0.5 needs one rank of 1 or both of 2.
    """
    kwargs = dict(null_size=100_000, threshold=0.05, seed=0, progress_bar=False)
    both_two = 99**2 * 2 / (197 * 198 * 199)  # int (99 (1 - u) u^98)^2 du
    for null, expected in (
        ("copied", np.array([0.01, 0.02])),
        ("joint", np.array([1 / 199, 2 / 100 - 1 / 199 + both_two])),
    ):
        got = mean_average_precision(
            perfect_retrieval_scores(), ["g"], null=null, cache_dir=tmp_path, **kwargs
        )
        se = np.sqrt(expected * (1 - expected) / kwargs["null_size"])
        assert (np.abs(got["p_value"].to_numpy() - expected) < 5 * se).all()


@pytest.mark.parametrize("null", ["copied", "joint"])
def test_float32_scores_count_ties(null, tmp_path):
    """float32-rounded scores still tie with the null atoms they round from.

    With one positive among 3, AP is 1, 1/2 or 1/3, so every null value is
    ``>= 1/3`` and the p-value of an mAP of 1/3 is exactly 1. pandas 2 averages
    these float32 APs to float32 and pandas 3 to float64; ties hold under both,
    and the reported mAP keeps the APs' dtype.
    """
    scores = pd.DataFrame(
        {
            "g": ["a", "a"],
            "average_precision": np.float32([1 / 3, 1 / 3]),
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 3,
        }
    )
    got = mean_average_precision(
        scores, ["g"], 1000, 0.05, 0, progress_bar=False, null=null, cache_dir=tmp_path
    )
    assert got["p_value"].iloc[0] == 1.0
    assert got["mean_average_precision"].dtype == np.float32


@pytest.mark.parametrize("keys", [str, int])
def test_map_column_is_the_score(keys, tmp_path, monkeypatch):
    """The mAP column is the float64 mean in the APs' dtype, and the nulls' score.

    pandas averages float32 APs to float32 or float64 depending on its version
    and the keys' dtype (pandas 3 string keys give float64), and its float32
    mean can differ in the last bit from the rounded float64 mean.
    """
    rng = np.random.default_rng(0)
    g = rng.integers(0, 50, 400)
    frame = ap_frame(
        list(
            zip(
                g.astype(keys),
                rng.random(400).astype(np.float32),
                [3] * 400,
                [30] * 400,
            )
        )
    )
    frame["average_precision"] = frame["average_precision"].astype(np.float32)
    seen = []

    def copied(scores, *args, **kwargs):
        seen.append(scores)
        return np.ones(len(scores))

    monkeypatch.setattr(map_module, "_copied_pvalues", copied)
    got = get_map_pvalue(frame, ["g"], 10, 0, progress_bar=False, null="copied")
    mean = frame.astype({"average_precision": np.float64}).groupby("g")
    expected = mean["average_precision"].mean().astype(np.float32)
    column = got.set_index("g")["mean_average_precision"]
    assert column.dtype == np.float32
    np.testing.assert_array_equal(column.loc[expected.index], expected)
    assert seen[0].tobytes() == got["mean_average_precision"].to_numpy().tobytes()


def test_nulls_share_the_pair_count_check():
    """Positive pairs that miss the AP table's counts raise alike under both nulls."""
    meta = query_meta()
    feats = np.random.default_rng(0).standard_normal((len(meta), 10))
    ap, pairs = average_precision(
        meta, feats, ["g"], [], [], ["g"], progress_bar=False, return_pairs=True
    )
    short = dataclasses.replace(pairs, pos=pairs.pos[1:])
    kwargs = dict(null_size=10, threshold=0.05, seed=0, progress_bar=False)
    for null in ("joint", "permutation"):
        with pytest.raises(ValueError, match="2 positive pairs but 3 in the AP table"):
            mean_average_precision(ap, ["g"], null=null, pairs=short, **kwargs)


def test_unknown_null():
    """Unknown nulls, nulls legacy does not implement, and stray pos_pairs raise."""
    frame = ap_frame([("a", 1.0, 1, 100)] * 2)
    kwargs = dict(null_size=10, threshold=0.05, seed=0, progress_bar=False)
    with pytest.raises(ValueError, match="unknown null"):
        mean_average_precision(frame, ["g"], null="exact", **kwargs)
    for null in ("permutation", "joint"):
        with pytest.raises(ValueError, match="supports only null='copied'"):
            mean_average_precision(frame, ["g"], method="legacy", null=null, **kwargs)
    with pytest.raises(ValueError, match="joint null only"):
        mean_average_precision(
            frame, ["g"], null="copied", pos_pairs=np.array([[0, 1]]), **kwargs
        )


def test_hierarchical_takes_each_null():
    """The hierarchical entry point validates null= and passes it to the p-values."""
    rows = [(g, d) for g in "abc" for d in (1, 2) for _ in range(3)] + [("ctrl", 0)] * 8
    meta = pd.DataFrame(rows, columns=["g", "dose"])
    meta["ref"] = np.where(meta["g"] == "ctrl", meta.index, -1)
    feats = np.random.default_rng(5).standard_normal((len(meta), 6))
    feats[meta["g"].eq("a").to_numpy()] += 2.0
    keys = ["g", "dose", "ref"]
    ap = average_precision(meta, feats, keys, [], [], keys, progress_bar=False)
    ap = ap[ap["g"] != "ctrl"]
    kwargs = dict(null_size=300, threshold=0.05, seed=0, progress_bar=False)
    sameby = ["g", "dose"]
    p = {}
    for null in NULLS:
        flat = mean_average_precision(ap, sameby, null=null, **kwargs)
        hier = mean_average_precision_hierarchical(
            ap, sameby, hierarchical_by=["g"], null=null, **kwargs
        )
        assert hier["p_value"].tolist() == flat["p_value"].tolist()
        p[null] = flat["p_value"].tolist()
    assert p["permutation"] != p["joint"] != p["copied"]
    default = mean_average_precision_hierarchical(
        ap, sameby, hierarchical_by=["g"], **kwargs
    )
    assert default["p_value"].tolist() == p["permutation"]
    with pytest.raises(ValueError, match="unknown null"):
        mean_average_precision_hierarchical(
            ap, sameby, hierarchical_by=["g"], null="exact", **kwargs
        )


def plate_aps(groups=12, reps=3, ctrls=15, plates=3, seed=0):
    """APs of ``groups`` with a replicate on each plate, ranked against its controls."""
    rows = []
    for p in range(plates):
        rows += [(f"q{i}", p) for i in range(groups)] + [("ctrl", p)] * ctrls
    meta = pd.DataFrame(rows, columns=["g", "plate"])
    meta["ctrl"] = meta["g"] == "ctrl"
    meta["ref"] = np.where(meta["ctrl"], np.arange(len(meta)), -1)
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((len(meta), 8))
    signal = meta["g"].isin([f"q{i}" for i in range(0, groups, 3)]).to_numpy()
    feats[signal] += 0.8
    ap = average_precision(
        meta, feats, ["g", "ref"], [], ["plate"], ["ctrl"], progress_bar=False
    )
    return ap[~ap["ctrl"]]


@pytest.mark.parametrize("null", NULLS)
def test_backends_give_identical_pvalues(null):
    """Each null's mAP p-values are bitwise identical on NumPy and Numba."""
    pytest.importorskip("numba")
    ap = plate_aps()
    kwargs = dict(null_size=3000, threshold=0.05, seed=7, progress_bar=False)
    got = {
        b: mean_average_precision(ap, ["g"], null=null, backend=b, **kwargs)
        for b in ("numpy", "numba")
    }
    pd.testing.assert_frame_equal(got["numba"], got["numpy"])
    assert 0 < got["numpy"]["below_p"].sum() < len(got["numpy"])


def test_auto_backend_runs_numba_kernels(monkeypatch):
    """With Numba importable, backend="auto" runs the joint and permutation kernels."""
    pytest.importorskip("numba")
    from copairs.nulls import _numba

    calls = []
    for name in ("joint_nulls", "permutation_maps"):

        def spy(*args, _real=getattr(_numba, name), _name=name):
            calls.append(_name)
            return _real(*args)

        monkeypatch.setattr(_numba, name, spy)
    ap = plate_aps(groups=4)
    kwargs = dict(null_size=500, threshold=0.05, seed=0, progress_bar=False)
    for null in ("joint", "permutation"):
        auto = mean_average_precision(ap, ["g"], null=null, **kwargs)
        numpy = mean_average_precision(ap, ["g"], null=null, backend="numpy", **kwargs)
        pd.testing.assert_frame_equal(auto, numpy)
    assert set(calls) == {"joint_nulls", "permutation_maps"}
    calls.clear()
    mean_average_precision(ap, ["g"], null="permutation", backend="numpy", **kwargs)
    assert calls == []


# User threads run every null's Numba kernels at once on Numba's workqueue
# threading layer, which aborts the process (SIGABRT) when two threads launch
# parallel kernels concurrently.
_CONCURRENT_NULLS = """
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["NUMBA_NUM_THREADS"] = "2"
import threading
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import pandas as pd
from copairs import compute
from copairs.map import average_precision, mean_average_precision

rows = []
for p in range(3):
    rows += [(f"q{i}", p) for i in range(12)] + [("ctrl", p)] * 15
meta = pd.DataFrame(rows, columns=["g", "plate"])
meta["ctrl"] = meta["g"] == "ctrl"
meta["ref"] = np.where(meta["ctrl"], np.arange(len(meta)), -1)
feats = np.random.default_rng(0).standard_normal((len(meta), 8))
ap = average_precision(
    meta, feats, ["g", "ref"], [], ["plate"], ["ctrl"], progress_bar=False
)
ap = ap[~ap["ctrl"]]
confs = ap[["n_pos_pairs", "n_total_pairs"]].to_numpy()
kw = dict(threshold=0.05, seed=0, progress_bar=False, backend="numba")
calls = {
    "joint": lambda: mean_average_precision(ap, ["g"], 20000, null="joint", **kw),
    # 12 groups on 2 Numba threads: each group's kernel on its pool thread
    "permutation": lambda: mean_average_precision(ap, ["g"], 20000, **kw),
    # one group at a time: Numba's parallel kernel
    "permutation-1": lambda: mean_average_precision(
        ap, ["g"], 20000, max_workers=1, **kw
    ),
    "copied": lambda: mean_average_precision(ap, ["g"], 20000, null="copied", **kw),
    "ap": lambda: compute.p_values(
        ap["average_precision"].to_numpy(), confs, 20000, 0, progress_bar=False,
        backend="numba",
    ),
}


def pvalues(name):
    out = calls[name]()
    return np.asarray(out if name == "ap" else out["p_value"]).tolist()


expected = {name: pvalues(name) for name in calls}
names = [name for name in calls for _ in range(4)]
barrier = threading.Barrier(len(calls))


def run(name):
    barrier.wait(timeout=120)
    return pvalues(name)


for _ in range(3):
    with ThreadPoolExecutor(len(calls)) as ex:
        got = list(ex.map(run, names))
    assert got == [expected[name] for name in names]
print(numba.threading_layer())
"""


def test_concurrent_calls_on_the_workqueue_layer():
    """User threads may run the Numba nulls at once, on any threading layer.

    Numba's workqueue layer (its fallback without TBB or OpenMP) aborts the
    process when two threads launch parallel kernels concurrently, so they
    launch one at a time; p-values match those of calls run alone.
    """
    pytest.importorskip("numba")
    assert run_python(_CONCURRENT_NULLS).strip() == "workqueue"
