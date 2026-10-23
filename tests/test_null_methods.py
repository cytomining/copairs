"""Tests for method= and backend= of copairs' public AP and mAP p-value functions."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import copairs.map.map as map_module
from copairs import nulls, compute
from copairs.map import (
    get_map_pvalue,
    mean_average_precision,
    mean_average_precision_hierarchical,
)
from copairs.nulls import sampler
from tests.helpers import brute_ap_pvalues, brute_map_pvalues
from copairs.map.normalization import expected_ap
from copairs.map.average_precision import p_values as frame_p_values

BACKENDS = nulls.available_backends()
CONFS = np.array([[1, 7], [3, 10], [5, 20], [9, 49], [99, 1099], [3, 1000]])


@pytest.fixture(autouse=True)
def no_home_cache(tmp_path, monkeypatch):
    """Keep legacy caches written without cache_dir out of ~/.copairs."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")


def random_ap_frame(rng, n=400, n_groups=60):
    """AP scores of ``n`` profiles in ``n_groups`` groups, near their expected AP."""
    frame = pd.DataFrame(
        {
            "g": rng.integers(n_groups, size=n).astype(str),
            "n_pos_pairs": rng.integers(3, 8, size=n),
        }
    )
    frame["n_total_pairs"] = frame["n_pos_pairs"] + rng.integers(40, 60, size=n)
    m, total = frame["n_pos_pairs"], frame["n_total_pairs"]
    mu = np.array([expected_ap(a, b - a) for a, b in zip(m, total)])
    frame["average_precision"] = np.clip(mu + rng.normal(0, 0.15, size=n), 0, 1)
    frame["normalized_average_precision"] = 0.0
    return frame


def test_legacy_method_is_the_seeded_0_5_5_null(tmp_path):
    """method="legacy" is copairs 0.5.5's null with per-configuration seeds.

    Each configuration's null is ``random_ap`` seeded by ``(seed, num_pos,
    total)``, cached under ``cache_dir``; the same configuration gets the same
    null whatever its position.
    """
    confs = np.array([[2, 10], [3, 30]])
    kwargs = dict(seed=0, cache_dir=tmp_path, progress_bar=False, method="legacy")
    got = compute.get_null_dists(confs, 100, **kwargs)
    for row, (num_pos, total) in zip(got, confs):
        seed = compute._conf_seed(0, num_pos, total)
        np.testing.assert_array_equal(row, compute.random_ap(100, num_pos, total, seed))
    assert len(list(tmp_path.rglob("*.npy"))) == 2
    flipped = compute.get_null_dists(confs[::-1], 100, **kwargs)
    np.testing.assert_array_equal(flipped, got[::-1])


def test_legacy_p_values_count_the_legacy_nulls():
    """Legacy AP and mAP p-values are counts over the legacy nulls (``>=`` / ``>``)."""
    rng = np.random.default_rng(5)
    frame = random_ap_frame(rng, n=60, n_groups=10)
    kwargs = dict(null_size=500, seed=3, progress_bar=False, method="legacy")
    confs, rev_ix = np.unique(
        frame[["n_pos_pairs", "n_total_pairs"]].values, axis=0, return_inverse=True
    )
    rev_ix = rev_ix.ravel()
    null = compute.get_null_dists(confs, 500, 3, progress_bar=False, method="legacy")
    ap = frame["average_precision"].to_numpy()
    num = (null[rev_ix] >= ap[:, None]).sum(axis=1)  # float64 comparison
    got = compute.p_values(ap, frame[["n_pos_pairs", "n_total_pairs"]].values, **kwargs)
    np.testing.assert_array_equal(got, ((num + 1) / 501).astype(np.float32))

    result = mean_average_precision(frame, ["g"], threshold=0.05, **kwargs)
    for _, row in result.iterrows():
        group_null = null[rev_ix[row["indices"]]].mean(axis=0)
        num = (group_null > row["mean_average_precision"]).sum()
        assert row["p_value"] == (num + 1) / 501


@pytest.mark.parametrize("backend", BACKENDS)
def test_fast_is_the_default(backend, tmp_path):
    """The default method samples with copairs.nulls and writes no cache."""
    nulls_fast = compute.get_null_dists(
        CONFS, 300, seed=4, cache_dir=tmp_path, progress_bar=False, backend=backend
    )
    np.testing.assert_array_equal(nulls_fast, nulls.ap_nulls(CONFS, 300, seed=4))
    frame = random_ap_frame(np.random.default_rng(0), n=50, n_groups=8)
    kwargs = dict(null_size=300, threshold=0.05, seed=4, progress_bar=False)
    default = mean_average_precision(frame, ["g"], cache_dir=tmp_path, **kwargs)
    fast = mean_average_precision(frame, ["g"], method="fast", **kwargs)
    pd.testing.assert_frame_equal(default, fast)
    assert list(tmp_path.rglob("*.npy")) == []
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_p_values_fast(backend):
    """compute.p_values with the fast method equals the brute-force count."""
    rng = np.random.default_rng(7)
    null_confs = CONFS[rng.integers(len(CONFS), size=200)]
    scores = rng.random(200)
    got = compute.p_values(
        scores, null_confs, 3000, seed=1, progress_bar=False, backend=backend
    )
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    expected = brute_ap_pvalues(scores, rev_ix.ravel(), confs, 3000, 1)
    np.testing.assert_array_equal(got, expected.astype(np.float32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_frame_p_values_fast(backend):
    """average_precision.p_values passes method and backend to compute.p_values."""
    rng = np.random.default_rng(8)
    frame = random_ap_frame(rng, n=40)
    frame.loc[:4, "n_pos_pairs"] = 0
    got = frame_p_values(frame, 1000, seed=2, progress_bar=False, backend=backend)
    assert np.isnan(got[:5]).all()
    valid = frame.iloc[5:]
    expected = compute.p_values(
        valid["average_precision"].to_numpy(),
        valid[["n_pos_pairs", "n_total_pairs"]].values,
        1000,
        2,
        progress_bar=False,
    )
    np.testing.assert_array_equal(got[5:], expected)
    legacy = frame_p_values(frame, 1000, seed=2, progress_bar=False, method="legacy")
    assert not np.array_equal(legacy[5:], got[5:])


@pytest.mark.parametrize("backend", BACKENDS)
def test_map_fast_matches_brute_force(backend):
    """Fast mAP p-values count group means of the exact AP nulls, ties as ``>=``."""
    rng = np.random.default_rng(2)
    frame = random_ap_frame(rng, n=200, n_groups=30)
    # A group whose mAP equals one of its null samples exactly.
    frame.loc[:2, "g"] = "tied"
    frame.loc[:2, ["n_pos_pairs", "n_total_pairs"]] = [3, 10]
    tied = nulls.ap_nulls([[3, 10]], 1, seed=6, dtype=np.float64)[0, 0]
    frame.loc[:2, "average_precision"] = tied
    result = mean_average_precision(
        frame, ["g"], 2000, 0.05, seed=6, progress_bar=False, backend=backend
    )
    confs, rev_ix = np.unique(
        frame[["n_pos_pairs", "n_total_pairs"]].values, axis=0, return_inverse=True
    )
    # Members sharing a configuration share samples.
    members = [
        np.unique(rev_ix.ravel()[ix], return_counts=True) for ix in result["indices"]
    ]
    ptr = np.cumsum([0] + [len(c) for c, _ in members])
    conf_ix = np.concatenate([c for c, _ in members])
    conf_cnt = np.concatenate([n for _, n in members])
    expected = brute_map_pvalues(
        result["mean_average_precision"].to_numpy(),
        ptr,
        conf_ix,
        conf_cnt,
        confs,
        2000,
        6,
    )
    np.testing.assert_array_equal(result["p_value"].to_numpy(), expected)
    tied_p = result.set_index("g").loc["tied", "p_value"]
    null = nulls.ap_nulls([[3, 10]], 2000, seed=6, dtype=np.float64)[0]
    assert tied_p > ((null > tied).sum() + 1) / 2001


def test_backends_identical_through_public_api():
    """NumPy and Numba give identical AP and mAP p-values."""
    pytest.importorskip("numba")
    frame = random_ap_frame(np.random.default_rng(4))
    kwargs = dict(null_size=5000, threshold=0.05, seed=1, progress_bar=False)
    a = mean_average_precision(frame, ["g"], backend="numpy", **kwargs)
    b = mean_average_precision(frame, ["g"], backend="numba", **kwargs)
    pd.testing.assert_frame_equal(a, b)
    confs = frame[["n_pos_pairs", "n_total_pairs"]].values
    ap = frame["average_precision"].to_numpy()
    np.testing.assert_array_equal(
        compute.p_values(ap, confs, 5000, 1, progress_bar=False, backend="numpy"),
        compute.p_values(ap, confs, 5000, 1, progress_bar=False, backend="numba"),
    )


def perfect_retrieval_scores():
    """Return two groups of two replicates (one positive each) among 100 profiles."""
    return pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "average_precision": [1.0, 1.0, 0.5, 0.5],
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 100,
        }
    )


def test_map_pvalue_counts_ties(tmp_path):
    """An mAP equal to a null atom counts the atom under the fast method.

    Both members of a group share one AP draw, so P(mAP >= 1) = P(AP >= 1) =
    1/100 and P(mAP >= 1/2) = 2/100. The legacy method counts ``>`` and gives
    mAP = 1 the smallest possible p-value.
    """
    kwargs = dict(null_size=100_000, threshold=0.05, seed=0, progress_bar=False)
    expected = np.array([0.01, 0.02])
    fast = mean_average_precision(perfect_retrieval_scores(), ["g"], **kwargs)
    se = np.sqrt(expected * (1 - expected) / kwargs["null_size"])
    assert (np.abs(fast["p_value"].to_numpy() - expected) < 5 * se).all()
    legacy = mean_average_precision(
        perfect_retrieval_scores(), ["g"], cache_dir=tmp_path, method="legacy", **kwargs
    )
    assert legacy["p_value"].iloc[0] == pytest.approx(1 / 100_001)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_scores_count_ties(backend):
    """float32-rounded scores still tie with the null atoms they round from.

    With one positive among 3, AP is 1, 1/2 or 1/3, so every null value is
    ``>= 1/3`` and the p-value of an AP or mAP of 1/3 is exactly 1, although
    float32(1/3) is above 1/3. pandas 2 averages these float32 APs to float32
    and pandas 3 to float64; ties hold under both, and the reported mAP keeps
    the APs' dtype.
    """
    kwargs = dict(null_size=1000, seed=0, progress_bar=False, backend=backend)
    for dtype in (np.float64, np.float32):
        p = compute.p_values(np.array([1 / 3], dtype=dtype), [[1, 3]], **kwargs)
        assert p[0] == 1.0
    scores = pd.DataFrame(
        {
            "g": ["a", "a"],
            "average_precision": np.float32([1 / 3, 1 / 3]),
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 1,
            "n_total_pairs": 3,
        }
    )
    fast = mean_average_precision(scores, ["g"], threshold=0.05, **kwargs)
    assert fast["p_value"].iloc[0] == 1.0
    assert fast["mean_average_precision"].dtype == np.float32


@pytest.mark.parametrize("keys", [str, int])
def test_map_column_is_the_score(keys, monkeypatch):
    """The mAP column is the float64 mean in the APs' dtype, and the nulls' score.

    pandas averages float32 APs to float32 or float64 depending on its version
    and the keys' dtype (pandas 3 string keys give float64), and its float32
    mean can differ in the last bit from the rounded float64 mean.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "g": rng.integers(0, 50, 400).astype(keys),
            "average_precision": rng.random(400).astype(np.float32),
            "normalized_average_precision": 0.0,
            "n_pos_pairs": 3,
            "n_total_pairs": 30,
        }
    )
    seen = []

    def copied(scores, *args, **kwargs):
        seen.append(scores)
        return np.ones(len(scores))

    monkeypatch.setattr(map_module, "_copied_pvalues", copied)
    got = get_map_pvalue(frame, ["g"], 10, 0, progress_bar=False)
    mean = frame.astype({"average_precision": np.float64}).groupby("g")
    expected = mean["average_precision"].mean().astype(np.float32)
    column = got.set_index("g")["mean_average_precision"]
    assert column.dtype == np.float32
    np.testing.assert_array_equal(column.loc[expected.index], expected)
    assert seen[0].tobytes() == got["mean_average_precision"].to_numpy().tobytes()


@pytest.mark.parametrize("backend", BACKENDS)
def test_max_workers_sets_numpy_threads(backend, monkeypatch):
    """max_workers sets the NumPy sampler's threads; p-values do not depend on it.

    Numba runs on its own threads, so its calls open no NumPy thread pool.
    """
    monkeypatch.setattr(sampler, "_numpy_workers", lambda: 16)
    pools = []

    def pool(n):
        pools.append(n)
        return ThreadPoolExecutor(n)

    monkeypatch.setattr(sampler, "ThreadPoolExecutor", pool)
    frame = random_ap_frame(np.random.default_rng(5))
    scores = frame["average_precision"].to_numpy()
    confs = frame[["n_pos_pairs", "n_total_pairs"]].to_numpy()
    kwargs = dict(null_size=2000, seed=0, progress_bar=False, backend=backend)
    runs = {
        "ap": lambda w: compute.p_values(scores, confs, max_workers=w, **kwargs),
        "map": lambda w: mean_average_precision(
            frame, ["g"], threshold=0.05, max_workers=w, **kwargs
        )["p_value"].to_numpy(),
    }
    for name, run in runs.items():
        got = {}
        for workers in (None, 1, 4):
            pools.clear()
            got[workers] = run(workers)
            if backend == "numba":
                assert pools == [], name
            else:  # one worker samples without a pool
                assert max(pools, default=1) == (workers or 16), (name, pools)
        for workers in (1, 4):
            assert got[workers].tobytes() == got[None].tobytes(), (name, workers)


@pytest.mark.parametrize("max_workers", [0, -1, 1.5])
def test_max_workers_validated(max_workers):
    """The fast method rejects worker counts ThreadPoolExecutor would."""
    with pytest.raises(ValueError, match="max_workers"):
        compute.p_values([0.5], [[2, 10]], 10, 0, max_workers=max_workers)
    with pytest.raises(ValueError, match="max_workers"):
        nulls.map_pvalues(
            [0.5], [0, 1], [0], [1], CONFS[:1], 10, 0, max_workers=max_workers
        )


def test_map_fast_vs_legacy(tmp_path):
    """Fast and legacy mAP p-values agree within Monte Carlo error."""
    frame = random_ap_frame(np.random.default_rng(3))
    kwargs = dict(null_size=20_000, threshold=0.05, seed=0, progress_bar=False)
    fast = mean_average_precision(frame, ["g"], **kwargs)
    legacy = mean_average_precision(
        frame, ["g"], cache_dir=tmp_path, method="legacy", **kwargs
    )
    p_fast, p_legacy = fast["p_value"].to_numpy(), legacy["p_value"].to_numpy()
    se = np.sqrt(p_legacy * (1 - p_legacy) / 20_000 * 2) + 1e-4
    assert (np.abs(p_fast - p_legacy) < 6 * se).all()
    assert (p_fast != p_legacy).any()  # different streams
    np.testing.assert_array_equal(
        fast["mean_average_precision"], legacy["mean_average_precision"]
    )


def test_ap_fast_vs_legacy():
    """Fast and legacy AP p-values agree within Monte Carlo error."""
    rng = np.random.default_rng(9)
    null_confs = CONFS[rng.integers(len(CONFS), size=300)]
    mu = np.array([expected_ap(p, t - p) for p, t in null_confs])
    scores = np.clip(mu + rng.normal(0, 0.1, size=300), 0, 1)
    kwargs = dict(null_size=20_000, seed=0, progress_bar=False)
    fast = compute.p_values(scores, null_confs, **kwargs)
    legacy = compute.p_values(scores, null_confs, method="legacy", **kwargs)
    se = np.sqrt(legacy * (1 - legacy) / 20_000 * 2) + 1e-4
    assert (np.abs(fast - legacy) < 6 * se).all()
    assert (fast != legacy).any()


def test_map_fast_independent_of_other_groups():
    """A group's fast mAP p-value does not depend on the other groups."""
    frame = random_ap_frame(np.random.default_rng(6), n=80, n_groups=6)
    kwargs = dict(sameby=["g"], null_size=500, threshold=0.05, seed=0)
    both = mean_average_precision(frame, progress_bar=False, **kwargs)
    only = frame[frame["g"] == both["g"].iloc[0]]
    alone = mean_average_precision(only, progress_bar=False, **kwargs)
    assert alone["p_value"].iloc[0] == both["p_value"].iloc[0]


def test_configurations_without_positives(tmp_path):
    """Configurations with no positives get NaN nulls and p = 1, under both methods.

    Callers such as the paper's simulations pass every ranked profile's
    configuration, controls (0 positives) included.
    """
    confs = np.array([[0, 10], [2, 10], [0, 0], [3, 3]])
    kwargs = dict(seed=0, progress_bar=False)
    legacy = compute.get_null_dists(
        confs, 50, method="legacy", cache_dir=tmp_path, **kwargs
    )
    fast = compute.get_null_dists(confs, 50, **kwargs)
    assert fast.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(fast), np.isnan(legacy))
    assert np.isnan(fast[[0, 2]]).all()
    np.testing.assert_array_equal(
        fast[[1, 3]], compute.get_null_dists(confs[[1, 3]], 50, **kwargs)
    )
    scores = np.array([0.3, 0.5, 0.0, 1.0])
    for method in ("fast", "legacy"):
        p = compute.p_values(scores, confs, 50, method=method, **kwargs)
        assert p[0] == p[2] == 1.0 and p[3] == 1.0 and p[1] < 1
    only_controls = compute.p_values(scores[[0, 2]], confs[[0, 2]], 50, **kwargs)
    np.testing.assert_array_equal(only_controls, [1.0, 1.0])


def empty_ap_frame():
    """AP scores of no profiles, with the usual columns."""
    return pd.DataFrame(
        {
            "g": pd.Series([], dtype=str),
            "h": pd.Series([], dtype=str),
            "average_precision": pd.Series([], dtype=float),
            "normalized_average_precision": pd.Series([], dtype=float),
            "n_pos_pairs": pd.Series([], dtype=int),
            "n_total_pairs": pd.Series([], dtype=int),
        }
    )


@pytest.mark.parametrize("method", ["fast", "legacy"])
@pytest.mark.parametrize("hierarchical", [False, True])
def test_map_empty_input(method, hierarchical, tmp_path):
    """No valid AP scores give an empty result with the usual columns."""
    empty = empty_ap_frame()
    kwargs = dict(progress_bar=False, method=method, cache_dir=tmp_path)
    if hierarchical:
        result = mean_average_precision_hierarchical(
            empty, ["g", "h"], 100, 0.05, 0, ["g"], **kwargs
        )
    else:
        result = mean_average_precision(empty, ["g"], 100, 0.05, 0, **kwargs)
    assert len(result) == 0
    assert {"p_value", "corrected_p_value", "below_corrected_p"} <= set(result.columns)


@pytest.mark.parametrize("method", ["fast", "legacy"])
def test_p_values_empty(method):
    """No AP scores give no p-values under either method."""
    pvals = compute.p_values(
        np.array([]), np.empty((0, 2), dtype=int), 100, 0, False, method=method
    )
    assert len(pvals) == 0
    nulls_ = compute.get_null_dists(np.empty((0, 2), dtype=int), 100, 0, method=method)
    assert nulls_.shape == (0, 100)


@pytest.mark.parametrize(
    "confs", [[2, 10], [[1, 5, 1], [2, 7, 3]], [[1, 5, 1, 9], [2, 7, 3, 9]], [[[1, 5]]]]
)
def test_null_dists_need_conf_rows(confs, tmp_path):
    """Configurations that are not (n, 2) rows raise rather than being reshaped.

    The legacy method fails on them too, while unpacking a row.
    """
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        compute.get_null_dists(confs, 10, 0)
    with pytest.raises((TypeError, ValueError)):
        compute.get_null_dists(confs, 10, 0, tmp_path, False, method="legacy")


@pytest.mark.parametrize("method", ["fast", "legacy"])
def test_null_dists_of_no_confs(method, tmp_path):
    """An empty list of configurations gives no nulls under either method."""
    got = compute.get_null_dists([], 10, 0, tmp_path, False, method=method)
    assert got.shape == (0, 10)


@pytest.mark.parametrize(
    "bad,match", [(dict(seed=-1), "seed must be"), (dict(null_size=-1), "null_size")]
)
def test_empty_input_still_validates(bad, match):
    """The fast method checks the seed and null size of inputs with nothing to rank."""
    kwargs = dict(null_size=100, seed=0, progress_bar=False) | bad
    for confs in (np.empty((0, 2), dtype=int), np.array([[0, 10]])):
        with pytest.raises(ValueError, match=match):
            compute.p_values(np.zeros(len(confs)), confs, **kwargs)
    with pytest.raises(ValueError, match=match):
        mean_average_precision(empty_ap_frame(), ["g"], threshold=0.05, **kwargs)


def test_unknown_method():
    """Unknown null methods raise in every entry point."""
    scores = perfect_retrieval_scores()
    with pytest.raises(ValueError, match="unknown method"):
        compute.get_null_dists(CONFS, 10, 0, method="exact")
    with pytest.raises(ValueError, match="unknown method"):
        compute.p_values(np.full(len(CONFS), 0.5), CONFS, 10, 0, method="exact")
    with pytest.raises(ValueError, match="unknown method"):
        frame_p_values(scores, 10, 0, method="exact")
    with pytest.raises(ValueError, match="unknown method"):
        mean_average_precision(scores, ["g"], 10, 0.05, 0, method="exact")


@pytest.mark.parametrize("method", ["fast", "legacy"])
def test_unknown_backend(method, tmp_path):
    """A misspelled backend raises under either method, though legacy ignores it."""
    scores = np.full(len(CONFS), 0.5)
    with pytest.raises(ValueError, match="unknown backend"):
        compute.p_values(scores, CONFS, 10, 0, method=method, backend="nmuba")
    with pytest.raises(ValueError, match="unknown backend"):
        compute.get_null_dists(
            CONFS, 10, 0, cache_dir=tmp_path, method=method, backend="nmuba"
        )
    with pytest.raises(ValueError, match="unknown backend"):
        mean_average_precision(
            perfect_retrieval_scores(), ["g"], 10, 0.05, 0, method=method, backend="x"
        )


@pytest.mark.skipif("numba" in BACKENDS, reason="numba is installed")
def test_numba_backend_without_numba_names_the_extra():
    """backend="numba" without numba raises an ImportError naming the extra."""
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        compute.p_values(np.array([0.5]), [[2, 10]], 10, 0, backend="numba")
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        compute.p_values(np.array([]), np.empty((0, 2)), 10, 0, backend="numba")
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        mean_average_precision(
            perfect_retrieval_scores(), ["g"], 10, 0.05, 0, backend="numba"
        )
    with pytest.raises(ImportError, match=r"copairs\[numba\]"):
        mean_average_precision(empty_ap_frame(), ["g"], 10, 0.05, 0, backend="numba")
