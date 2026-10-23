"""Tests for the counter-based random stream, the AP null sampler and p-values."""

import sys
import itertools
import subprocess
import tracemalloc
import importlib.util

import numpy as np
import pytest
from scipy.stats import chisquare

from copairs import nulls
from copairs.nulls import common, philox, backend, pvalues, sampler
from tests.helpers import brute_ap_pvalues, brute_group_nulls, brute_map_pvalues
from copairs.map.normalization import expected_ap

BACKENDS = nulls.available_backends()
HAS_NUMBA = importlib.util.find_spec("numba") is not None
CONFS = np.array(
    [[1, 1], [1, 7], [2, 2], [3, 10], [5, 20], [9, 49], [40, 45], [99, 1099], [3, 1000]]
)
# Configurations where the scalar sampler (and Numba) take the guided gap search.
GUIDED = [(2, 5000), (3, 3000), (8, 20000), (4, 70000)]


@pytest.mark.parametrize(
    "ctr,key,expected",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_known_answers(ctr, key, expected):
    """Philox4x32-10 matches the Random123 known-answer vectors."""
    words = philox.philox4x32(*map(np.uint64, ctr), *map(np.uint64, key))
    assert [int(w) for w in words] == list(expected)


def test_uniform53_range_and_resolution():
    """Uniforms lie in [0, 1) and carry 53 bits: all-ones words map just below 1."""
    u = np.uint64
    assert philox.uniform53(u(0), u(0)) == 0.0
    top = philox.uniform53(u(0xFFFFFFFF), u(0xFFFFFFFF))
    assert top < 1.0 and top == 1.0 - 2.0**-53


def scalar_config_key(seed, num_pos, total):
    """Philox key of one configuration, one block at a time."""
    u = np.uint64
    w = philox.philox4x32(
        u(num_pos),
        u(total),
        u(seed & 0xFFFFFFFF),
        u(seed >> 32),
        philox.SALT0,
        philox.SALT1,
    )
    return int(w[0]), int(w[1])


@pytest.mark.parametrize("seed", [0, 3, 2**33 + 7, 2**64 - 1])
def test_config_key_arrays_match_scalar(seed):
    """Vectorized configuration keys equal the scalar derivation."""
    rng = np.random.default_rng(seed % 1000)
    num_pos = rng.integers(1, 2**32 - 1, 500)
    total = rng.integers(1, 2**32 - 1, 500)
    k0, k1 = philox.config_key_arrays(seed, num_pos, total)
    expected = [scalar_config_key(seed, int(p), int(t)) for p, t in zip(num_pos, total)]
    np.testing.assert_array_equal(
        np.stack([k0, k1], axis=1), np.array(expected, dtype=np.uint64)
    )


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_out_of_range(seed):
    """Seeds outside [0, 2**64) raise."""
    with pytest.raises(ValueError, match="seed must be"):
        philox.config_key_arrays(seed, np.array([1]), np.array([2]))


def test_config_keys_reject_wide_configurations():
    """Configurations that do not fit a 32-bit counter word raise."""
    with pytest.raises(ValueError, match="< 2\\*\\*32"):
        philox.config_key_arrays(0, np.array([1]), np.array([2**32]))


def test_resolve_seed():
    """An int seed passes through; None draws fresh entropy in [0, 2**64)."""
    assert common.resolve_seed(7) == 7
    seeds = {common.resolve_seed(None) for _ in range(4)}
    assert len(seeds) == 4 and all(0 <= s < 2**64 for s in seeds)


@pytest.mark.parametrize("null_size", [-1, -100, 2.5])
def test_check_null_size_rejects_invalid(null_size):
    """Negative or fractional null sizes raise instead of giving invalid p-values."""
    with pytest.raises(ValueError, match="null_size"):
        common.check_null_size(null_size)


def test_check_null_size_accepts_integral_floats():
    """An integral float null size is accepted and returned as an int."""
    assert common.check_null_size(1000.0) == 1000
    assert isinstance(common.check_null_size(1000.0), int)


def test_chunks_cover_null_lazily():
    """Chunks tile [0, null_size) and are generated lazily."""
    assert list(common.chunks(10, 4)) == [(0, 4), (4, 4), (8, 2)]
    assert list(common.chunks(0, 4)) == []
    assert next(common.chunks(10**12, 1)) == (0, 1)  # a generator: no 1e12 list
    assert common.chunk_size(100, 10, 8 * 10 * 7) == 7
    assert common.chunk_size(100, 10, 1) == 1
    assert common.chunk_size(5, 1, 2**30) == 5


def test_tie_thresholds():
    """float64 scores lose TIE_TOL; float32 scores lose their own rounding too."""
    s64 = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(common.tie_thresholds(s64), s64 - common.TIE_TOL)
    s32 = np.array([0.5, 1.0], dtype=np.float32)
    tol32 = common.TIE_ULPS * float(np.finfo(np.float32).eps) * s32.astype(np.float64)
    np.testing.assert_array_equal(
        common.tie_thresholds(s32), s32.astype(np.float64) - tol32
    )
    ints = common.tie_thresholds(np.array([1, 2]))
    assert ints.dtype == np.float64
    np.testing.assert_array_equal(ints, np.array([1.0, 2.0]) - common.TIE_TOL)


def exact_null(num_pos, total):
    """Every AP value of a (num_pos, total) rank list, one per placement of positives."""
    ranks = np.array(list(itertools.combinations(range(1, total + 1), num_pos)))
    return (np.arange(1, num_pos + 1) / ranks).sum(axis=1) / num_pos


def scalar_nulls(confs, null_size, seed):
    """AP nulls from the plain-Python scalar sampler, the reference of every backend."""
    _, _, k0, k1 = sampler.null_plan(np.asarray(confs), seed)
    return np.array(
        [
            [
                sampler._ap_sample(int(p), int(t), j, k0[c], k1[c])
                for j in range(null_size)
            ]
            for c, (p, t) in enumerate(confs)
        ]
    )


def test_available_backends():
    """NumPy is always available; Numba exactly when it is installed, and first."""
    assert BACKENDS == (["numba", "numpy"] if HAS_NUMBA else ["numpy"])
    assert nulls.resolve_backend("auto") == BACKENDS[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_bitwise_identical(backend):
    """Every backend returns the scalar reference's float64 values bit for bit.

    The scalar sampler takes the guided gap search on the GUIDED
    configurations, and the NumPy backend its own bisection or the plain loop.
    float32 output is the float64 sample rounded once.
    """
    confs = np.concatenate([CONFS, GUIDED])
    ref = scalar_nulls(confs, 64, 3)
    out = nulls.ap_nulls(confs, 64, seed=3, backend=backend, dtype=np.float64)
    np.testing.assert_array_equal(out.view(np.uint64), ref.view(np.uint64))
    out32 = nulls.ap_nulls(confs, 64, seed=3, backend=backend)
    np.testing.assert_array_equal(out32, ref.astype(np.float32))


@pytest.mark.parametrize("num_pos,total", GUIDED)
def test_guided_gap_search_matches_loop(num_pos, total):
    """The scalar guided gap search returns Algorithm A's gaps."""
    assert total - num_pos > sampler.GUIDED_RATIO * num_pos * (num_pos + 1)
    rng = np.random.default_rng(num_pos)
    for u in np.concatenate([rng.random(200), [1e-300, 0.5, 1 - 2**-53]]):
        k = int(rng.integers(2, num_pos + 1))
        assert sampler._gap_guided(total, k, u) == sampler.gap_loop(total, k, u)


@pytest.mark.parametrize(
    "remaining,k", [(30, 2), (1000, 3), (5000, 2), (3000, 7), (70_000, 4), (500, 20)]
)
def test_numpy_bisection_matches_loop(remaining, k):
    """The NumPy bisection returns Algorithm A's gap, also for u on its products.

    A u equal to the product at some gap, or one ulp either side, is where the
    closed form cannot decide and the probe replays the exact product.
    """
    rng = np.random.default_rng(remaining + k)
    products, quot, top = [], (remaining - k) / remaining, remaining - k
    for gap in range(1, remaining - k + 1):
        products.append(quot)
        top -= 1
        quot = quot * (top / (remaining - gap))
    products = np.array(products)
    picks = products[rng.integers(len(products), size=300)]
    picks = picks[picks > 0]
    u = np.concatenate(
        [
            rng.random(300),
            picks,
            np.nextafter(picks, 0),
            np.nextafter(picks, 1),
            [2.0**-53, 1e-300, 0.5, 1 - 2**-53],
        ]
    )
    rem = np.full(len(u), remaining)
    expected = [sampler.gap_loop(remaining, k, x) for x in u]
    np.testing.assert_array_equal(sampler._gap_bisect_numpy(rem, k, u), expected)
    np.testing.assert_array_equal(sampler._gap_loop_numpy(rem, k, u), expected)


def test_numba_matches_numpy_at_scale():
    """Numba and NumPy agree bitwise over many samples of large configurations."""
    pytest.importorskip("numba")
    confs = np.array([[2, 5000], [5, 300], [3, 100_000], [30, 3000], [1, 10**6]])
    kwargs = dict(seed=2**63 + 5, start=2**40, dtype=np.float64)
    a = nulls.ap_nulls(confs, 20_000, backend="numpy", **kwargs)
    b = nulls.ap_nulls(confs, 20_000, backend="numba", **kwargs)
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize("backend", BACKENDS)
def test_chunks_and_prefixes(backend):
    """Sample j depends only on (seed, num_pos, total, j)."""
    whole = nulls.ap_nulls(CONFS, 3000, seed=11, backend=backend)
    parts = [
        nulls.ap_nulls(CONFS, size, seed=11, start=start, backend=backend)
        for start, size in [(0, 1000), (1000, 1500), (2500, 500)]
    ]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), whole)
    alone = nulls.ap_nulls(CONFS[[4]], 100, seed=11, backend=backend)
    np.testing.assert_array_equal(alone[0], whole[4, :100])
    other_seed = nulls.ap_nulls(CONFS, 100, seed=12, backend=backend)
    assert not np.array_equal(other_seed, whole[:, :100])


def test_dtype_must_be_float32_or_float64():
    """Other output dtypes raise."""
    with pytest.raises(ValueError, match="dtype"):
        nulls.ap_nulls(CONFS, 5, seed=5, dtype=np.float16)


@pytest.mark.parametrize("num_pos,total", [(1, 9), (3, 10), (5, 20), (2, 60)])
def test_matches_exact_distribution(num_pos, total):
    """Sample frequencies of each AP value match exact enumeration."""
    exact = exact_null(num_pos, total)
    atoms, weights = np.unique(np.round(exact, 9), return_counts=True)
    null = nulls.ap_nulls([[num_pos, total]], 200_000, seed=1, dtype=np.float64)[0]
    observed = np.bincount(
        np.searchsorted(atoms, np.round(null, 9)), minlength=len(atoms)
    )
    assert observed.sum() == len(null)
    expected = weights / weights.sum() * len(null)
    big = expected >= 5
    obs = np.append(observed[big], observed[~big].sum())
    exp = np.append(expected[big], expected[~big].sum())
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    assert chisquare(obs, exp).pvalue > 1e-3
    # The upper tail, which decides small p-values, in particular.
    q = np.quantile(exact, 0.99)
    tail = np.mean(exact >= q)
    se = np.sqrt(tail * (1 - tail) / len(null))
    assert abs(np.mean(null >= q - common.TIE_TOL) - tail) < 5 * se


@pytest.mark.parametrize("num_pos,total", [(10, 100), (99, 1099), (3, 5000)])
def test_mean_matches_expected_ap(num_pos, total):
    """The null mean matches the closed-form E[AP] within Monte Carlo error."""
    null = nulls.ap_nulls([[num_pos, total]], 100_000, seed=2, dtype=np.float64)[0]
    se = null.std() / np.sqrt(len(null))
    assert abs(null.mean() - expected_ap(num_pos, total - num_pos)) < 5 * se


def test_degenerate_configurations():
    """All-positive lists have AP 1; a single positive has AP 1/rank, rank uniform."""
    null = nulls.ap_nulls([[4, 4], [1, 1]], 100, seed=0)
    assert (null == 1).all()
    ap = nulls.ap_nulls([[1, 5]], 50_000, seed=0, dtype=np.float64)[0]
    ranks = np.rint(1 / ap).astype(int)
    np.testing.assert_allclose(1 / ranks, ap)
    freq = np.bincount(ranks, minlength=6)[1:] / len(ap)
    np.testing.assert_allclose(freq, 0.2, atol=0.01)


def test_invalid_configurations():
    """Configurations without positives, or with more positives than ranks, raise."""
    for confs in ([[0, 5]], [[6, 5]], [1, 5]):
        with pytest.raises(ValueError):
            nulls.ap_nulls(confs, 10, seed=0)


@pytest.mark.parametrize("confs", [[[1.9, 5.9]], [[2, 5.5]], [[np.nan, 5]]])
def test_ap_nulls_rejects_fractional_configurations(confs):
    """Counts are not truncated: (1.9, 5.9) is not a (1, 5) configuration."""
    with pytest.raises(ValueError, match="integer"):
        nulls.ap_nulls(confs, 10, seed=1)


def test_ap_nulls_accepts_integral_floats():
    """Integral floats, e.g. counts that went through a float column, are fine."""
    np.testing.assert_array_equal(
        nulls.ap_nulls([[2.0, 5.0]], 10, seed=1), nulls.ap_nulls([[2, 5]], 10, seed=1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "budget,null_size", [(pvalues.DEFAULT_BUDGET, 2000), (8 * 700, 2000), (8, 300)]
)
def test_ap_pvalues_match_brute_force(backend, budget, null_size):
    """Streamed AP p-values equal counting over the materialized null, ties included.

    A budget of 8 bytes streams one sample per chunk.
    """
    rng = np.random.default_rng(0)
    conf_ix = rng.integers(len(CONFS), size=300)
    scores = rng.random(300)
    # Include exact ties with null values.
    null = nulls.ap_nulls(CONFS, 50, seed=4, dtype=np.float64)
    scores[:50] = null[conf_ix[:50], np.arange(50)]
    got = nulls.ap_pvalues(
        scores, conf_ix, CONFS, null_size, seed=4, backend=backend, budget_bytes=budget
    )
    np.testing.assert_array_equal(
        got, brute_ap_pvalues(scores, conf_ix, CONFS, null_size, 4)
    )
    # Ties count: dropping the tolerance changes the tied scores' p-values.
    strict = (null[conf_ix[:50]] > scores[:50, None]).sum(axis=1)
    assert (got[:50] > (strict + 1) / (null_size + 1)).all()


def random_groups(rng, n_groups):
    """CSR (group -> configuration counts) with mixed and repeated configurations."""
    ptr, conf_ix, conf_cnt = [0], [], []
    for _ in range(n_groups):
        confs = rng.choice(len(CONFS), size=rng.integers(1, 4), replace=False)
        conf_ix.extend(sorted(confs))
        conf_cnt.extend(rng.integers(1, 5, size=len(confs)))
        ptr.append(len(conf_ix))
    return np.array(ptr), np.array(conf_ix), np.array(conf_cnt)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("budget", [pvalues.DEFAULT_BUDGET, 8 * len(CONFS) * 300])
def test_map_pvalues_match_brute_force(backend, budget):
    """Streamed mAP p-values equal counting over materialized group nulls."""
    rng = np.random.default_rng(1)
    ptr, conf_ix, conf_cnt = random_groups(rng, 40)
    group_null = brute_group_nulls(ptr, conf_ix, conf_cnt, CONFS, 1500, 9)
    map_scores = rng.random(40) * 0.6
    map_scores[:10] = group_null[:10, 7]  # exact ties
    got = nulls.map_pvalues(
        map_scores,
        ptr,
        conf_ix,
        conf_cnt,
        CONFS,
        1500,
        seed=9,
        backend=backend,
        budget_bytes=budget,
    )
    np.testing.assert_array_equal(
        got, brute_map_pvalues(map_scores, ptr, conf_ix, conf_cnt, CONFS, 1500, 9)
    )


def test_backend_validation():
    """Unknown backends raise ValueError (unknown methods: test_unknown_method)."""
    with pytest.raises(ValueError, match="unknown backend"):
        nulls.resolve_backend("tpu")
    with pytest.raises(ValueError, match="unknown backend"):
        nulls.ap_nulls(CONFS, 4, seed=0, backend="cuda")
    backend.check_method("legacy", "numba")  # valid names, whether installed or not


def test_ap_pvalues_validates_conf_ix():
    """Scores pointing outside confs raise instead of being dropped."""
    with pytest.raises(ValueError):
        nulls.ap_pvalues([0.0], [5], CONFS[:2], 10, seed=1)
    with pytest.raises(ValueError):
        nulls.ap_pvalues([0.0, 0.5], [0], CONFS[:2], 10, seed=1)


@pytest.mark.parametrize("ptr", [[0, 1, 1], [0, 0, 1], [1, 2], [0, 1]])
def test_map_pvalues_validates_groups(ptr):
    """Empty groups and inconsistent CSR pointers raise ValueError."""
    with pytest.raises(ValueError):
        nulls.map_pvalues(
            [0.1] * (len(ptr) - 1), ptr, [0, 1], [1, 1], CONFS[:2], 10, seed=1
        )


@pytest.mark.parametrize(
    "conf_ix,conf_cnt", [([0, 2], [1, 1]), ([-1, 0], [1, 1]), ([0, 1], [1])]
)
def test_map_pvalues_validates_members(conf_ix, conf_cnt):
    """Configuration indices outside confs, or counts per member, raise."""
    with pytest.raises(ValueError, match="conf_"):
        nulls.map_pvalues([0.1, 0.2], [0, 1, 2], conf_ix, conf_cnt, CONFS[:2], 10, 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "ptr,conf_cnt",
    [([0, 1, 2], [0, 1]), ([0, 2], [0, 0]), ([0, 2], [2, -1]), ([0, 1, 2], [-1, 1])],
)
def test_map_pvalues_validates_member_counts(backend, ptr, conf_cnt):
    """Negative member counts and groups without members raise ValueError.

    A group of no members would divide 0 by 0 and get the smallest p-value.
    """
    scores = [0.1] * (len(ptr) - 1)
    with pytest.raises(ValueError, match="conf_cnt"):
        nulls.map_pvalues(scores, ptr, [0, 1], conf_cnt, CONFS[:2], 10, 1, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_map_pvalues_zero_count_member(backend):
    """A member counted zero times does not enter its group's null."""
    args = ([0.3], [0, 2], [0, 1], [0, 3], CONFS[:2], 500, 1)
    got = nulls.map_pvalues(*args, backend=backend)
    np.testing.assert_array_equal(got, brute_map_pvalues(*args))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_scores", [1, 3])
def test_map_pvalues_validates_score_count(backend, n_scores):
    """One mAP score per group; the kernels would read past a shorter array."""
    with pytest.raises(ValueError, match="groups"):
        nulls.map_pvalues(
            [0.1] * n_scores, [0, 1, 2], [0, 1], [1, 1], CONFS[:2], 10, 1, backend
        )


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_sampler_seed_out_of_range(seed):
    """Seeds outside [0, 2**64) raise, whether or not the caller resolves them."""
    with pytest.raises(ValueError, match="seed must be"):
        nulls.ap_nulls(CONFS, 4, seed)
    with pytest.raises(ValueError, match="seed must be"):
        nulls.ap_pvalues([0.5], [0], CONFS, 4, seed)


@pytest.mark.parametrize("null_size", [-1, -100, 2.5])
def test_pvalues_reject_invalid_null_size(null_size):
    """Negative or fractional null sizes raise instead of giving invalid p-values."""
    with pytest.raises(ValueError, match="null_size"):
        nulls.ap_pvalues([0.1], [0], CONFS[:1], null_size, seed=0)
    with pytest.raises(ValueError, match="null_size"):
        nulls.map_pvalues([0.1], [0, 1], [0], [1], CONFS[:1], null_size, seed=0)


MEMORY_CONFS = np.array([[5, 505], [3, 50003], [1, 10**6], [8, 20000]])


@pytest.mark.parametrize("n_conf", [1, 4])
@pytest.mark.parametrize("budget", [2**18, 2**24])
@pytest.mark.parametrize("which", ["ap", "map"])
def test_numpy_pvalues_memory_within_budget(which, budget, n_conf, monkeypatch):
    """Peak memory of the NumPy p-values stays within 2.5x budget_bytes.

    The chunk of null samples, the sampler's temporaries (summed over its
    threads: 4 of the 16 offered fit the larger budget) and the counts' each
    fit the budget, and a chunk is freed before the next is sampled. Sampling a whole chunk at
    once peaked at 31x (AP) and 9.7x (mAP). With one configuration, the mAP
    counts' per-group rows are as large as the null chunk itself (3.1x before
    they shared its budget).
    """
    monkeypatch.setattr(sampler, "_numpy_workers", lambda: 16)
    confs = MEMORY_CONFS[2:3] if n_conf == 1 else MEMORY_CONFS
    ix = np.arange(len(confs))
    kwargs = dict(null_size=2**16, seed=0, backend="numpy", budget_bytes=budget)
    tracemalloc.start()
    try:
        if which == "ap":
            nulls.ap_pvalues(np.full(len(confs), 0.5), ix, confs, **kwargs)
        else:
            nulls.map_pvalues(
                [0.5], [0, len(confs)], ix, [1] * len(confs), confs, **kwargs
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 2.5 * budget, peak / budget


def test_unseeded_pvalues_are_fresh():
    """seed=None draws a fresh null per call."""
    a = nulls.ap_nulls(CONFS, 50, common.resolve_seed(None))
    b = nulls.ap_nulls(CONFS, 50, common.resolve_seed(None))
    assert not np.array_equal(a, b)


# Makes numba unimportable in a child interpreter, as if it were not installed.
_WITHOUT_NUMBA = """
import sys
import importlib.util
from importlib.abc import MetaPathFinder

BLOCKED = ("numba", "llvmlite")


class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)


_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: (
    None if name.split(".")[0] in BLOCKED else _find_spec(name, *args)
)
sys.meta_path.insert(0, Block())
"""


def run_python(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout."""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_default_path_runs_without_numba():
    """Without numba, copairs imports, "auto" runs on NumPy and "numba" names the extra."""
    out = run_python(
        _WITHOUT_NUMBA
        + """
import numpy as np
import pandas as pd
import copairs, copairs.map, copairs.compute
from copairs import nulls, compute
from copairs.map import mean_average_precision

assert "numba" not in sys.modules
assert nulls.available_backends() == ["numpy"], nulls.available_backends()
assert nulls.resolve_backend("auto") == "numpy"
try:
    nulls.ap_nulls([[2, 5]], 4, 0, backend="numba")
except ImportError as exc:
    assert "copairs[numba]" in str(exc), exc
else:
    raise AssertionError("backend='numba' ran without numba")
p = compute.p_values(np.array([0.5, 1.0]), np.array([[2, 10], [1, 4]]), 100, 0,
                     progress_bar=False)
frame = pd.DataFrame({"g": ["a", "a"], "average_precision": [0.5, 0.6],
                      "normalized_average_precision": 0.0, "n_pos_pairs": 1,
                      "n_total_pairs": 10})
m = mean_average_precision(frame, ["g"], 100, 0.05, 0, progress_bar=False)
assert "numba" not in sys.modules
print(list(p), list(m["p_value"]))
"""
    )
    assert out.strip()


def test_auto_falls_back_when_numba_fails_to_import(tmp_path):
    """An installed Numba that fails to import leaves "auto" on NumPy, warning once.

    ``backend="numba"`` still raises the import error.
    """
    fake = tmp_path / "numba"
    fake.mkdir()
    (fake / "__init__.py").write_text('raise ImportError("simulated broken numba")\n')
    out = run_python(
        f"""
import sys, warnings
sys.path.insert(0, {str(tmp_path)!r})
import numpy as np
import pandas as pd
from copairs import nulls, compute
from copairs.map import mean_average_precision

frame = pd.DataFrame({{"g": ["a", "a"], "average_precision": [0.5, 0.6],
                      "normalized_average_precision": 0.0, "n_pos_pairs": 1,
                      "n_total_pairs": 10}})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert nulls.available_backends() == ["numpy"], nulls.available_backends()
    assert nulls.resolve_backend("auto") == "numpy"
    p = compute.p_values(np.array([0.5, 1.0]), [[2, 10], [1, 4]], 100, 0,
                         progress_bar=False)
    m = mean_average_precision(frame, ["g"], 100, 0.05, 0, progress_bar=False)
shown = [str(w.message) for w in caught if w.category is RuntimeWarning]
assert len(shown) == 1 and "simulated broken numba" in shown[0], shown
expected = compute.p_values(np.array([0.5, 1.0]), [[2, 10], [1, 4]], 100, 0,
                            progress_bar=False, backend="numpy")
assert np.array_equal(p, expected)
try:
    nulls.ap_nulls([[2, 5]], 4, 0, backend="numba")
except ImportError as exc:
    assert "simulated broken numba" in str(exc), exc
else:
    raise AssertionError("backend='numba' ran with a broken numba")
print(list(p), list(m["p_value"]))
"""
    )
    assert out.strip()


def test_import_does_not_load_numba():
    """With numba installed, importing copairs and its null modules leaves it unloaded."""
    pytest.importorskip("numba")
    out = run_python(
        "import sys\nimport copairs, copairs.map, copairs.compute, copairs.nulls\n"
        "import copairs.map.multilabel, copairs.nulls.sampler, copairs.nulls.pvalues\n"
        "print('numba' in sys.modules)"
    )
    assert out.strip() == "False"
