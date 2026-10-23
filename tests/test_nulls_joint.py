"""Tests for the joint mAP null (copairs.nulls.joint)."""

import math

import numpy as np
import pytest
from scipy import stats

from copairs import compute
from copairs.nulls import joint, common, permutation

COMPLETE4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
SIGNATURES = [
    ([(3, 100)] * 4, COMPLETE4),
    ([(3, 50), (3, 60), (3, 50), (3, 60)], [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ([(1, 30), (2, 40), (2, 40), (1, 30)], [(0, 1), (1, 2), (2, 3)]),
    ([(3, 100)], []),
    ([(1, 0), (1, 0)], [(0, 1)]),
    # BTRS binomials, and more positives than the insertion sort handles
    ([(40, 5000), (40, 3000)], [(0, 1)]),
    ([(5, 20_000)] * 3, [(0, 1), (0, 2), (1, 2)]),
]


def null_of(members, edges, size, seed=0, start=0):
    """Joint null samples of one group."""
    plan = joint.joint_plan([joint.canonical_signature(members, edges)], seed)
    return joint.joint_nulls(plan, size, start)[0]


def random_map(members, edges, null_size, seed):
    """#151's joint sampler, the reference the module reproduces in distribution.

    Draws a group's edge uniforms and each member's private positives, then drops
    the member's negatives into the gaps between its sorted positives with one
    multinomial draw.
    """
    rng = np.random.default_rng(seed)
    edges = sorted({(min(i, j), max(i, j)) for i, j in edges})
    shared = rng.random((null_size, len(edges)))
    total = np.zeros(null_size)
    for i, (num_pos, num_neg) in enumerate(members):
        mine = [e for e, pair in enumerate(edges) if i in pair]
        private = rng.random((null_size, num_pos - len(mine)))
        vals = -np.sort(-np.concatenate([shared[:, mine], private], axis=1), axis=1)
        ends = np.ones((null_size, 1)), np.zeros((null_size, 1))
        gaps = -np.diff(np.concatenate([ends[0], vals, ends[1]], axis=1), axis=1)
        above = np.cumsum(rng.multinomial(num_neg, gaps), axis=1)[:, :-1]
        t = np.arange(1, num_pos + 1)
        total += (t / (t + above)).sum(axis=1) / num_pos
    return total / len(members)


def explicit_map_null(members, edges, null_size, seed):
    """Draw every similarity explicitly and sort each full list, as a reference for `random_map`."""
    rng = np.random.default_rng(seed)
    shared = rng.random((null_size, len(edges)))
    total = np.zeros(null_size)
    for i, (num_pos, num_neg) in enumerate(members):
        mine = [e for e, pair in enumerate(edges) if i in pair]
        private = rng.random((null_size, num_pos - len(mine) + num_neg))
        vals = np.concatenate([shared[:, mine], private], axis=1)
        labels = np.r_[np.ones(num_pos), np.zeros(num_neg)]
        rel = labels[np.argsort(-vals, axis=1)]
        precision = np.cumsum(rel, axis=1) / np.arange(1, num_pos + num_neg + 1)
        total += (precision * rel).sum(axis=1) / num_pos
    return total / len(members)


def scalar_nulls(plan, size, start):
    """Return every signature's samples from the scalar sampler, one at a time."""
    width = max(int(np.max(plan[1])), int(np.max(np.diff(plan[3]))), 1)
    ev, vals = np.empty(width), np.empty(width)
    n_sig = len(plan[0]) - 1
    return np.array(
        [
            [joint.joint_sample(s, start + j, *plan, ev, vals) for j in range(size)]
            for s in range(n_sig)
        ]
    )


def test_vectorized_matches_scalar_draws():
    """The NumPy sampler returns the scalar sampler's samples, bit for bit."""
    plan = joint.joint_plan([joint.canonical_signature(*sig) for sig in SIGNATURES], 7)
    got = joint.joint_nulls(plan, 300, start=2**33 - 100)
    np.testing.assert_array_equal(got, scalar_nulls(plan, 300, 2**33 - 100))


@pytest.mark.parametrize(
    "rows", [slice(None, None, 2), slice(None, None, -1), slice(5, 1, -2), slice(1, 3)]
)
def test_rows_take_any_slice(rows):
    """``rows`` selects signatures as indexing a list with that slice does."""
    plan = joint.joint_plan([joint.canonical_signature(*sig) for sig in SIGNATURES], 7)
    every = joint.joint_nulls(plan, 50, start=3)
    np.testing.assert_array_equal(joint.joint_nulls(plan, 50, 3, rows), every[rows])


def test_chunks_bound_the_per_sample_arrays(monkeypatch):
    """Samples per vectorized call shrink with edges and positives; draws do not change."""
    complete = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    sigs = [
        joint.canonical_signature([(9, 40)] * 10, complete),  # 45 edges
        joint.canonical_signature([(90, 30), (3, 30)], [(0, 1)]),  # 90 positives
        joint.canonical_signature(*SIGNATURES[3]),
    ]
    plan = joint.joint_plan(sigs, 5)
    expected = joint.joint_nulls(plan, 500, start=11)
    sizes = []
    full = joint._joint_null_np

    def spy(plan, s, start, size):
        n_edge = plan[3][s + 1] - plan[3][s]
        width = max(n_edge, int(plan[1][plan[0][s] : plan[0][s + 1]].max()))
        sizes.append(size * width)
        return full(plan, s, start, size)

    monkeypatch.setattr(joint, "_joint_null_np", spy)
    monkeypatch.setattr(joint, "_CHUNK_ELEMENTS", 2**9, raising=False)
    np.testing.assert_array_equal(joint.joint_nulls(plan, 500, start=11), expected)
    assert max(sizes) <= 2**9
    p = joint.joint_pvalues(np.full(3, 0.3), np.arange(3), sigs, 500, 5)
    assert max(sizes) <= 2**9 and 0 < p.min() and p.max() <= 1


def test_pvalue_hits_match_scalar_draws(monkeypatch):
    """Joint p-values count exactly the scalar samples at or above each score."""
    monkeypatch.setattr(joint, "_CHUNK", 64)  # several chunks per signature
    sigs = [joint.canonical_signature(*sig) for sig in SIGNATURES[:4]]
    scores = np.array([0.1, 0.07, 0.3, 0.5, 0.02, 0.2])
    sig_ix = np.array([0, 0, 1, 2, 3, 1])
    p = joint.joint_pvalues(scores, sig_ix, sigs, 400, seed=11)
    null = scalar_nulls(joint.joint_plan(sigs, 11), 400, 0)
    hits = (null[sig_ix] >= (scores - 1e-10)[:, None]).sum(axis=1)
    np.testing.assert_array_equal(p, (hits + 1) / 401)


def test_log_is_accurate_and_vectorizes_exactly():
    """The series logarithm is within a few ulp and identical elementwise."""
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.random(5000), np.exp(rng.normal(0, 30, 5000)), [1.0, 0.5, 2.0, 5e-324]]
    )
    got = joint._log_np(x)
    np.testing.assert_array_equal(got, [joint._log(float(v)) for v in x])
    exact = np.log(x.astype(np.longdouble))
    assert (np.abs(got - exact) <= 4 * np.spacing(np.abs(got))).all()
    assert joint._log(0.0) == -math.inf and joint._log_np(np.zeros(1))[0] == -np.inf


@pytest.mark.parametrize(
    "n, p",
    [(5, 0.3), (40, 0.1), (100, 0.5), (1000, 0.03), (1000, 0.9), (7, 0.99)],
)
def test_binomial_matches_scipy(n, p):
    """The counter-based binomial (inversion and BTRS branches) has the right law."""
    j = np.arange(20_000, dtype=np.uint64)
    c0, c1 = j & joint.M32, j >> joint.S32
    key = (np.uint64(1), np.uint64(2))
    draws, blocks = joint._binomial_np(
        np.full(len(j), n), np.full(len(j), p), c0, c1, np.full(len(j), 3), 0, *key
    )
    scalar = [joint.binomial(n, p, c0[i], c1[i], 3, 0, *key) for i in range(500)]
    assert [tuple(map(int, d)) for d in scalar] == list(zip(draws[:500], blocks[:500]))
    ks = np.arange(n + 1)
    expected = stats.binom.pmf(ks, n, p) * len(draws)
    observed = np.bincount(draws, minlength=n + 1)
    keep = expected > 5
    chi2 = ((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum()
    assert stats.chi2.sf(chi2, keep.sum() - 1) > 1e-3


@pytest.mark.parametrize("conf", [(3, 100), (1, 30), (20, 50)])
def test_singleton_is_the_ap_null(conf):
    """A group of one member follows the AP null of its configuration."""
    got = null_of([conf], [], 200_000).astype(np.float32)  # random_ap's precision
    ref = compute.random_ap(200_000, conf[0], conf[0] + conf[1], seed=1)
    assert stats.ks_2samp(got, ref).pvalue > 1e-3


@pytest.mark.parametrize("members, edges", SIGNATURES[:3] + [([(2, 4)] * 2, [(0, 1)])])
def test_matches_numpy_reference(members, edges):
    """The sampler has the law of `random_map` (#151's sampler)."""
    got = null_of(members, edges, 100_000)
    ref = random_map(members, edges, 100_000, seed=4)
    assert stats.ks_2samp(got, ref).pvalue > 1e-3


@pytest.mark.parametrize("members, edges", SIGNATURES[:3])
def test_reference_matches_explicit_simulation(members, edges):
    """`random_map`'s gaps reproduce drawing every similarity explicitly."""
    fast = random_map(members, edges, 50_000, seed=3)
    slow = explicit_map_null(members, edges, 50_000, seed=4)
    assert stats.ks_2samp(fast, slow).pvalue > 1e-3


def test_replicate_moments():
    """Mean and sd of four replicates with 3 positives among 103 (closed forms)."""
    got = null_of([(3, 100)] * 4, COMPLETE4, 400_000)
    assert abs(got.mean() - 0.069263) < 3e-4
    assert abs(got.std() - 0.049071) < 3e-4


def test_atom_of_two_members():
    """Perfect retrieval of two replicates among 4 controls has probability 1/9."""
    got = null_of([(1, 4)] * 2, [(0, 1)], 200_000)
    rate = np.mean(got >= 1 - 1e-12)
    assert abs(rate - 1 / 9) < 4 * np.sqrt(1 / 9 * 8 / 9 / 200_000)


def test_stream_is_reproducible(monkeypatch):
    """Samples depend only on (seed, signature, index), not chunking or rows."""
    sigs = [
        joint.canonical_signature([(3, 20)] * 4, COMPLETE4),
        joint.canonical_signature([(1, 5), (2, 9)], [(0, 1)]),
    ]
    plan = joint.joint_plan(sigs, 7)
    full = joint.joint_nulls(plan, 1000)
    np.testing.assert_array_equal(full[:, 300:700], joint.joint_nulls(plan, 400, 300))
    np.testing.assert_array_equal(
        full[1:], joint.joint_nulls(plan, 1000, rows=slice(1, 2))
    )
    alone = joint.joint_plan(sigs[1:], 7)
    np.testing.assert_array_equal(full[1], joint.joint_nulls(alone, 1000)[0])
    monkeypatch.setattr(joint, "_CHUNK", 64)
    np.testing.assert_array_equal(full, joint.joint_nulls(plan, 1000))
    other = joint.joint_nulls(joint.joint_plan(sigs, 8), 1000)
    assert not np.array_equal(full, other)


def test_canonical_signature_ignores_member_order():
    """Replicate groups listing the same members in another order share a null."""
    a = joint.canonical_signature([(3, 100), (3, 99), (3, 100), (3, 100)], COMPLETE4)
    b = joint.canonical_signature([(3, 99), (3, 100), (3, 100), (3, 100)], COMPLETE4)
    assert a == b
    path = [(0, 1), (1, 2)]
    c = joint.canonical_signature([(1, 9), (2, 9), (1, 8)], path)
    d = joint.canonical_signature([(1, 8), (2, 9), (1, 9)], path)
    assert c != d  # not complete: member order carries the structure


def test_joint_pvalues_counts_ties():
    """Two perfect replicates among 100 profiles: P(mAP >= 1) = 1/199."""
    sig = joint.canonical_signature([(1, 99)] * 2, [(0, 1)])
    p = joint.joint_pvalues(np.array([1.0, 1.0]), [0, 0], [sig], 200_000, seed=0)
    assert p[0] == p[1]
    assert abs(p[0] - 1 / 199) < 5 * np.sqrt(1 / 199 / 200_000)


def test_plan_validates():
    """Edges outside the group or beyond a member's positives are rejected."""
    with pytest.raises(ValueError, match="distinct members"):
        joint.joint_plan([(((1, 3), (1, 3)), ((0, 0),))], 0)
    with pytest.raises(ValueError, match="positive pairs inside the group"):
        joint.joint_plan([(((1, 3), (1, 3), (1, 3)), ((0, 1), (0, 2), (1, 2)))], 0)
    with pytest.raises(ValueError, match="sig_ix must index"):
        joint.joint_pvalues(np.array([0.5]), [1], [((((1, 3),), ()))], 10, 0)


def test_stream_keys_are_pinned():
    """Both nulls key their streams with common.stream_key; its derivation is fixed."""
    sig = joint.canonical_signature([(1, 5), (2, 9)], [(0, 1)])
    k0, k1 = joint.signature_keys(7, [sig])
    assert (int(k0[0]), int(k1[0])) == common.stream_key(7, sig)
    assert common.stream_key(7, sig) == (1817989492, 2765743222)
    assert permutation.group_key(3, [12, 4, 7]) == common.stream_key(3, (4, 7, 12))
    assert common.stream_key(3, (4, 7, 12)) == (1999174365, 1542470604)


@pytest.mark.parametrize("n", [0, 1, 2, 31, 32, 33, 100])
def test_head_sort_sorts_the_head_only(n):
    """The kernels' head sort sorts vals[:n] by insertion up to its threshold, beyond it too."""
    vals = np.random.default_rng(n).random(n + 4)
    head, tail = np.sort(vals[:n]), vals[n:].copy()
    common.make_head_sort(32)(vals, n)
    np.testing.assert_array_equal(vals[:n], head)
    np.testing.assert_array_equal(vals[n:], tail)
