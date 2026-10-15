"""Tests for the Numba joint mAP null (copairs.nulls.joint)."""

import numpy as np
import pytest
from scipy import stats

from copairs import nulls, compute
from copairs.nulls import joint

COMPLETE4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def null_of(members, edges, size, seed=0, start=0):
    """Joint null samples of one group."""
    plan = joint.joint_plan([joint.canonical_signature(members, edges)], seed)
    return joint.joint_nulls(plan, size, start)[0]


@pytest.mark.parametrize(
    "n, p", [(5, 0.3), (40, 0.1), (100, 0.5), (1000, 0.03), (1000, 0.9), (7, 0.99)]
)
def test_binomial_matches_scipy(n, p):
    """The counter-based binomial (inversion and BTRS branches) has the right law."""
    draws = np.array(
        [
            joint.binomial(
                n, p, np.uint64(j), np.uint64(0), 0, 3, np.uint64(1), np.uint64(2)
            )[0]
            for j in range(20_000)
        ]
    )
    ks = np.arange(n + 1)
    expected = stats.binom.pmf(ks, n, p) * len(draws)
    observed = np.bincount(draws, minlength=n + 1)
    keep = expected > 5
    chi2 = ((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum()
    assert stats.chi2.sf(chi2, keep.sum() - 1) > 1e-3


@pytest.mark.parametrize("conf", [(3, 100), (1, 30), (20, 50)])
def test_singleton_is_the_ap_null(conf):
    """A group of one member follows the exact AP null of its configuration."""
    got = null_of([conf], [], 200_000)
    exact = nulls.ap_nulls(
        np.array([[conf[0], conf[0] + conf[1]]]), 200_000, seed=1, dtype=np.float64
    )[0]
    assert stats.ks_2samp(got, exact).pvalue > 1e-3


@pytest.mark.parametrize(
    "members, edges",
    [
        ([(3, 100)] * 4, COMPLETE4),
        ([(3, 50), (3, 60), (3, 50), (3, 60)], [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ([(1, 30), (2, 40), (2, 40), (1, 30)], [(0, 1), (1, 2), (2, 3)]),
        ([(2, 4)] * 2, [(0, 1)]),
    ],
)
def test_matches_numpy_reference(members, edges):
    """The kernel samples the same law as compute.random_map (#151's sampler)."""
    got = null_of(members, edges, 100_000)
    ref = compute.random_map(members, edges, 100_000, seed=4)
    assert stats.ks_2samp(got, ref).pvalue > 1e-3


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


def test_stream_is_reproducible():
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
