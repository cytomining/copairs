"""The label-permutation mAP null: relabel profiles, keep their similarities.

The paper's null hypothesis is that a group's profiles and its reference
profiles are drawn from the same distribution, so which of them carry the
group's label is arbitrary. This module tests exactly that: a draw picks, in
each block, as many profiles as the group has members there from the block's
pool (its members and their shared references), gives them the group's
label, and recomputes the group's mAP from the observed similarities with the
same pair rules. Unlike the joint and copied nulls, it keeps whatever
dependence the data induces between members' rank lists.

Blocks are sets of members with identical negatives (e.g. replicates of one
plate ranked against that plate's controls); pools of different blocks must
be disjoint and must not contain members. A member's positives must be other
members of its group. Draws come from Philox streams, so draw ``j`` of a
group is a pure function of ``(seed, group members, j)``.
"""

import hashlib

import numba
import numpy as np

from copairs.nulls.philox import M32, S32, uniform53, philox4x32
from copairs.nulls.pvalues import TIE_TOL, _progress, resolve_seed, _check_null_size

_philox = numba.njit(inline="always")(philox4x32)
_uniform = numba.njit(inline="always")(uniform53)

# Draws per parallel task; each task allocates its scratch once.
_TASK = 128
# Reference profiles per group pool; a pool keeps its key matrix and each
# reference's sorted keys, about 2 * MAX_POOL**2 float32 (2 GiB).
MAX_POOL = 2**14
# Pairs per similarity call (and per sorted chunk) while building a pool.
_PAIR_CHUNK = 2**20
# Pools up to this size also store each reference's rank of every pool
# profile (RANK_POOL**2 int32), replacing a binary search per positive.
RANK_POOL = 2**12


@numba.njit(inline="always")
def _less(a, key):
    """Return whether ``a`` ranks before ``key`` (NaN sorts last, as in NumPy)."""
    return not np.isnan(a) and (np.isnan(key) or a < key)


@numba.njit(inline="always")
def _count_less(row, lo, hi, key):
    """Return ``lo`` plus the entries of sorted ``row[lo:hi]`` ranking before ``key``."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if _less(row[mid], key):
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(inline="always")
def _key(x, y, n, kmu, krr):
    """Rank key between pool codes ``x`` and ``y`` (members ``< n``, references after)."""
    if x < n:
        return kmu[x, y]
    if y < n:
        return kmu[y, x]
    return krr[x - n, y - n]


@numba.njit(inline="always")
def _rank(x, y, n, rmu, rcm, rrr):
    """Count the references of ``x``'s block ranked before ``y`` in ``x``'s list."""
    if x < n:
        return rmu[x, y]
    if y < n:
        return rcm[x - n, y]
    return rrr[x - n, y - n]


@numba.njit(parallel=True, cache=True)
def _row_ranks(row_ptr, rows, keys, out):
    """``out[i, j]``: entries of sorted ``rows[row_ptr[i]:row_ptr[i + 1]]`` before ``keys[i, j]``."""
    for i in numba.prange(keys.shape[0]):
        lo, hi = row_ptr[i], row_ptr[i + 1]
        for j in range(keys.shape[1]):
            out[i, j] = _count_less(rows, lo, hi, keys[i, j]) - lo


@numba.njit(inline="always")
def _sort_with(keys, vals, size):
    """Sort ``keys[:size]`` in place (NaN last), carrying ``vals`` along."""
    for i in range(1, size):
        k, v = keys[i], vals[i]
        z = i - 1
        while z >= 0 and _less(k, keys[z]):
            keys[z + 1] = keys[z]
            vals[z + 1] = vals[z]
            z -= 1
        keys[z + 1] = k
        vals[z + 1] = v


@numba.njit(inline="always")
def _sort_keys(keys, size):
    for i in range(1, size):
        k = keys[i]
        z = i - 1
        while z >= 0 and _less(k, keys[z]):
            keys[z + 1] = keys[z]
            z -= 1
        keys[z + 1] = k


@numba.njit(cache=True)
def _group_map(
    pm, in_pm, n, slot_block, adj_ptr, adj, bm_ptr, bm_slot, kmu, krr,
    srr_ptr, srr, smr_ptr, smr, rmu, rcm, rrr, ranked, pk, pr, plus, minus,
):  # fmt: skip
    """Return the mAP of the group whose slot ``s`` holds pool code ``pm[s]``.

    A slot's rank list is its adjacent slots (positives) and its block's pool
    minus the block's pseudo-members (negatives). Counting the negatives
    ranked before each positive: the block's references (a stored rank when
    ``ranked``, else a binary search in the code's sorted row), plus original
    members not drawn, minus drawn references.
    """
    total = 0.0
    for s in range(n):
        x = pm[s]
        b = slot_block[s]
        deg = adj_ptr[s + 1] - adj_ptr[s]
        for d in range(deg):
            y = pm[adj[adj_ptr[s] + d]]
            pk[d] = _key(x, y, n, kmu, krr)
            pr[d] = _rank(x, y, n, rmu, rcm, rrr) if ranked else 0
        _sort_with(pk, pr, deg)
        n_plus = 0
        n_minus = 0
        for e in range(bm_ptr[b], bm_ptr[b + 1]):
            m = bm_slot[e]
            if not in_pm[m]:
                plus[n_plus] = _key(x, m, n, kmu, krr)
                n_plus += 1
            q = pm[m]
            if q >= n and q != x:
                minus[n_minus] = _key(x, q, n, kmu, krr)
                n_minus += 1
        _sort_keys(plus, n_plus)
        _sort_keys(minus, n_minus)
        if x < n:
            row, lo, hi = smr, smr_ptr[x], smr_ptr[x + 1]
        else:
            row, lo, hi = srr, srr_ptr[x - n], srr_ptr[x - n + 1]
        ip = 0
        im = 0
        acc = 0.0
        for d in range(deg):
            key = pk[d]
            while ip < n_plus and _less(plus[ip], key):
                ip += 1
            while im < n_minus and _less(minus[im], key):
                im += 1
            refs = pr[d] if ranked else _count_less(row, lo, hi, key) - lo
            before = refs + ip - im
            acc += (d + 1) / (d + 1 + before)
        total += acc / deg
    return total / n


@numba.njit(inline="always")
def _next_uniform(c0, c1, blk, spare, k0, k1):
    """Next uniform of draw ``(c0, c1)``'s stream: ``(u, blk, spare)``."""
    if spare >= 0.0:
        return spare, blk, -1.0
    w0, w1, w2, w3 = _philox(c0, c1, np.uint64(blk), np.uint64(0), k0, k1)
    return _uniform(w0, w1), blk + 1, _uniform(w2, w3)


@numba.njit(inline="always")
def _draw(j, n, n_block, bm_ptr, bm_slot, cr_lo, cr_hi, k0, k1, pm, in_pm, chosen):
    """Fill ``pm`` with draw ``j``.

    In each block, the members' slots get a uniform random subset of the
    block's pool (codes of its members, then of its references) in random order.
    """
    c0 = np.uint64(j) & M32
    c1 = np.uint64(j) >> S32
    blk = 0
    spare = -1.0
    in_pm[:] = False
    for b in range(n_block):
        k = bm_ptr[b + 1] - bm_ptr[b]
        p = k + cr_hi[b] - cr_lo[b]
        cnt = 0
        for t in range(p - k, p):  # Floyd's algorithm: a uniform k-subset
            u, blk, spare = _next_uniform(c0, c1, blk, spare, k0, k1)
            r = min(int(u * (t + 1)), t)
            for z in range(cnt):
                if chosen[z] == r:
                    r = t
                    break
            chosen[cnt] = r
            cnt += 1
        for z in range(k - 1, 0, -1):  # Fisher-Yates: a uniform order
            u, blk, spare = _next_uniform(c0, c1, blk, spare, k0, k1)
            r = min(int(u * (z + 1)), z)
            chosen[z], chosen[r] = chosen[r], chosen[z]
        for z in range(k):
            e = chosen[z]
            if e < k:
                code = bm_slot[bm_ptr[b] + e]
                in_pm[code] = True
            else:
                code = n + cr_lo[b] + e - k
            pm[bm_slot[bm_ptr[b] + z]] = code


@numba.njit(parallel=True, cache=True)
def _permutation_count(
    n, slot_block, adj_ptr, adj, bm_ptr, bm_slot, cr_lo, cr_hi, kmu, krr,
    srr_ptr, srr, smr_ptr, smr, rmu, rcm, rrr, ranked, k0, k1, tol, null_size,
    out,
):  # fmt: skip
    """Count draws with mAP ``>= identity - tol`` (stored in ``out`` if it has room).

    Returns ``(count, identity)``, where ``identity`` is the mAP of the
    observed labels.
    """
    n_block = len(bm_ptr) - 1
    max_deg = max(1, np.max(np.diff(adj_ptr)))
    max_k = max(1, np.max(np.diff(bm_ptr)))
    identity = _group_map(
        np.arange(n), np.ones(n, dtype=np.bool_), n, slot_block, adj_ptr, adj,
        bm_ptr, bm_slot, kmu, krr, srr_ptr, srr, smr_ptr, smr, rmu, rcm, rrr,
        ranked, np.empty(max_deg, kmu.dtype), np.empty(max_deg, rmu.dtype),
        np.empty(max_k, kmu.dtype), np.empty(max_k, kmu.dtype),
    )  # fmt: skip
    thr = identity - tol
    n_tasks = (null_size + _TASK - 1) // _TASK
    hits = np.zeros(n_tasks, dtype=np.int64)
    for task in numba.prange(n_tasks):
        pm = np.empty(n, dtype=np.int64)
        in_pm = np.empty(n, dtype=np.bool_)
        chosen = np.empty(max_k, dtype=np.int64)
        pk = np.empty(max_deg, dtype=kmu.dtype)
        pr = np.empty(max_deg, dtype=rmu.dtype)
        plus = np.empty(max_k, dtype=kmu.dtype)
        minus = np.empty(max_k, dtype=kmu.dtype)
        for j in range(task * _TASK, min(null_size, (task + 1) * _TASK)):
            _draw(
                j, n, n_block, bm_ptr, bm_slot, cr_lo, cr_hi, k0, k1, pm, in_pm, chosen
            )
            v = _group_map(
                pm, in_pm, n, slot_block, adj_ptr, adj, bm_ptr, bm_slot, kmu, krr,
                srr_ptr, srr, smr_ptr, smr, rmu, rcm, rrr, ranked, pk, pr, plus,
                minus,
            )  # fmt: skip
            if j < len(out):
                out[j] = v
            if v >= thr:
                hits[task] += 1
    return hits.sum(), identity


def _keys(similarity, a, b):
    """float32 rank keys of the pairs ``(a[i], b[i])``, oriented ``(min, max)``."""
    pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
    # As copairs ranks pairs (fastap.ranking.rank_keys; not imported, since
    # fastap imports this package).
    return np.float32(1) - np.asarray(similarity(pairs), dtype=np.float32)


class _Pool:
    """Reference keys of one pool: its blocks' references, concatenated."""

    def __init__(self, blocks, similarity):
        self.blocks = blocks
        self.refs = np.concatenate(blocks)
        if len(self.refs) > MAX_POOL:
            raise ValueError(
                f"a group pool has {len(self.refs)} reference profiles; the "
                f"permutation null supports up to {MAX_POOL}. Split references "
                "into blocks (neg_sameby) or rank fewer of them."
            )
        size = np.cumsum([0] + [len(b) for b in blocks])
        self.cr_lo, self.cr_hi = size[:-1].astype(np.int64), size[1:].astype(np.int64)
        r = len(self.refs)
        krr = np.zeros((r, r), dtype=np.float32)
        lo = 0
        while lo < r:  # rows lo:hi against columns lo:, mirrored below the diagonal
            hi = min(r, lo + max(1, _PAIR_CHUNK // (r - lo)))
            keys = _keys(
                similarity,
                np.repeat(self.refs[lo:hi], r - lo),
                np.tile(self.refs[lo:], hi - lo),
            ).reshape(hi - lo, r - lo)
            krr[lo:hi, lo:] = keys
            krr[lo:, lo:hi] = keys.T
            lo = hi
        np.fill_diagonal(krr, 0)
        self.krr = krr
        # Each reference's keys to the other references of its block, sorted.
        # Its own key becomes NaN, which sorts last, and is cut off.
        width = self.cr_hi - self.cr_lo
        self.srr_ptr = np.concatenate(
            [[0], np.cumsum(np.repeat(np.maximum(width - 1, 0), width))]
        ).astype(np.int64)
        self.srr = np.empty(self.srr_ptr[-1], dtype=np.float32)
        for lo, hi in zip(self.cr_lo, self.cr_hi):
            step = max(1, _PAIR_CHUNK // max(hi - lo, 1))
            for a in range(lo, hi, step):
                b = min(hi, a + step)
                rows = krr[a:b, lo:hi].copy()
                rows[np.arange(b - a), np.arange(a - lo, b - lo)] = np.nan
                rows.sort(axis=1)
                self.srr[self.srr_ptr[a] : self.srr_ptr[b]] = rows[
                    :, : hi - lo - 1
                ].ravel()
        self.ranked = r <= RANK_POOL
        self.rrr = np.zeros((r, r) if self.ranked else (0, 0), dtype=np.int32)
        if self.ranked:
            _row_ranks(self.srr_ptr, self.srr, krr, self.rrr)

    def matches(self, blocks):
        return len(blocks) == len(self.blocks) and all(
            np.array_equal(a, b) for a, b in zip(blocks, self.blocks)
        )


@numba.njit(inline="always")
def _mix(x):
    """splitmix64 finalizer: a well-spread hash of a uint64."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@numba.njit(cache=True)
def _scan_pairs(pairs, group_of, positive, count, h1, h2, bad):
    """Count each member's pairs, hash its partner set, flag broken groups.

    A positive must join two members of one group; a negative must not join
    members of one group. ``h1``, ``h2`` are order-independent hashes of each
    member's partners (a set, since pairs are unique).
    """
    for p in range(len(pairs)):
        a, b = pairs[p, 0], pairs[p, 1]
        ga, gb = group_of[a], group_of[b]
        same = ga >= 0 and ga == gb
        if positive != same:
            if ga >= 0:
                bad[ga] = True
            if gb >= 0:
                bad[gb] = True
        for x, y in ((a, b), (b, a)):
            if group_of[x] >= 0:
                count[x] += 1
                h1[x] += _mix(np.uint64(y) + np.uint64(1))
                h2[x] += _mix(np.uint64(y) ^ np.uint64(0x9E3779B97F4A7C15))


@numba.njit(cache=True)
def _gather(pairs, rep_of, ptr, out):
    """Partners of each representative profile into its CSR segment."""
    fill = ptr[:-1].copy()
    for p in range(len(pairs)):
        a, b = pairs[p, 0], pairs[p, 1]
        for x, y in ((a, b), (b, a)):
            r = rep_of[x]
            if r >= 0:
                out[fill[r]] = y
                fill[r] += 1


def group_layouts(groups, profile, num_pos, num_neg, pos_pairs, neg_pairs):
    """Edges and blocks of every group, checked against its AP counts.

    Returns one ``(members, edges, blocks, block_of)`` per group: its profile
    indices, positive pairs between local member indices, each block's sorted
    reference profiles and each member's block.
    """
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    neg_pairs = np.asarray(neg_pairs, dtype=np.int64).reshape(-1, 2)
    n_prof = 1 + max(
        int(profile.max(initial=-1)),
        int(pos_pairs.max(initial=-1)),
        int(neg_pairs.max(initial=-1)),
    )
    group_of = np.full(n_prof, -1, dtype=np.int64)
    rows = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
    if len(np.unique(profile[rows])) != len(rows):
        raise ValueError("the permutation null needs each profile in one group")
    for g, ix in enumerate(groups):
        group_of[profile[ix]] = g
    bad = np.zeros(len(groups), dtype=np.bool_)
    counts, hashes = [], []
    for pairs, positive in ((pos_pairs, True), (neg_pairs, False)):
        count = np.zeros(n_prof, dtype=np.int64)
        h1 = np.zeros(n_prof, dtype=np.uint64)
        h2 = np.zeros(n_prof, dtype=np.uint64)
        _scan_pairs(pairs, group_of, positive, count, h1, h2, bad)
        counts.append(count)
        hashes.append((h1, h2))
    member = profile[rows]
    expected = (num_pos[rows], num_neg[rows])
    for count, want, kind in zip(counts, expected, ("positive", "negative")):
        wrong = np.flatnonzero(count[member] != want)
        if len(wrong):
            m = member[wrong[0]]
            raise ValueError(
                f"profile {m} has {count[m]} {kind} pairs but {want[wrong[0]]} in "
                "the AP table; pass the pairs average_precision(..., "
                "return_pairs=True) returned with the frame it returned"
            )
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} group(s) have positives outside the group or "
            "negatives inside it; the permutation null needs each group's "
            "positives to be its other members and its references outside it"
        )
    # One representative per distinct negative set; gather only their lists.
    h1, h2 = hashes[1]
    fingerprint = np.stack([counts[1][member], h1[member], h2[member]], axis=1)
    _, rep_ix, block_id = np.unique(
        fingerprint.astype(np.uint64), axis=0, return_index=True, return_inverse=True
    )
    block_id = block_id.ravel()
    rep_of = np.full(n_prof, -1, dtype=np.int64)
    rep_of[member[rep_ix]] = np.arange(len(rep_ix))
    sizes = counts[1][member[rep_ix]]
    ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    refs = np.empty(ptr[-1], dtype=np.int64)
    _gather(neg_pairs, rep_of, ptr, refs)
    ref_sets = [np.sort(refs[ptr[k] : ptr[k + 1]]) for k in range(len(rep_ix))]
    pos_group = group_of[pos_pairs[:, 0]]
    inside = pos_group >= 0
    edge_group = pos_group[inside]
    edge_order = np.argsort(edge_group, kind="stable")
    edge_ptr = np.searchsorted(edge_group[edge_order], np.arange(len(groups) + 1))
    edge_pairs = pos_pairs[inside][edge_order]
    layouts = []
    offset = 0
    for g, ix in enumerate(groups):
        members = profile[ix]
        local = {int(m): i for i, m in enumerate(members)}
        edges = [
            (local[int(a)], local[int(b)])
            for a, b in edge_pairs[edge_ptr[g] : edge_ptr[g + 1]]
        ]
        ids = block_id[offset : offset + len(ix)]
        offset += len(ix)
        uniq, block_of = np.unique(ids, return_inverse=True)
        blocks = [ref_sets[k] for k in uniq]
        if len(blocks) > 1 and len(np.unique(np.concatenate(blocks))) != sum(
            len(b) for b in blocks
        ):
            raise ValueError(
                "members of a group rank overlapping but different references; "
                "the permutation null needs each block of members to share them"
            )
        layouts.append((members, edges, blocks, block_of.ravel().astype(np.int64)))
    return layouts


def group_key(seed: int, members) -> tuple[np.uint64, np.uint64]:
    """Philox key of a group's draws: a pure function of ``(seed, members)``."""
    rep = repr((int(seed), tuple(sorted(int(m) for m in members))))
    word = int.from_bytes(
        hashlib.blake2b(rep.encode(), digest_size=8).digest(), "little"
    )
    return np.uint64(word & 0xFFFFFFFF), np.uint64(word >> 32)


def permutation_null(
    members, edges, blocks, block_of, similarity, null_size, seed, pool=None,
    out=None,
):  # fmt: skip
    """Run ``null_size`` label permutations of one group.

    Returns ``(count, identity, pool)``: the draws whose mAP ties or exceeds
    the observed labels' (``identity``, within :data:`TIE_TOL`), that mAP, and
    the reference pool, reusable by groups ranking the same references.
    ``out`` receives the draws' mAPs if given.
    """
    members = np.asarray(members, dtype=np.int64)
    n = len(members)
    if pool is None or not pool.matches(blocks):
        pool = _Pool(blocks, similarity)
    order = np.argsort(block_of, kind="stable")
    bm_slot = order.astype(np.int64)
    bm_ptr = np.searchsorted(block_of[order], np.arange(len(blocks) + 1)).astype(
        np.int64
    )
    adj_lists = [[] for _ in range(n)]
    for i, j in edges:
        adj_lists[i].append(j)
        adj_lists[j].append(i)
    adj_ptr = np.cumsum([0] + [len(a) for a in adj_lists]).astype(np.int64)
    adj = np.asarray([j for a in adj_lists for j in a], dtype=np.int64)
    targets = np.concatenate([members, pool.refs])
    kmu = _keys(
        similarity, np.repeat(members, len(targets)), np.tile(targets, n)
    ).reshape(n, len(targets))
    smr_rows = [
        np.sort(kmu[i, n + pool.cr_lo[block_of[i]] : n + pool.cr_hi[block_of[i]]])
        for i in range(n)
    ]
    smr_ptr = np.cumsum([0] + [len(r) for r in smr_rows]).astype(np.int64)
    smr = np.concatenate(smr_rows)
    if pool.ranked:
        rmu = np.empty(kmu.shape, dtype=np.int32)
        _row_ranks(smr_ptr, smr, kmu, rmu)
        rcm = np.empty((len(pool.refs), n), dtype=np.int32)
        _row_ranks(pool.srr_ptr, pool.srr, np.ascontiguousarray(kmu[:, n:].T), rcm)
    else:
        rmu = rcm = np.zeros((0, 0), dtype=np.int32)
    k0, k1 = group_key(seed, members)
    out = np.zeros(0) if out is None else out
    count, identity = _permutation_count(
        n, block_of.astype(np.int64), adj_ptr, adj, bm_ptr, bm_slot, pool.cr_lo,
        pool.cr_hi, kmu, pool.krr, pool.srr_ptr, pool.srr, smr_ptr, smr, rmu, rcm,
        pool.rrr, pool.ranked, k0, k1, TIE_TOL, int(null_size), out,
    )  # fmt: skip
    return int(count), float(identity), pool


def permutation_pvalues(
    groups, profile, num_pos, num_total, pos_pairs, neg_pairs, similarity,
    null_size, seed, observed=None, progress_bar=False,
):  # fmt: skip
    """P-value of each group's mAP against its label permutations.

    Parameters
    ----------
    groups : list of np.ndarray
        Each group's rows of the AP table.
    profile : np.ndarray
        Profile index (row of the pairs' frame) of each AP row.
    num_pos, num_total : np.ndarray
        ``n_pos_pairs`` and ``n_total_pairs`` of each AP row, checked against
        the pairs.
    pos_pairs, neg_pairs : np.ndarray
        The ``(n, 2)`` pairs the APs were computed from.
    similarity : callable
        The ``pairs -> similarities`` function the APs were computed with.
    null_size, seed
        Draws per group and seed. ``p = (1 + #{draw >= mAP - tol}) /
        (null_size + 1)``, with the mAP of the observed labels recomputed from
        the pairs and ``tol`` = :data:`TIE_TOL`.
    observed : np.ndarray, optional
        Each group's mAP from the AP table; a group whose recomputed mAP
        differs by more than 1e-6 raises, since then its pairs or similarity
        do not match the table.

    Returns
    -------
    np.ndarray
        float64 p-values.
    """
    null_size = _check_null_size(null_size)
    seed = resolve_seed(seed)
    profile = np.asarray(profile, dtype=np.int64)
    num_pos = np.asarray(num_pos, dtype=np.int64)
    num_neg = np.asarray(num_total, dtype=np.int64) - num_pos
    groups = [np.asarray(ix, dtype=np.int64) for ix in groups]
    layouts = group_layouts(groups, profile, num_pos, num_neg, pos_pairs, neg_pairs)
    # Visit groups pool by pool, so each pool's reference keys are built once.
    order = sorted(
        range(len(groups)),
        key=lambda g: tuple((len(b), *b[:2]) for b in layouts[g][2]),
    )
    pvals = np.empty(len(groups), dtype=np.float64)
    pool = None
    for g in _progress(order, len(order), progress_bar, "permutation null"):
        count, identity, pool = permutation_null(
            *layouts[g], similarity, null_size, seed, pool
        )
        if observed is not None and abs(identity - observed[g]) > 1e-6:
            raise ValueError(
                f"group {g}: mAP {identity} from the pairs differs from "
                f"{observed[g]} in the AP table; the pairs or similarity do not "
                "match the AP scores"
            )
        pvals[g] = (count + 1) / (null_size + 1)
    return pvals
