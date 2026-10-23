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

Draw ``j`` is defined by the scalar functions :func:`_make_kernels` builds:
``draw`` places pool profiles in the group's slots and ``group_map``
recomputes the group's mAP from integer rank codes, with IEEE divisions and
additions in a fixed order. :func:`_draws_np` and :func:`_group_maps_np`
perform the same operations vectorized over draws and give identical mAPs.

With ``stop_after``, a group's draws stop at its ``stop_after``-th draw that
reaches its mAP (Besag and Clifford's sequential p-value); draw ``j``
stays the same, so a stopped run is a prefix of the full one.

A multilabel group (:func:`label_permutation_pvalues`) holds one label of a
multilabel column, and its pools are whole components of the graph of
negative pairs, members included: a draw moves the label to as many of each
component's profiles as hold it there. :func:`_make_label_kernels` defines
its draws, and :func:`_draws_pool_np` and :func:`_label_maps_np` are their
vectorized twins.
"""

import os
import hashlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from copairs.nulls.common import (
    chunks,
    progress,
    stream_key,
    group_edges,
    resolve_seed,
    make_head_sort,
    tie_thresholds,
    check_null_size,
    check_pair_counts,
)
from copairs.nulls.philox import (
    M32,
    S32,
    uniform53,
    philox4x32,
    uniform_pair,
    make_uniform_pair,
)

# A slot sorts up to this many drawn profiles by insertion, more by quicksort.
_INSERTION = 128
# Reference profiles per group pool; a pool keeps its key matrix and each
# reference's sorted keys, about 2 * MAX_POOL**2 float32 (2 GiB).
MAX_POOL = 2**14
# Estimated bytes of the pools alive at once (:func:`_pool_bytes`): a group
# with a new pool starts once the pool fits beside the others, so concurrent
# pools hold at most this much, or one pool if it alone is larger (MAX_POOL
# references take about 2 GiB unranked).
POOL_BUDGET = 2**32
# Bytes of one key table over the profiles of all pools of a call, which
# pools then slice (8192 profiles).
KEY_TABLE_BUDGET = 2**28
# Pairs per similarity call (and per sorted chunk) while building a pool.
_PAIR_CHUNK = 2**20
# Pools up to this size also store each reference's rank of every pool
# profile (RANK_POOL**2 int32), replacing a binary search per positive.
RANK_POOL = 2**12
# Elements of the per-slot arrays of one vectorized chunk of draws.
_CHUNK_ELEMENTS = 2**22
_FULL_TABLE = 2**20  # largest (code, code) table a layout gathers from
# A run that may stop early (``stop_after``) starts with this many draws and
# doubles them each round, up to the layout's chunk.
_FIRST_ROUND = 256

# Rank code of a NaN key: after every number.
_NAN = np.uint32(0xFFFFFFFF)
# The CPU count of ThreadPoolExecutor's default threads (3.13+: usable CPUs).
_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)


def _make_count_less(jit=lambda f: f):
    """Build ``count_less``, a binary search; ``jit`` as in :func:`_make_kernels`."""

    @jit
    def count_less(row, lo, hi, key):
        """Return ``lo`` plus the entries of sorted ``row[lo:hi]`` below ``key``."""
        while lo < hi:
            mid = (lo + hi) >> 1
            if row[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return count_less


def _make_ordered_subset(philox, uniform, jit=lambda f: f):
    """Build ``ordered_subset``, which draws one block's profiles from a draw's stream.

    ``jit`` as in :func:`_make_kernels`.
    """
    uniform_pair = make_uniform_pair(philox, uniform, jit)

    @jit
    def next_uniform(c0, c1, blk, spare, k0, k1):
        """Next uniform of draw ``(c0, c1)``'s stream: ``(u, blk, spare)``."""
        if spare >= 0.0:
            return spare, blk, -1.0
        u, spare = uniform_pair(c0, c1, blk, 0, k0, k1)
        return u, blk + 1, spare

    @jit
    def ordered_subset(k, p, c0, c1, blk, spare, k0, k1, chosen):
        """Fill ``chosen[:k]`` with a uniform ``k``-subset of ``range(p)`` in random order.

        Takes the next uniforms of draw ``(c0, c1)``'s stream, which is at
        Philox block ``blk`` with ``spare`` left over (-1 if none), and
        returns the stream's new ``(blk, spare)``.
        """
        cnt = 0
        for t in range(p - k, p):  # Floyd's algorithm: a uniform k-subset
            u, blk, spare = next_uniform(c0, c1, blk, spare, k0, k1)
            r = min(int(u * (t + 1)), t)
            for z in range(cnt):
                if chosen[z] == r:
                    r = t
                    break
            chosen[cnt] = r
            cnt += 1
        for z in range(k - 1, 0, -1):  # Fisher-Yates: a uniform order
            u, blk, spare = next_uniform(c0, c1, blk, spare, k0, k1)
            r = min(int(u * (z + 1)), z)
            chosen[z], chosen[r] = chosen[r], chosen[z]
        return blk, spare

    return ordered_subset


def _make_kernels(philox, uniform, jit=lambda f: f):
    """Build the scalar draw and mAP functions around ``philox`` and ``uniform``.

    ``jit`` compiles each part (e.g. ``numba.njit``) or leaves it Python.
    Returns ``(draw, group_map)``.
    """
    count_less = _make_count_less(jit)
    sort_entries = make_head_sort(_INSERTION, jit)
    ordered_subset = _make_ordered_subset(philox, uniform, jit)

    @jit
    def group_map(
        pm, n, slot_block, adjm, deg, kmu, krr, gptr, grow, smr_ptr, smr, srr_ptr,
        srr, rmu, rcm, rrr, ranked, members, refs, ent, rank, term,
    ):  # fmt: skip
        """Return the mAP of the group whose slot ``s`` holds pool code ``pm[s]``.

        The profile ``x`` in slot ``s`` ranks its adjacent slots (positives)
        against its block's pool minus the profiles drawn into the block
        (negatives). Its codes to the drawn profiles and their reference ranks
        are gathered first, splitting the slots into drawn members and drawn
        references so that each loop reads one table. The drawn profiles that
        count, one ``(code, slot)`` integer each, are then sorted and walked in
        ``x``'s rank order: the negatives before a positive are the pool
        profiles ranked before it (references: its rank; members: a merge with
        ``x``'s sorted member codes) minus the drawn ones ranked before it.
        Precisions are summed in rank order.
        """
        n_mem = 0
        n_ref = 0
        for t in range(n):
            if pm[t] < n:
                members[n_mem] = t
                n_mem += 1
            else:
                refs[n_ref] = t
                n_ref += 1
        total = 0.0
        for s in range(n):
            x = pm[s]
            b = slot_block[s]
            if x < n:
                for t in range(n):
                    y = pm[t]
                    ent[t] = (np.uint64(kmu[x, y]) << np.uint64(32)) | np.uint64(t)
                    if ranked:
                        rank[t] = rmu[x, y]
            else:
                xr = x - n
                for i in range(n_mem):
                    t = members[i]
                    y = pm[t]
                    ent[t] = (np.uint64(kmu[y, x]) << np.uint64(32)) | np.uint64(t)
                    if ranked:
                        rank[t] = rcm[xr, y]
                for i in range(n_ref):
                    t = refs[i]
                    y = pm[t] - n
                    ent[t] = (np.uint64(krr[xr, y]) << np.uint64(32)) | np.uint64(t)
                    if ranked:
                        rank[t] = rrr[xr, y]
            size = 0
            for t in range(n):  # keep the slots that count, in place
                ent[size] = ent[t]
                size += t != s and (adjm[s, t] or slot_block[t] == b)
            sort_entries(ent, size)
            if x < n:
                row, lo, hi = smr, smr_ptr[x], smr_ptr[x + 1]
            else:
                row, lo, hi = srr, srr_ptr[x - n], srr_ptr[x - n + 1]
            g0, g1 = gptr[x], gptr[x + 1]
            g = g0
            d = 0
            drawn = 0  # drawn block profiles ranked strictly before ``key``
            tied = 0  # drawn block profiles tied with ``key`` so far
            key = np.uint64(0)
            for e in range(size):
                k = ent[e] >> np.uint64(32)
                t = np.int64(ent[e] & np.uint64(0xFFFFFFFF))
                if e == 0 or k != key:
                    drawn += tied
                    tied = 0
                    key = k
                    while g < g1 and grow[g] < key:
                        g += 1
                if adjm[s, t]:
                    if ranked:
                        before = np.int64(rank[t])
                    else:
                        before = count_less(row, lo, hi, key) - lo
                    before += g - g0 - drawn
                    term[d] = (d + 1) / (d + 1 + before)
                    d += 1
                tied += slot_block[t] == b
            acc = 0.0
            for d in range(deg[s]):
                acc += term[d]
            total += acc / deg[s]
        return total / n

    @jit
    def draw(j, n, n_block, bm_ptr, bm_slot, cr_lo, cr_hi, k0, k1, pm, chosen):
        """Fill ``pm`` with draw ``j``.

        In each block, the members' slots get a uniform random subset of the
        block's pool (codes of its members, then of its references) in random
        order.
        """
        c0 = np.uint64(j) & M32
        c1 = np.uint64(j) >> S32
        blk = 0
        spare = -1.0
        for b in range(n_block):
            k = bm_ptr[b + 1] - bm_ptr[b]
            p = k + cr_hi[b] - cr_lo[b]
            blk, spare = ordered_subset(k, p, c0, c1, blk, spare, k0, k1, chosen)
            for z in range(k):
                e = chosen[z]
                if e < k:
                    code = bm_slot[bm_ptr[b] + e]
                else:
                    code = n + cr_lo[b] + e - k
                pm[bm_slot[bm_ptr[b] + z]] = code

    return draw, group_map


# The scalar functions in Python: the definition of every draw.
_draw, _group_map = _make_kernels(philox4x32, uniform53)


def _keys(similarity, a, b):
    """Rank codes of the pairs ``(a[i], b[i])``, oriented ``(min, max)``.

    A pair's float32 rank key ``1 - similarity``, as copairs ranks pairs,
    mapped to a uint32 with the same order: NaN after every number, and -0
    equal to 0.
    """
    pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
    keys = np.float32(1) - np.asarray(similarity(pairs), dtype=np.float32)
    keys += np.float32(0)  # -0 -> +0
    bits = keys.view(np.uint32)
    codes = np.where(bits >= 0x80000000, ~bits, bits | np.uint32(0x80000000))
    codes[np.isnan(keys)] = _NAN
    return codes.astype(np.uint32)


def _key_matrix(similarity, profiles, map=map):
    """Rank codes between all ``profiles``, ``_PAIR_CHUNK`` pairs at a time.

    ``map`` runs the chunks (e.g. an executor's, to compute them on threads);
    each writes its own rows and columns.
    """
    r = len(profiles)
    out = np.zeros((r, r), dtype=np.uint32)
    bands = []
    lo = 0
    while lo < r:  # rows lo:hi against columns lo:, mirrored below the diagonal
        hi = min(r, lo + max(1, _PAIR_CHUNK // (r - lo)))
        bands.append((lo, hi))
        lo = hi

    def band(lo_hi):
        lo, hi = lo_hi
        keys = _keys(
            similarity,
            np.repeat(profiles[lo:hi], r - lo),
            np.tile(profiles[lo:], hi - lo),
        ).reshape(hi - lo, r - lo)
        out[lo:hi, lo:] = keys
        out[lo:, lo:hi] = keys.T

    for _ in map(band, bands):
        pass
    np.fill_diagonal(out, 0)
    return out


class _KeyTable:
    """Rank codes between all profiles of a set of pools, computed once.

    Pools slice their key matrices from it instead of recomputing them; a
    pair's code does not depend on the chunk it is computed in.
    """

    def __init__(self, profiles, similarity, map=map):
        self.profiles = np.asarray(profiles, dtype=np.int64)  # sorted
        self.keys = _key_matrix(similarity, self.profiles, map)

    def matrix(self, refs):
        """Return the key matrix of ``refs``, as ``_key_matrix`` computes it."""
        ix = np.searchsorted(self.profiles, refs)
        return self.keys[np.ix_(ix, ix)]


def _key_table(pools, similarity, map=map):
    """Return a ``_KeyTable`` of the profiles of ``pools``, or None if it does not pay.

    ``pools`` lists the profiles of each distinct pool. The table is built
    when its ``4 * U**2`` bytes over the ``U`` profiles of all pools fit in
    ``KEY_TABLE_BUDGET`` and its ``U**2`` pairs are fewer than the pools' key
    matrices hold together.
    """
    if not pools:
        return None
    profiles = np.unique(np.concatenate(pools))
    u = len(profiles)
    if 4 * u * u > KEY_TABLE_BUDGET or u * u >= sum(len(r) ** 2 for r in pools):
        return None
    return _KeyTable(profiles, similarity, map)


def _pool_key(blocks):
    """Hashable key of a pool's ``blocks`` of references."""
    return tuple(b.tobytes() for b in blocks)


def _row_words(row_ptr, rows):
    """Sorted rows ``rows[row_ptr[i]:row_ptr[i + 1]]`` as one sorted array of (row, code) words."""
    seg = np.repeat(np.arange(len(row_ptr) - 1, dtype=np.uint64), np.diff(row_ptr))
    return (seg << np.uint64(32)) | rows.astype(np.uint64)


def _row_ranks(row_ptr, flat, keys):
    """``out[i, j]``: entries of sorted row ``i`` below ``keys[i, j]``.

    ``flat`` is the rows' :func:`_row_words`.
    """
    n_rows = len(row_ptr) - 1
    out = np.empty(keys.shape, dtype=np.int32)
    step = max(1, _CHUNK_ELEMENTS // max(keys.shape[1], 1))
    for lo, size in chunks(n_rows, step):
        hi = lo + size
        rows_ix = np.arange(lo, hi, dtype=np.uint64)[:, None] << np.uint64(32)
        q = rows_ix | keys[lo:hi].astype(np.uint64)
        out[lo:hi] = np.searchsorted(flat, q, side="left") - row_ptr[lo:hi, None]
    return out


class _Pool:
    """Reference keys of one pool: its blocks' references, concatenated."""

    def __init__(self, blocks, similarity, table=None):
        self.blocks = blocks
        self.refs = np.concatenate(blocks).astype(np.int64)
        if len(self.refs) > MAX_POOL:
            raise ValueError(
                f"a group pool has {len(self.refs)} reference profiles; the "
                f"permutation null supports up to {MAX_POOL}. Split references "
                "into blocks (neg_sameby) or rank fewer of them."
            )
        size = np.cumsum([0] + [len(b) for b in blocks])
        self.cr_lo, self.cr_hi = size[:-1].astype(np.int64), size[1:].astype(np.int64)
        r = len(self.refs)
        if table is None:
            self.krr = krr = _key_matrix(similarity, self.refs)
        else:
            self.krr = krr = table.matrix(self.refs)
        # Each reference's codes to the other references of its block, sorted.
        # Its own code becomes NaN's, which sorts last, and is cut off.
        width = self.cr_hi - self.cr_lo
        self.srr_ptr = np.concatenate(
            [[0], np.cumsum(np.repeat(np.maximum(width - 1, 0), width))]
        ).astype(np.int64)
        self.srr = np.empty(self.srr_ptr[-1], dtype=np.uint32)
        for lo, hi in zip(self.cr_lo, self.cr_hi):
            step = max(1, _PAIR_CHUNK // max(hi - lo, 1))
            for off, size in chunks(hi - lo, step):
                a, b = lo + off, lo + off + size
                rows = krr[a:b, lo:hi].copy()
                rows[np.arange(b - a), np.arange(a - lo, b - lo)] = _NAN
                rows.sort(axis=1)
                self.srr[self.srr_ptr[a] : self.srr_ptr[b]] = rows[
                    :, : hi - lo - 1
                ].ravel()
        self.ranked = r <= RANK_POOL
        self.rrr = np.zeros((0, 0), dtype=np.int32)
        if self.ranked:  # the sorted rows' words serve every layout's rcm too
            self.srr_words = _row_words(self.srr_ptr, self.srr)
            self.rrr = _row_ranks(self.srr_ptr, self.srr_words, krr)

    def matches(self, blocks):
        return len(blocks) == len(self.blocks) and all(
            np.array_equal(a, b) for a, b in zip(blocks, self.blocks)
        )


class _Layout:
    """One group's slots, blocks and rank codes, as the draw functions take them.

    Pool codes number the group's members ``0 .. n - 1`` (a member's code is
    its slot) and then the pool's references.
    """

    def __init__(self, members, edges, block_of, similarity, pool, draws=0):
        members = np.asarray(members, dtype=np.int64)
        block_of = np.asarray(block_of, dtype=np.int64)
        self.n = n = len(members)
        self.identity = np.arange(n)  # the observed labels' codes
        self.pool = pool
        self.slot_block = block_of
        self.bm_slot = np.argsort(block_of, kind="stable").astype(np.int64)
        n_block = len(pool.blocks)
        bm_ptr = np.searchsorted(block_of[self.bm_slot], np.arange(n_block + 1))
        self.bm_ptr = bm_ptr.astype(np.int64)
        self.adjm = adjm = np.zeros((n, n), dtype=np.bool_)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        adjm[edges[:, 0], edges[:, 1]] = True
        adjm[edges[:, 1], edges[:, 0]] = True
        self.deg = adjm.sum(axis=1).astype(np.int64)
        targets = np.concatenate([members, pool.refs])
        self.kmu = kmu = _keys(
            similarity, np.repeat(members, len(targets)), np.tile(targets, n)
        ).reshape(n, len(targets))
        # Each member's codes to its block's references, sorted.
        smr_rows = [
            np.sort(kmu[i, n + pool.cr_lo[block_of[i]] : n + pool.cr_hi[block_of[i]]])
            for i in range(n)
        ]
        self.smr_ptr = np.cumsum([0] + [len(r) for r in smr_rows]).astype(np.int64)
        self.smr = np.concatenate(smr_rows).astype(np.uint32)
        # Each pool code's codes to its block's members, sorted, its own cut off.
        widths = np.diff(self.bm_ptr)
        code_block = np.concatenate(
            [block_of, np.repeat(np.arange(n_block), pool.cr_hi - pool.cr_lo)]
        )
        self.gptr = np.concatenate([[0], np.cumsum(widths[code_block])]).astype(
            np.int64
        )
        self.grow = np.empty(self.gptr[-1], dtype=np.uint32)
        for b in range(n_block):
            block_members = self.bm_slot[self.bm_ptr[b] : self.bm_ptr[b + 1]]
            codes = np.concatenate(
                [block_members, n + np.arange(pool.cr_lo[b], pool.cr_hi[b])]
            )
            rows = kmu[block_members][:, codes].T.copy()
            own = np.arange(len(block_members))
            rows[own, own] = _NAN
            rows.sort(axis=1)
            self.grow[self.gptr[codes][:, None] + np.arange(widths[b])] = rows
        # The rows as one sorted array of (code, key) words, for one searchsorted.
        code_of = np.repeat(
            np.arange(len(widths[code_block]), dtype=np.uint64), widths[code_block]
        )
        self.gkeys = (code_of << np.uint64(32)) | self.grow.astype(np.uint64)
        if pool.ranked:
            self.rmu = _row_ranks(self.smr_ptr, _row_words(self.smr_ptr, self.smr), kmu)
            self.rcm = _row_ranks(
                pool.srr_ptr, pool.srr_words, np.ascontiguousarray(kmu[:, n:].T)
            )
        else:
            self.rmu = self.rcm = np.zeros((0, 0), dtype=np.int32)
        # Small layouts gather keys from one (code, code) table. A positive y
        # of x is ranked after full_before[x, y] pool profiles (x's rank of y
        # among references plus its block members before y); that table is
        # built when ``draws`` draws gather more positives than it has
        # entries, else ranks are gathered from a (code, code) rank table.
        self.full_keys = self.full_ranks = self.full_before = None
        m = n + len(pool.refs)
        if m**2 <= _FULL_TABLE:
            self.full_keys = _full_table(n, kmu, kmu.T[n:], pool.krr)
            if pool.ranked:
                self.full_ranks = _full_table(n, self.rmu, self.rcm, pool.rrr)
            if pool.ranked and m**2 <= draws * int(self.deg.sum()):
                x = np.repeat(np.arange(m), m)
                word = (x.astype(np.uint64) << np.uint64(32)) | self.full_keys.ravel()
                members_before = np.searchsorted(self.gkeys, word) - self.gptr[x]
                before = self.full_ranks.ravel() + members_before
                self.full_before, self.full_ranks = before.reshape(m, m), None
        # Slots each slot's rank list counts: its positives and its block.
        self.counted = [
            np.flatnonzero(
                (np.arange(n) != s) & (adjm[s] | (block_of == block_of[s]))
            ).astype(np.int64)
            for s in range(n)
        ]
        # Each code's sorted reference codes, for searches where no rank
        # tables replace them.
        if not pool.ranked:
            self.rows = np.concatenate([self.smr, pool.srr])
            self.row_lo = np.concatenate(
                [self.smr_ptr[:-1], len(self.smr) + pool.srr_ptr[:-1]]
            )
            self.row_hi = np.concatenate(
                [self.smr_ptr[1:], len(self.smr) + pool.srr_ptr[1:]]
            )

    def draw_args(self):
        """Arguments of the scalar ``draw`` after ``j``, before the keys."""
        pool = self.pool
        return (
            self.n,
            len(self.bm_ptr) - 1,
            self.bm_ptr,
            self.bm_slot,
            pool.cr_lo,
            pool.cr_hi,
        )

    def map_args(self):
        """Arguments of the scalar ``group_map`` after ``pm``, before its scratch."""
        pool = self.pool
        return (
            self.n, self.slot_block, self.adjm, self.deg, self.kmu, pool.krr,
            self.gptr, self.grow, self.smr_ptr, self.smr, pool.srr_ptr, pool.srr,
            self.rmu, self.rcm, pool.rrr, pool.ranked,
        )  # fmt: skip

    def chunk(self):
        """Draws per vectorized chunk, bounding the per-slot arrays."""
        return _draw_chunk(self.n)

    def draws(self, start, size, k0, k1):
        """``pm`` of draws ``start, ...`` (:func:`_draws_np`)."""
        return _draws_np(self, start, size, k0, k1)

    def maps(self, pm):
        """Return the mAP of each draw in ``pm`` (:func:`_group_maps_np`)."""
        return _group_maps_np(self, pm)


def _draw_chunk(n):
    """Draws per vectorized chunk of a group of ``n`` members."""
    return int(min(2**16, max(256, _CHUNK_ELEMENTS // max(n * n, 1))))


def _uniforms_np(c0, c1, count, k0, k1):
    """Return the first ``count`` uniforms of each draw's stream, as ``next_uniform``."""
    out = np.empty((len(c0), count + count % 2))
    for blk in range((count + 1) // 2):
        out[:, 2 * blk], out[:, 2 * blk + 1] = uniform_pair(c0, c1, blk, 0, k0, k1)
    return out


def _ordered_subsets_np(u, i, k, p):
    """``ordered_subset`` of each row of uniforms ``u``, read from column ``i``.

    Returns ``(chosen, i)``: each row's ``k`` choices from ``range(p)`` and
    the column after the last uniform read.
    """
    rows = np.arange(len(u))
    chosen = np.empty((len(u), k), dtype=np.int64)
    for cnt, t in enumerate(range(p - k, p)):  # Floyd's algorithm
        r = np.minimum((u[:, i] * (t + 1)).astype(np.int64), t)
        i += 1
        if cnt:
            r = np.where((chosen[:, :cnt] == r[:, None]).any(axis=1), t, r)
        chosen[:, cnt] = r
    for z in range(k - 1, 0, -1):  # Fisher-Yates
        r = np.minimum((u[:, i] * (z + 1)).astype(np.int64), z)
        i += 1
        old = chosen[:, z].copy()
        chosen[:, z] = chosen[rows, r]
        chosen[rows, r] = old
    return chosen, i


def _draws_np(layout, start, size, k0, k1):
    """``pm`` of draws ``start, ..., start + size - 1``, one row each, as ``draw``."""
    n, bm_ptr, bm_slot = layout.n, layout.bm_ptr, layout.bm_slot
    cr_lo, cr_hi = layout.pool.cr_lo, layout.pool.cr_hi
    j = np.arange(start, start + size, dtype=np.uint64)
    ks = np.diff(bm_ptr)
    u = _uniforms_np(j & M32, j >> S32, int((2 * ks - 1).sum()), k0, k1)
    pm = np.empty((size, n), dtype=np.int64)
    i = 0
    for b, k in enumerate(ks):
        chosen, i = _ordered_subsets_np(u, i, k, k + cr_hi[b] - cr_lo[b])
        slots = bm_slot[bm_ptr[b] : bm_ptr[b + 1]]
        pm[:, slots] = np.where(
            chosen < k, slots[np.minimum(chosen, k - 1)], n + cr_lo[b] + chosen - k
        )
    return pm


def _pair_table(x, y, n, mu, cm, rr):
    """``mu[x, y]`` for member codes ``x``, else ``cm[x - n, y]`` or ``rr[x - n, y - n]``.

    The layout of the key tables (``kmu``, ``kmu.T``, ``krr``) and of the rank
    tables (``rmu``, ``rcm``, ``rrr``); ``cm`` is indexed ``(x - n, y)``.
    """
    member = x < n
    out = mu[np.minimum(x, n - 1), y]
    if rr.size:
        xr = np.maximum(x - n, 0)
        ref_y = y >= n
        out = np.where(
            member,
            out,
            np.where(ref_y, rr[xr, np.maximum(y - n, 0)], cm[xr, np.minimum(y, n - 1)]),
        )
    return out


def _full_table(n, mu, cm, rr):
    """Return the ``(code, code)`` table ``_pair_table`` reads, as one array."""
    m = n + len(rr)
    out = np.empty((m, m), dtype=mu.dtype)
    out[:n] = mu
    out[n:, :n] = cm
    out[n:, n:] = rr
    return out


def _gather(table, x, y):
    """``table[x, y]`` elementwise, as one flat take."""
    return table.ravel()[x * table.shape[1] + y]


def _count_less_np(row, lo, hi, key):
    """``count_less`` elementwise: ``lo`` plus the entries of ``row[lo:hi]`` below ``key``."""
    lo, hi = lo.copy(), hi.copy()
    last = max(len(row) - 1, 0)
    live = lo < hi
    while live.any():
        mid = (lo + hi) >> 1
        less = row[np.minimum(mid, last)] < key if len(row) else np.zeros_like(live)
        lo = np.where(live & less, mid + 1, lo)
        hi = np.where(live & ~less, mid, hi)
        live = lo < hi
    return lo


def _group_maps_np(layout, pm):
    """Return the mAP of each draw in ``pm`` (one row each), as ``group_map``."""
    n, size = layout.n, len(pm)
    pool = layout.pool
    kmu_t = np.ascontiguousarray(layout.kmu.T)  # (pool code, member) -> key
    total = np.zeros(size)
    for s in range(n):
        b = layout.slot_block[s]
        slots = layout.counted[s]
        x = pm[:, s][:, None]
        if layout.full_keys is None:
            keys = _pair_table(x, pm[:, slots], n, layout.kmu, kmu_t[n:], pool.krr)
        else:
            keys = _gather(layout.full_keys, x, pm[:, slots])
        k, t = _rank_entries(keys, slots)
        # Drawn block profiles ranked strictly before each entry's key.
        drawn = _drawn_before(k, layout.slot_block[t] == b)
        cols, yp = _positives(layout, s, t, pm)
        xp = np.broadcast_to(x, yp.shape)
        if layout.full_before is not None:
            before = _gather(layout.full_before, xp, yp)
        else:
            kp = np.take_along_axis(k, cols, axis=1)
            if layout.full_ranks is not None:
                before = _gather(layout.full_ranks, xp, yp)
            elif pool.ranked:
                before = _pair_table(xp, yp, n, layout.rmu, layout.rcm, pool.rrr)
            else:
                lo = layout.row_lo[xp]
                before = _count_less_np(layout.rows, lo, layout.row_hi[xp], kp) - lo
            # Block members x ranks before the key: count_less on its sorted g-row.
            word = (xp.astype(np.uint64) << np.uint64(32)) | kp
            before = before + (np.searchsorted(layout.gkeys, word) - layout.gptr[xp])
        total += _mean_precision(before - np.take_along_axis(drawn, cols, axis=1))
    return total / n


def _rank_entries(keys, slots):
    """``(k, t)``: each row's keys and slots, sorted by key and then slot."""
    ent = (keys.astype(np.uint64) << np.uint64(32)) | slots.astype(np.uint64)
    ent.sort(axis=1)
    return ent >> np.uint64(32), (ent & np.uint64(0xFFFFFFFF)).astype(np.int64)


def _drawn_before(k, block):
    """Entries where ``block`` holds ranked strictly before each entry's key ``k``."""
    new = np.ones(k.shape, dtype=np.bool_)
    new[:, 1:] = k[:, 1:] != k[:, :-1]
    run = np.maximum.accumulate(np.where(new, np.arange(k.shape[1]), 0), axis=1)
    return np.take_along_axis(np.cumsum(block, axis=1) - block, run, axis=1)


def _positives(layout, s, t, pm):
    """``(cols, yp)``: slot ``s``'s positives' columns in ``t`` and their codes."""
    cols = np.nonzero(layout.adjm[s][t])[1].reshape(len(t), layout.deg[s])
    return cols, np.take_along_axis(pm, np.take_along_axis(t, cols, axis=1), axis=1)


def _mean_precision(before):
    """Each row's AP, its ``d``-th positive ranked after ``before[:, d]`` negatives.

    The precisions add left to right (a cumulative sum, not a pairwise
    one), as the scalar ``group_map`` adds them.
    """
    deg = before.shape[1]
    rank = np.arange(1, deg + 1)
    return np.cumsum(rank / (rank + before), axis=1)[:, -1] / deg


def _mix(x):
    """splitmix64 finalizer: a well-spread hash of uint64s."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _scan_negatives(pairs, group_of, n_groups, n_prof):
    """Count each member's negative pairs, hash its references, flag broken groups.

    A negative must not join two members of one group. Returns ``(count, h1,
    h2, inside)``: each profile's negative pairs, two order-independent
    hashes of its partners (a set, since pairs are unique), and whether each
    group has a negative pair inside it.
    """
    a, b = pairs[:, 0], pairs[:, 1]
    ga, gb = group_of[a], group_of[b]
    inside = np.zeros(n_groups, dtype=np.bool_)
    inside[ga[(ga >= 0) & (ga == gb)]] = True
    x, y = np.concatenate([a, b]), np.concatenate([b, a]).astype(np.uint64)
    keep = group_of[x] >= 0
    x, y = x[keep], y[keep]
    count = np.bincount(x, minlength=n_prof).astype(np.int64)
    h1 = np.zeros(n_prof, dtype=np.uint64)
    h2 = np.zeros(n_prof, dtype=np.uint64)
    np.add.at(h1, x, _mix(y + np.uint64(1)))
    np.add.at(h2, x, _mix(y ^ np.uint64(0x9E3779B97F4A7C15)))
    return count, h1, h2, inside


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
    edges_of, leaves = group_edges(groups, profile, num_pos, pos_pairs)
    member = profile[rows]
    count, h1, h2, inside = _scan_negatives(neg_pairs, group_of, len(groups), n_prof)
    check_pair_counts(count[member], member, num_neg[rows], "negative")
    # Positives to profiles in no group: members dropped from the AP scores.
    ga, gb = group_of[pos_pairs[:, 0]], group_of[pos_pairs[:, 1]]
    orphaned = np.unique(np.concatenate([ga[gb < 0], gb[ga < 0]]))
    orphaned = orphaned[orphaned >= 0]
    if len(orphaned):
        raise ValueError(
            f"{len(orphaned)} group(s) have positives among profiles the AP "
            "scores no longer have (e.g. some of a group's members were "
            "dropped after average_precision). Their members' APs were "
            "computed with them as positives, so relabelling the kept members "
            "does not test those APs. Drop whole groups only, recompute "
            "average_precision on the kept profiles, or pass null='joint' or "
            "null='copied'"
        )
    bad = leaves | inside
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} group(s) have positives outside the group or "
            "negatives inside it; the permutation null needs each group's "
            "positives to be its other members and its references outside it. "
            "Group so they are, or pass null='joint', which for a group of one "
            "profile (e.g. one profile per guide, positives by gene) is that "
            "profile's AP null"
        )
    # One representative per distinct negative set; gather only their lists.
    fingerprint = np.stack([count[member], h1[member], h2[member]], axis=1)
    _, rep_ix, block_id = np.unique(
        fingerprint.astype(np.uint64), axis=0, return_index=True, return_inverse=True
    )
    block_id = block_id.ravel()
    rep_of = np.full(n_prof, -1, dtype=np.int64)
    rep_of[member[rep_ix]] = np.arange(len(rep_ix))
    x = np.concatenate([neg_pairs[:, 0], neg_pairs[:, 1]])
    y = np.concatenate([neg_pairs[:, 1], neg_pairs[:, 0]])
    rep = rep_of[x]
    keep = rep >= 0
    order = np.argsort(rep[keep], kind="stable")
    refs = y[keep][order]
    ptr = np.searchsorted(rep[keep][order], np.arange(len(rep_ix) + 1))
    ref_sets = [np.sort(refs[ptr[k] : ptr[k + 1]]) for k in range(len(rep_ix))]
    layouts = []
    offset = 0
    for g, ix in enumerate(groups):
        members = profile[ix]
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
        block_of = block_of.ravel().astype(np.int64)
        layouts.append((members, edges_of[g], blocks, block_of))
    return layouts


def check_pos_diffby(layouts, codes):
    """Raise unless relabelling keeps each group's positive pairs under ``pos_diffby``.

    ``codes[p, c]`` is profile ``p``'s value code in the APs' ``pos_diffby``
    column ``c``; two profiles are positives only if they differ in every
    column. A draw puts profiles of a block's pool (its members and
    references) into the block's slots, so the pairs it keeps are the
    group's only when, for each pair of slots, whether two drawn profiles
    differ is fixed by their blocks. Per column that holds within a block
    whose pool has one value (never differ) or no repeated value (always
    differ), and across two blocks whose pools have one value each or share
    none. A column whose profiles never differ for a pair of blocks fixes
    the pair (never positive) whatever the other columns. Layouts outside
    these rules raise, so this check is sufficient, not necessary.
    """
    codes = np.asarray(codes, dtype=np.int64)
    n_col = codes.shape[1]
    ref_uniq = {}

    def refs_stats(refs):  # per column: the references' sorted distinct values
        key = id(refs)
        if key not in ref_uniq:
            ref_uniq[key] = (refs, [np.unique(codes[refs, c]) for c in range(n_col)])
        return ref_uniq[key][1]

    def contains(sorted_vals, x):
        i = np.minimum(np.searchsorted(sorted_vals, x), max(len(sorted_vals) - 1, 0))
        return len(sorted_vals) > 0 and bool((sorted_vals[i] == x).any())

    bad = []
    for g, (members, _, blocks, block_of) in enumerate(layouts):
        stats = []  # per block, per column: (its one value or None, no repeats,
        # member values, sorted reference values)
        for b, refs in enumerate(blocks):
            mine = codes[members[block_of == b]]
            col = []
            for c, u in enumerate(refs_stats(refs)):
                m = mine[:, c]
                one = len(u) <= 1 and (m == (u[0] if len(u) else m[0])).all()
                distinct = (
                    len(u) == len(refs)
                    and len(np.unique(m)) == len(m)
                    and not contains(u, m)
                )
                col.append((m[0] if one else None, distinct, m, u))
            stats.append(col)
        sizes = np.bincount(block_of, minlength=len(blocks))
        ok = True
        for b in range(len(blocks)):
            for d in range(b, len(blocks)):
                if b == d and sizes[b] < 2:
                    continue
                fixed = []
                for c in range(n_col):
                    vb, db, mb, ub = stats[b][c]
                    vd, dd, md, ud = stats[d][c]
                    if b == d:
                        rel = False if vb is not None else (True if db else None)
                    elif vb is not None and vd is not None:
                        rel = bool(vb != vd)
                    elif not (
                        np.intersect1d(ub, ud).size
                        or contains(ud, mb)
                        or contains(ub, md)
                        or np.isin(mb, md).any()
                    ):
                        rel = True
                    else:
                        rel = None
                    fixed.append(rel)
                ok &= False in fixed or all(fixed)
        if not ok:
            bad.append(g)
    if bad:
        raise ValueError(
            f"{len(bad)} group(s) (e.g. group {bad[0]}) rank references whose "
            "pos_diffby values do not fix which relabelled profiles are "
            "positives: a draw could pair profiles that share a pos_diffby "
            "value, or split profiles that do not, unlike the group's "
            "members. The permutation null keeps the members' positive "
            "pairs, so it would test the wrong rule. Block references by "
            "those columns (e.g. neg_sameby), or pass null='joint' or "
            "null='copied'"
        )


def group_key(seed: int, members) -> tuple[np.uint64, np.uint64]:
    """Philox key of a group's draws: a pure function of ``(seed, members)``."""
    return stream_key(seed, tuple(sorted(int(m) for m in members)))


def _label_key(seed: int, members, label) -> tuple[np.uint64, np.uint64]:
    """Philox key of a multilabel group's draws: a pure function of ``(seed, members, label)``.

    ``label``, the index of the group's label, gives groups of the same
    profiles but different labels independent draws.
    """
    members = tuple(sorted(int(m) for m in members))
    text = f"{(int(seed), members)!r}/{int(label)}"
    digest = hashlib.blake2b(text.encode(), digest_size=8)
    word = int.from_bytes(digest.digest(), "little")
    return np.uint64(word & 0xFFFFFFFF), np.uint64(word >> 32)


class _Run:
    """One group's draws on its layout (``_Layout`` or ``_LabelLayout``).

    ``key`` is the Philox key of the group's draws. ``identity`` is the
    observed labels' mAP, recomputed from the pairs, and ``thr`` its
    :func:`~copairs.nulls.common.tie_thresholds`.
    """

    def __init__(self, layout, key):
        self.layout = layout
        self.key = key
        self.identity = float(layout.maps(layout.identity[None])[0])
        self.thr = float(tie_thresholds(self.identity))

    def maps_of(self, start, size):
        """Return the mAPs of draws ``start, ..., start + size - 1``."""
        return self.layout.maps(self.layout.draws(start, size, *self.key))

    def count(self, null_size, stop_after=None, out=None):
        """``(count, draws)`` of up to ``null_size`` draws, as :func:`_count_hits`."""
        return _count_hits(
            self.maps_of, null_size, self.thr, self.layout.chunk(), stop_after, out
        )


def permutation_null(
    members, edges, blocks, block_of, similarity, null_size, seed, pool=None,
    out=None,
):  # fmt: skip
    """Run ``null_size`` label permutations of one group.

    Returns ``(count, identity, pool)``: the draws whose mAP ties or exceeds
    the observed labels' (``identity``, within
    :func:`~copairs.nulls.common.tie_thresholds`' tolerance), that mAP,
    and the reference pool, reusable by groups ranking the same references.
    ``out`` receives the draws' mAPs if given.
    """
    if pool is None or not pool.matches(blocks):
        pool = _Pool(blocks, similarity)
    layout = _Layout(members, edges, block_of, similarity, pool, int(null_size))
    run = _Run(layout, group_key(seed, members))
    count, _ = run.count(int(null_size), out=out)
    return count, run.identity, pool


class _Hits:
    """A group's draws that reach its mAP, counted as their mAPs arrive.

    Without ``stop_after`` all ``null_size`` draws run, in chunks of
    ``chunk`` that may arrive in any order. With it, rounds start at
    :data:`_FIRST_ROUND` draws and double up to ``chunk``, one round at a
    time, and the run stops at its ``stop_after``-th hit: ``draws`` is then
    the draw of that hit. Draw ``j`` does not depend on the chunks or
    rounds, so neither does the result.
    """

    def __init__(self, null_size, thr, chunk, stop_after=None):
        self.null_size = null_size
        self.thr = thr
        self.chunk = chunk
        self.stop_after = stop_after
        self.size = chunk if stop_after is None else min(_FIRST_ROUND, chunk)
        self.count = 0  # hits so far
        self.draws = null_size  # draws the hits came from
        self.issued = 0  # draws handed out by tasks()
        self.arrived = 0  # draws whose mAPs were added
        self.done = null_size == 0

    def tasks(self):
        """``(start, size)`` of the draws to run next: every chunk, or one round."""
        if self.stop_after is None:
            todo = list(chunks(self.null_size, self.chunk))
        else:
            todo = [(self.issued, min(self.size, self.null_size - self.issued))]
            self.size = min(2 * self.size, self.chunk)
        self.issued = sum(todo[-1])
        return todo

    def add(self, start, maps):
        """Count the mAPs of draws ``start, ...``; return whether the run is done."""
        hits = maps >= self.thr
        if self.stop_after is not None:
            seen = np.cumsum(hits)
            if self.count + seen[-1] >= self.stop_after:
                at = int(np.searchsorted(seen, self.stop_after - self.count))
                self.count, self.draws = self.stop_after, start + at + 1
                self.done = True
                return True
        self.count += int(hits.sum())
        self.arrived += len(maps)
        self.done = self.arrived == self.null_size
        return self.done

    def pvalue(self):
        """Return the run's p-value (:func:`_sequential_pvalue`)."""
        return _sequential_pvalue(
            self.count, self.draws, self.null_size, self.stop_after
        )


def _count_hits(maps_of, null_size, thr, chunk, stop_after=None, out=None):
    """Count a group's draws whose mAP reaches ``thr``, one task after another.

    ``maps_of(start, size)`` returns the mAPs of draws ``start, ...,
    start + size - 1``; chunks, rounds and ``stop_after`` as in
    :class:`_Hits`. Returns ``(count, draws)``: the hits and the draws they
    came from. ``out`` receives the mAPs of the draws run.
    """
    hits = _Hits(null_size, thr, chunk, stop_after)
    while not hits.done:
        for start, size in hits.tasks():
            maps = maps_of(start, size)
            if out is not None:
                out[start : start + size] = maps
            hits.add(start, maps)
    return hits.count, hits.draws


def _sequential_pvalue(count, draws, null_size, stop_after=None) -> float:
    """P-value of a group's run of draws (:meth:`_Run.count`).

    ``(count + 1) / (null_size + 1)`` after all draws, and ``count / draws``
    when the run stopped at its ``stop_after``-th hit (Besag and Clifford,
    1991, Biometrika 78:301). Both are valid p-values; stopping spends few
    draws on groups with large p-values and none of the resolution of small
    ones, which run all draws.
    """
    if stop_after is not None and count >= stop_after:
        return count / draws
    return (count + 1) / (null_size + 1)


def check_stop_after(stop_after):
    """Return ``stop_after`` as an int, or None; raise unless a positive integer."""
    if stop_after is None:
        return None
    if int(stop_after) != stop_after or stop_after < 1:
        raise ValueError(f"stop_after must be a positive integer, got {stop_after}")
    return int(stop_after)


def _pool_bytes(blocks):
    """Estimated bytes of the ``_Pool`` of ``blocks``: its key and rank tables."""
    r = sum(len(b) for b in blocks)
    rows = sum(len(b) * max(len(b) - 1, 0) for b in blocks)  # sorted rows
    ranked = 8 * rows + 4 * r * r if r <= RANK_POOL else 0  # words, rank table
    return 4 * r * r + 4 * rows + ranked


class _SharedPools:
    """The pools of a set of groups, each built once and freed after its last group.

    Groups ranking the same references share one read-only pool, so threads
    can run them concurrently. Only the thread that schedules the groups
    (:func:`_schedule`) uses this: it starts a group whose pool is new once
    ``reserve`` finds the pool's estimated bytes fit in ``POOL_BUDGET``
    beside the pools alive (or none is alive), and ``release`` frees a pool
    after its last group. ``keys[g]`` names group ``g``'s pool and
    ``nbytes(g)`` estimates its bytes.
    """

    def __init__(self, keys, nbytes):
        self.key = list(keys)
        self.nbytes = nbytes
        self.left = Counter(self.key)
        self.pools = {}
        self.bytes = {}
        self.used = 0

    def reserve(self, g):
        """Count group ``g``'s new pool in the budget; False if it does not fit yet."""
        need = self.nbytes(g)
        if self.used and self.used + need > POOL_BUDGET:
            return False
        self.used += need
        self.bytes[self.key[g]] = need
        return True

    def release(self, g):
        """Free group ``g``'s pool if no other group needs it."""
        key = self.key[g]
        self.left[key] -= 1
        if self.left[key]:
            return
        self.pools.pop(key, None)
        self.used -= self.bytes.pop(key, 0)


def _check_identity(g, identity, observed):
    """Raise if group ``g``'s recomputed mAP ``identity`` differs from ``observed[g]``."""
    if observed is not None and abs(identity - observed[g]) > 1e-6:
        raise ValueError(
            f"group {g}: mAP {identity} from the pairs differs from "
            f"{observed[g]} in the AP table; the pairs or similarity do not "
            "match the AP scores (keep the AP frame's index: no reset_index)"
        )


def permutation_pvalues(
    groups, profile, num_pos, num_total, pos_pairs, neg_pairs, similarity,
    null_size, seed, observed=None, progress_bar=False, max_workers=None,
    pos_diffby=None, stop_after=None,
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
        Draws per group and seed. ``p = (1 + #{draw >= thr}) /
        (null_size + 1)``, with ``thr`` the
        :func:`~copairs.nulls.common.tie_thresholds` of the observed labels'
        mAP, recomputed from the pairs.
    observed : np.ndarray, optional
        Each group's mAP from the AP table; a group whose recomputed mAP
        differs by more than 1e-6 raises, since then its pairs or similarity
        do not match the table.
    progress_bar : bool
        Whether to show a progress bar over groups.
    max_workers : int, optional
        Threads that run the groups' draws (default: ``ThreadPoolExecutor``'s),
        a group's chunks of draws on several threads at once, and at most this
        many groups at a time. Each group's p-value depends only on its own
        draws, so results do not depend on it. The pools built at once hold
        at most ``POOL_BUDGET`` (4 GiB) of estimated key and rank tables, or
        one pool if it alone is larger.
    pos_diffby : np.ndarray, optional
        ``(profiles, columns)`` integer codes of the APs' ``pos_diffby``
        columns, indexed like the pairs; layouts whose draws could break
        that rule raise (:func:`check_pos_diffby`). Not checked when omitted.
    stop_after : int, optional
        Stop a group's draws at its ``stop_after``-th draw that reaches its
        mAP, with ``p = stop_after / draws`` (Besag and Clifford's
        sequential p-value). Default: run all draws.

    Returns
    -------
    np.ndarray
        float64 p-values.
    """
    null_size = check_null_size(null_size)
    seed = resolve_seed(seed)
    stop_after = check_stop_after(stop_after)
    profile = np.asarray(profile, dtype=np.int64)
    num_pos = np.asarray(num_pos, dtype=np.int64)
    num_neg = np.asarray(num_total, dtype=np.int64) - num_pos
    groups = [np.asarray(ix, dtype=np.int64) for ix in groups]
    layouts = group_layouts(groups, profile, num_pos, num_neg, pos_pairs, neg_pairs)
    if pos_diffby is not None and np.shape(pos_diffby)[1]:
        check_pos_diffby(layouts, pos_diffby)

    keys = [_pool_key(blocks) for _, _, blocks, _ in layouts]
    refs = {}  # each pool's references, for one key table
    for key, (_, _, blocks, _) in zip(keys, layouts):
        if key not in refs:
            refs[key] = np.concatenate(blocks)

    def build(g, table):
        return _Pool(layouts[g][2], similarity, table)

    def start(g, pool):
        members, edges, _, block_of = layouts[g]
        layout = _Layout(members, edges, block_of, similarity, pool, null_size)
        run = _Run(layout, group_key(seed, members))
        _check_identity(g, run.identity, observed)
        return run

    return _run_groups(
        keys, refs, similarity, build, lambda g: _pool_bytes(layouts[g][2]), start,
        null_size, stop_after, progress_bar, max_workers,
    )  # fmt: skip


def _run_groups(
    keys, refs, similarity, build, nbytes, start, null_size, stop_after,
    progress_bar, max_workers,
):  # fmt: skip
    """P-value of each group of a permutation null, its draws run on threads.

    ``keys[g]`` names group ``g``'s pool and ``refs`` the profiles of each
    named pool. ``build(g, table)`` builds group ``g``'s pool, slicing its
    keys from ``table`` (a ``_KeyTable`` of all pools' profiles, or None),
    ``nbytes(g)`` estimates the pool's bytes (:class:`_SharedPools`), and
    ``start(g, pool)`` returns the group's ``_Run``.
    """
    if max_workers is None:  # ThreadPoolExecutor's default
        max_workers = min(32, (_cpu_count() or 1) + 4)
    pvals = np.empty(len(keys), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        try:
            # Pools ranking overlapping profiles slice one key table.
            table = _key_table(list(refs.values()), similarity, ex.map)
            done = _schedule(
                ex, max_workers, keys, lambda g: build(g, table), nbytes, start,
                null_size, stop_after,
            )  # fmt: skip
            for g, p in progress(done, len(keys), progress_bar, "permutation null"):
                pvals[g] = p
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    return pvals


def _schedule(ex, workers, keys, build, nbytes, start, null_size, stop_after):
    """Yield ``(g, p_value)`` as groups finish, running their steps on ``ex``.

    A group's pool is built once (``build(g)``), the group starts on it
    (``start(g, pool)`` gives its ``_Run``), and its draws run as chunk
    tasks, or round by round with ``stop_after`` (:class:`_Hits`), so the
    draws of a large group spread over the threads. Only this thread submits
    tasks, and no task waits for another: a group starts once its pool is
    built, and a group with a new pool only once ``_SharedPools.reserve``
    admits it. Groups start pool by pool, at most ``workers`` at a time, so
    every pool alive has unfinished groups, which free it, and few layouts
    are alive at once.
    """
    pools = _SharedPools(keys, nbytes)
    order = sorted(range(len(keys)), key=lambda g: keys[g])
    tasks = {}  # future -> (step, group, first draw)
    waiting = {}  # pool key -> groups waiting for its build
    runs = {}  # group -> (_Run, _Hits)
    nxt = active = 0

    def draw(g):
        run, hits = runs[g]
        for first, size in hits.tasks():
            tasks[ex.submit(run.maps_of, first, size)] = ("draws", g, first)

    while nxt < len(order) or tasks:
        while nxt < len(order) and active < workers:
            g = order[nxt]
            key = keys[g]
            if key in waiting:
                waiting[key].append(g)
            elif key in pools.pools:
                tasks[ex.submit(start, g, pools.pools[key])] = ("start", g, 0)
            elif pools.reserve(g):
                waiting[key] = [g]
                tasks[ex.submit(build, g)] = ("build", g, 0)
            else:  # wait until a pool is freed
                break
            nxt += 1
            active += 1
        finished, _ = wait(tasks, return_when=FIRST_COMPLETED)
        for future in finished:
            step, g, first = tasks.pop(future)
            result = future.result()
            if step == "build":
                pools.pools[keys[g]] = result
                for h in waiting.pop(keys[g]):
                    tasks[ex.submit(start, h, result)] = ("start", h, 0)
                continue
            if step == "start":
                runs[g] = (
                    result,
                    _Hits(null_size, result.thr, result.layout.chunk(), stop_after),
                )
                done = runs[g][1].done
                if not done:
                    draw(g)
            else:
                done = runs[g][1].add(first, result)
                if not done and stop_after is not None:
                    draw(g)  # the next round
            if done:
                _, hits = runs.pop(g)
                pools.release(g)
                active -= 1
                yield g, hits.pvalue()


# Multilabel groups: a draw moves one label to other profiles.


def _make_label_kernels(philox, uniform, jit=lambda f: f):
    """Build the scalar draw and mAP functions of a multilabel group.

    As :func:`_make_kernels`, with the pool's profiles as codes, members
    included. Returns ``(draw_pool, label_group_map)``.
    """
    count_less = _make_count_less(jit)
    sort_entries = make_head_sort(_INSERTION, jit)
    ordered_subset = _make_ordered_subset(philox, uniform, jit)

    @jit
    def label_group_map(
        pm, n, adjm, deg, keys, rank_table, ranked, nptr, nrow, nbits, loc, lone,
        gptr, grow, ent, rank, inb, term,
    ):  # fmt: skip
        """Return the mAP of a multilabel group whose slot ``s`` holds pool code ``pm[s]``.

        As ``group_map``: the negatives of the profile ``x`` in slot ``s``
        are its own negatives (bits of ``nbits``, sorted codes in ``nrow``)
        plus the profiles that share only the moved label with it (``lone``,
        among the group's members; sorted codes in ``grow``), minus the
        drawn ones. ``inb`` marks the drawn profiles among them.
        """
        total = 0.0
        for s in range(n):
            x = pm[s]
            lx = loc[x]
            for t in range(n):
                y = pm[t]
                ent[t] = (np.uint64(keys[x, y]) << np.uint64(32)) | np.uint64(t)
                if ranked:
                    rank[t] = rank_table[x, y]
                ly = loc[y]
                negative = (nbits[x, y >> 6] >> np.uint64(y & 63)) & np.uint64(1)
                inb[t] = negative != 0 or (lx >= 0 and ly >= 0 and lone[lx, ly])
            size = 0
            for t in range(n):  # keep the slots that count, in place
                ent[size] = ent[t]
                size += t != s and (adjm[s, t] or inb[t])
            sort_entries(ent, size)
            lo, hi = nptr[x], nptr[x + 1]
            if lx >= 0:
                g0, g1 = gptr[lx], gptr[lx + 1]
            else:
                g0, g1 = 0, 0
            g = g0
            d = 0
            drawn = 0  # drawn negatives ranked strictly before ``key``
            tied = 0  # drawn negatives tied with ``key`` so far
            key = np.uint64(0)
            for e in range(size):
                k = ent[e] >> np.uint64(32)
                t = np.int64(ent[e] & np.uint64(0xFFFFFFFF))
                if e == 0 or k != key:
                    drawn += tied
                    tied = 0
                    key = k
                    while g < g1 and grow[g] < key:
                        g += 1
                if adjm[s, t]:
                    if ranked:
                        before = np.int64(rank[t])
                    else:
                        before = count_less(nrow, lo, hi, key) - lo
                    before += g - g0 - drawn
                    term[d] = (d + 1) / (d + 1 + before)
                    d += 1
                tied += inb[t]
            acc = 0.0
            for d in range(deg[s]):
                acc += term[d]
            total += acc / deg[s]
        return total / n

    @jit
    def draw_pool(j, n_block, bm_ptr, bm_slot, cr_lo, cr_hi, k0, k1, pm, chosen):
        """Fill ``pm`` with draw ``j``.

        In each block (a component of the negative graph), the members' slots
        get a uniform random subset of the block's profiles in random order.
        """
        c0 = np.uint64(j) & M32
        c1 = np.uint64(j) >> S32
        blk = 0
        spare = -1.0
        for b in range(n_block):
            k = bm_ptr[b + 1] - bm_ptr[b]
            p = cr_hi[b] - cr_lo[b]
            blk, spare = ordered_subset(k, p, c0, c1, blk, spare, k0, k1, chosen)
            for z in range(k):
                pm[bm_slot[bm_ptr[b] + z]] = cr_lo[b] + chosen[z]

    return draw_pool, label_group_map


# The scalar functions in Python: the definition of every multilabel draw.
_draw_pool, _label_group_map = _make_label_kernels(philox4x32, uniform53)


def _components(n_prof, pairs):
    """Component of each profile in the graph of ``pairs``: its smallest profile."""
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n_prof, n_prof),
    )
    _, labels = connected_components(graph, directed=False)
    _, smallest = np.unique(labels, return_index=True)
    return smallest[labels].astype(np.int64)


def _adjacency(n_prof, pairs):
    """CSR ``(ptr, partners)`` of each profile's partners in ``pairs``."""
    x = np.concatenate([pairs[:, 0], pairs[:, 1]])
    y = np.concatenate([pairs[:, 1], pairs[:, 0]])
    ptr = np.zeros(n_prof + 1, dtype=np.int64)
    np.cumsum(np.bincount(x, minlength=n_prof), out=ptr[1:])
    return ptr, y[np.argsort(x, kind="stable")].astype(np.int64)


def _ranges(lo, hi):
    """``arange(lo[i], hi[i])`` for each ``i``, concatenated."""
    size = hi - lo
    return np.repeat(lo - np.cumsum(size) + size, size) + np.arange(size.sum())


def _adjacency_matrix(n, pairs):
    """``(n, n)`` symmetric bool matrix of the local ``pairs``."""
    adjm = np.zeros((n, n), dtype=np.bool_)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    adjm[pairs[:, 0], pairs[:, 1]] = True
    adjm[pairs[:, 1], pairs[:, 0]] = True
    return adjm


class _LabelPool:
    """Rank codes among whole components of the negative graph.

    Codes are the profiles' positions in the pool, members included. Keeps the
    codes between all pool profiles, each profile's sorted codes to its
    negatives (all inside its component) with their bitset, and, up to
    ``RANK_POOL`` profiles, each profile's rank of every other one among its
    negatives.
    """

    def __init__(self, comps, comp_profiles, nbr_ptr, nbr, similarity, table=None):
        self.comps = comps
        parts = [comp_profiles[c] for c in comps]
        self.profiles = np.concatenate(parts).astype(np.int64)
        p = len(self.profiles)
        if p > MAX_POOL:
            raise ValueError(
                f"a group pool has {p} profiles; the permutation null supports up "
                f"to {MAX_POOL}. Split profiles into blocks (neg_sameby) or rank "
                "fewer of them."
            )
        size = np.cumsum([0] + [len(c) for c in parts])
        self.cr_lo, self.cr_hi = size[:-1].astype(np.int64), size[1:].astype(np.int64)
        self.code = np.full(len(nbr_ptr) - 1, -1, dtype=np.int64)
        self.code[self.profiles] = np.arange(p)
        if table is None:
            self.keys = _key_matrix(similarity, self.profiles)
        else:
            self.keys = table.matrix(self.profiles)
        lo, hi = nbr_ptr[self.profiles], nbr_ptr[self.profiles + 1]
        self.nptr = np.concatenate([[0], np.cumsum(hi - lo)]).astype(np.int64)
        self.nrow = np.empty(self.nptr[-1], dtype=np.uint32)
        width = (p + 63) // 64
        self.nbits = np.zeros((p, width), dtype=np.uint64)
        for a, m in chunks(p, max(1, _PAIR_CHUNK // max(p, 1))):
            rows = np.repeat(np.arange(m), hi[a : a + m] - lo[a : a + m])
            codes = self.code[nbr[_ranges(lo[a : a + m], hi[a : a + m])]]
            words = _row_words(
                self.nptr[a : a + m + 1] - self.nptr[a], self.keys[a + rows, codes]
            )
            words.sort()
            self.nrow[self.nptr[a] : self.nptr[a + m]] = words & M32
            dense = np.zeros((m, 64 * width), dtype=np.bool_)
            dense[rows, codes] = True
            bits = np.packbits(dense, axis=1, bitorder="little")
            self.nbits[a : a + m] = bits.view("<u8")
        self.ranked = p <= RANK_POOL
        self.rank = np.zeros((0, 0), dtype=np.int32)
        if self.ranked:
            words = _row_words(self.nptr, self.nrow)
            self.rank = _row_ranks(self.nptr, words, self.keys)


def _label_pool_bytes(p, n_neg):
    """Estimated bytes of a ``_LabelPool`` of ``p`` profiles with ``n_neg`` negatives."""
    ranked = 4 * p * p + 8 * n_neg if p <= RANK_POOL else 0  # rank table, words
    return 4 * p * p + p * p // 8 + 4 * n_neg + ranked


class _LabelLayout:
    """One multilabel group's slots and codes, as the label draw functions take them."""

    def __init__(self, members, edges, lone, block_of, pool, draws=0):
        self.n = n = len(members)
        self.pool = pool
        self.member_code = pool.code[np.asarray(members, dtype=np.int64)]
        self.identity = self.member_code  # the observed labels' codes
        block_of = np.asarray(block_of, dtype=np.int64)
        self.bm_slot = np.argsort(block_of, kind="stable").astype(np.int64)
        bm_ptr = np.searchsorted(block_of[self.bm_slot], np.arange(len(pool.comps) + 1))
        self.bm_ptr = bm_ptr.astype(np.int64)
        self.adjm = _adjacency_matrix(n, edges)
        self.deg = self.adjm.sum(axis=1).astype(np.int64)
        self.lone = _adjacency_matrix(n, lone)
        self.loc = np.full(len(pool.profiles), -1, dtype=np.int64)
        self.loc[self.member_code] = np.arange(n)
        # Each member's codes to the members sharing only the moved label, sorted.
        codes = pool.keys[np.ix_(self.member_code, self.member_code)]
        rows = [np.sort(codes[i, self.lone[i]]) for i in range(n)]
        self.gptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
        self.grow = np.concatenate(rows).astype(np.uint32)
        # The rows as one sorted array of (member, code) words, for one searchsorted.
        self.gwords = _row_words(self.gptr, self.grow)
        # A positive y of a drawn profile that is member i ranks after
        # lone_before[i, y] of i's lone partners (row n: other profiles have
        # none). The table is built when ``draws`` draws gather more
        # positives than it has entries; else the counts are searched in
        # gwords.
        self.lone_before = None
        p = len(pool.profiles)
        if (n + 1) * p <= min(_FULL_TABLE, draws * int(self.deg.sum())):
            word = (
                np.arange(n, dtype=np.uint64)[:, None] << np.uint64(32)
            ) | pool.keys[self.member_code].astype(np.uint64)
            self.lone_before = np.zeros((n + 1, p), dtype=np.int32)
            self.lone_before[:n] = (
                np.searchsorted(self.gwords, word) - self.gptr[:-1, None]
            )

    def draw_args(self):
        """Arguments of the scalar ``draw_pool`` after ``j``, before the keys."""
        pool = self.pool
        return len(self.bm_ptr) - 1, self.bm_ptr, self.bm_slot, pool.cr_lo, pool.cr_hi

    def map_args(self):
        """Arguments of the scalar ``label_group_map`` after ``pm``, before its scratch."""
        pool = self.pool
        return (
            self.n, self.adjm, self.deg, pool.keys, pool.rank, pool.ranked, pool.nptr,
            pool.nrow, pool.nbits, self.loc, self.lone, self.gptr, self.grow,
        )  # fmt: skip

    def chunk(self):
        """Draws per vectorized chunk, bounding the per-slot arrays."""
        return _draw_chunk(self.n)

    def draws(self, start, size, k0, k1):
        """``pm`` of draws ``start, ...`` (:func:`_draws_pool_np`)."""
        return _draws_pool_np(self, start, size, k0, k1)

    def maps(self, pm):
        """Return the mAP of each draw in ``pm`` (:func:`_label_maps_np`)."""
        return _label_maps_np(self, pm)


def _draws_pool_np(layout, start, size, k0, k1):
    """``pm`` of draws ``start, ..., start + size - 1``, one row each, as ``draw_pool``."""
    pool, bm_ptr, bm_slot = layout.pool, layout.bm_ptr, layout.bm_slot
    j = np.arange(start, start + size, dtype=np.uint64)
    ks = np.diff(bm_ptr)
    u = _uniforms_np(j & M32, j >> S32, int((2 * ks - 1).sum()), k0, k1)
    pm = np.empty((size, layout.n), dtype=np.int64)
    i = 0
    for b, k in enumerate(ks):
        chosen, i = _ordered_subsets_np(u, i, k, pool.cr_hi[b] - pool.cr_lo[b])
        pm[:, bm_slot[bm_ptr[b] : bm_ptr[b + 1]]] = pool.cr_lo[b] + chosen
    return pm


def _label_maps_np(layout, pm):
    """Return the mAP of each draw in ``pm`` (one row each), as ``label_group_map``."""
    n, size = layout.n, len(pm)
    pool = layout.pool
    total = np.zeros(size)
    inb = np.zeros((size, n), dtype=np.bool_)
    for s in range(n):
        slots = np.flatnonzero(np.arange(n) != s)
        x = pm[:, s][:, None]
        y = pm[:, slots]
        lx, ly = layout.loc[x], layout.loc[y]
        negative = (_gather(pool.nbits, x, y >> 6) >> (y & 63).astype(np.uint64)) & 1
        lone = (lx >= 0) & (ly >= 0)
        lone &= _gather(layout.lone, np.maximum(lx, 0), np.maximum(ly, 0))
        inb[:, slots] = (negative != 0) | lone
        keys = _gather(pool.keys, x, y)
        k, t = _rank_entries(keys, slots)
        # Drawn negatives ranked strictly before each entry's key.
        drawn = _drawn_before(k, np.take_along_axis(inb, t, axis=1))
        cols, yp = _positives(layout, s, t, pm)
        xp = np.broadcast_to(x, yp.shape)
        kp = np.take_along_axis(k, cols, axis=1)
        if pool.ranked:
            before = _gather(pool.rank, xp, yp)
        else:
            lo = pool.nptr[xp]
            before = _count_less_np(pool.nrow, lo, pool.nptr[xp + 1], kp) - lo
        # Members sharing only the label ranked before the key: x's sorted g-row.
        if layout.lone_before is not None:
            before = before + _gather(layout.lone_before, np.where(lx >= 0, lx, n), yp)
        else:
            lxp = np.broadcast_to(np.maximum(lx, 0), kp.shape)
            word = (lxp.astype(np.uint64) << np.uint64(32)) | kp
            lone_before = np.searchsorted(layout.gwords, word) - layout.gptr[lxp]
            before = before + np.where(lx >= 0, lone_before, 0)
        total += _mean_precision(before - np.take_along_axis(drawn, cols, axis=1))
    return total / n


def _segments(key, n_keys):
    """``(order, ptr)``: pair indices sorted by key, and each key's slice."""
    order = np.argsort(key, kind="stable")
    return order, np.searchsorted(key[order], np.arange(n_keys + 1))


def _label_layouts(groups, rows, num_pos, num_neg, pairs):
    """Members, edges, lone pairs and components of each multilabel group.

    Parameters
    ----------
    groups : list of np.ndarray
        Each group's positions in ``rows``; a group holds one label.
    rows : np.ndarray
        The AP rows (indices of ``pairs.profile`` and ``pairs.row_key``).
    num_pos, num_neg : np.ndarray
        Each row's positive and negative pair counts, checked against
        ``pairs``.
    pairs : copairs.map.LabelPairs
        The pairs the APs were computed from.

    Returns
    -------
    layouts : list of tuple
        One ``(members, edges, lone, comps, block_of, key)`` per group: its
        profiles, its positive and lone pairs between local member indices,
        the components of the negative graph holding its members, each
        member's component (index into ``comps``) and its label's index.
    comp_profiles : dict
        The sorted profiles of every component, by component.
    neighbours : tuple of np.ndarray
        CSR ``(ptr, partners)`` of each profile's negatives.
    """
    if pairs.lone is None:
        raise ValueError(
            f"the permutation null of multilabel APs needs {pairs.label_col!r} "
            "among the negative pairs' diffby columns (and not their sameby)"
        )
    if len(rows) and rows.max() >= len(pairs.profile):
        raise ValueError(
            f"the AP scores' index reaches row {rows.max()}, but the frame "
            "multilabel.average_precision returned with these pairs has "
            f"{len(pairs.profile)} rows; pass the pairs of the frame these "
            "scores come from"
        )
    profile_of = np.asarray(pairs.profile, dtype=np.int64)
    row_key = np.asarray(pairs.row_key, dtype=np.int64)
    profile, label_of = profile_of[rows], row_key[rows]
    # Positive pairs of AP rows as pairs of profiles, with their labels.
    pos_rows = np.asarray(pairs.pos, dtype=np.int64).reshape(-1, 2)
    pos, pos_key = profile_of[pos_rows], row_key[pos_rows[:, 0]]
    neg = np.asarray(pairs.neg, dtype=np.int64).reshape(-1, 2)
    lone = np.asarray(pairs.lone, dtype=np.int64).reshape(-1, 2)
    n_prof = 1 + max(
        int(profile.max(initial=-1)),
        int(pos.max(initial=-1)),
        int(neg.max(initial=-1)),
        int(lone.max(initial=-1)),
    )
    nbr_ptr, nbr = _adjacency(n_prof, neg)
    comp = _components(n_prof, neg)
    order = np.argsort(comp, kind="stable")
    roots, starts = np.unique(comp[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    comp_profiles = {
        int(c): order[a:b] for c, a, b in zip(roots, starts, ends, strict=True)
    }
    labels = np.asarray(pairs.keys).tolist()
    n_keys = len(labels)
    pos_order, pos_ptr = _segments(pos_key, n_keys)
    lone_order, lone_ptr = _segments(np.asarray(pairs.lone_key), n_keys)
    local = np.full(n_prof, -1, dtype=np.int64)
    layouts = []
    for ix in groups:
        values = np.unique(label_of[ix])
        if len(values) != 1:
            raise ValueError(
                "the permutation null of multilabel APs needs each group to hold "
                f"one {pairs.label_col!r} value; group by it"
            )
        key = int(values[0])
        label = labels[key]
        members = profile[ix]
        n = len(members)
        if len(np.unique(members)) != n:
            raise ValueError(f"group {label!r}: its rows must be distinct profiles")
        local[members] = np.arange(n)
        layout = [members]
        for found, order_, ptr in (
            (pos, pos_order, pos_ptr),
            (lone, lone_order, lone_ptr),
        ):
            seg = found[order_[ptr[key] : ptr[key + 1]]]
            ends_ = local[seg]
            if (ends_ < 0).any():
                local[members] = -1
                missing = np.unique(seg[ends_ < 0])
                raise ValueError(
                    f"group {label!r}: {len(missing)} profile(s) holding its "
                    f"label (e.g. profile {missing[0]}) have no row for it in the "
                    "group: some of the label's AP rows were dropped after "
                    "multilabel.average_precision or grouped apart, or those "
                    "profiles have no positive pairs for it. The permutation "
                    "null moves the label among all profiles of its components, "
                    "so it needs every profile holding it in the group. Group "
                    "by the label alone and keep all of a label's rows, or pass "
                    "null='joint' or null='copied'"
                )
            layout.append(ends_)
        degree = np.bincount(layout[1].ravel(), minlength=n)
        local[members] = -1
        n_neg = nbr_ptr[members + 1] - nbr_ptr[members]
        check_pair_counts(degree, members, num_pos[ix], "positive", label)
        check_pair_counts(n_neg, members, num_neg[ix], "negative", label)
        comps, block_of = np.unique(comp[members], return_inverse=True)
        layout += [tuple(int(c) for c in comps), block_of.ravel().astype(np.int64), key]
        layouts.append(tuple(layout))
    return layouts, comp_profiles, (nbr_ptr, nbr)


def _label_run(members, edges, lone, comps, block_of, key, pool, seed, draws=0):
    """Return a multilabel group's ``_Run`` on ``pool``, the ``_LabelPool`` of its ``comps``.

    ``draws`` is the number of draws it will run (:class:`_LabelLayout`).
    """
    layout = _LabelLayout(members, edges, lone, block_of, pool, draws)
    return _Run(layout, _label_key(seed, members, key))


def label_permutation_pvalues(
    groups, rows, num_pos, num_total, pairs, null_size, seed, observed=None,
    progress_bar=False, max_workers=None, stop_after=None,
):  # fmt: skip
    """P-value of each multilabel group's mAP against moves of its label.

    Under the null hypothesis the profiles holding a label are a random set of
    their components' profiles (components of the graph of negative pairs). A
    draw moves the group's label to as many profiles of each component as
    hold it there, keeping every other label, and recomputes the group's mAP
    from the observed similarities: a profile that receives the label ranks
    the other receivers (positives) against its own negatives and the
    profiles that share only this label with it, minus the receivers.

    Parameters
    ----------
    groups : list of np.ndarray
        Each group's rows of the AP table; a group holds one label.
    rows : np.ndarray
        The AP table's index (rows of ``pairs.profile``), which also gives
        each row's label (``pairs.row_key``).
    num_pos, num_total, null_size, seed, observed, progress_bar, max_workers, stop_after
        As in :func:`permutation_pvalues`; pools are components' profiles.
    pairs : copairs.map.LabelPairs
        The pairs the APs were computed from. Layouts whose draws could
        break their ``pos_diffby`` rule raise (:func:`check_pos_diffby`).

    Returns
    -------
    np.ndarray
        float64 p-values.
    """
    null_size = check_null_size(null_size)
    seed = resolve_seed(seed)
    stop_after = check_stop_after(stop_after)
    num_pos = np.asarray(num_pos, dtype=np.int64)
    num_neg = np.asarray(num_total, dtype=np.int64) - num_pos
    groups = [np.asarray(ix, dtype=np.int64) for ix in groups]
    layouts, comp_profiles, (nbr_ptr, nbr) = _label_layouts(
        groups, np.asarray(rows, dtype=np.int64), num_pos, num_neg, pairs
    )
    if pairs.pos_diffby is not None and np.shape(pairs.pos_diffby)[1]:
        # A draw fills a component's slots from all its profiles: the
        # members and, as references, the rest.
        def as_single(members, edges, lone, comps, block_of, key):
            refs = [np.setdiff1d(comp_profiles[c], members) for c in comps]
            return members, edges, refs, block_of

        check_pos_diffby([as_single(*lay) for lay in layouts], pairs.pos_diffby)
    degree = np.diff(nbr_ptr)

    def pool_bytes(g):
        parts = [comp_profiles[c] for c in layouts[g][3]]
        p = sum(len(part) for part in parts)
        return _label_pool_bytes(p, int(sum(degree[part].sum() for part in parts)))

    def build(g, table):
        comps = layouts[g][3]
        return _LabelPool(comps, comp_profiles, nbr_ptr, nbr, pairs.similarity, table)

    def start(g, pool):
        run = _label_run(*layouts[g], pool, seed, null_size)
        _check_identity(g, run.identity, observed)
        return run

    keys = [lay[3] for lay in layouts]
    refs = {
        comps: np.concatenate([comp_profiles[c] for c in comps]) for comps in set(keys)
    }
    return _run_groups(
        keys, refs, pairs.similarity, build, pool_bytes, start, null_size,
        stop_after, progress_bar, max_workers,
    )  # fmt: skip
