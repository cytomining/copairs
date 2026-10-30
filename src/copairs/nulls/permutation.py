"""The label-permutation mAP null: relabel profiles, keep their similarities.

The paper's null hypothesis is that a group's profiles and its reference
profiles are drawn from the same distribution, so which of them carry the
group's label is arbitrary. This module tests exactly that: a draw picks, in
each block, as many profiles as the group has members there from the block's
pool (its members and their shared references), gives them the group's
label, and recomputes the group's mAP from the observed similarities: each
labelled profile ranks the other labelled profiles (its positives, by the
group's positive pairs) against the block's unlabelled pool profiles, as the
observed members rank their references. The statistic is thus the same
function of the labelling for the observed labelling and every draw. The
negative rules are not re-applied to relabelled profiles: a column of
``neg_diffby`` such as a control flag moves with the label, which is what
designs ranking replicates against controls need, and a layout column there
(e.g. negatives only from other plates) is not enforced for relabelled
profiles. Unlike the joint and copied nulls, it keeps whatever dependence the
data induces between members' rank lists.

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
"""

import threading
from operator import index as integer_index
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor

import numpy as np

from copairs.nulls.common import (
    chunks,
    progress,
    stream_key,
    check_pairs,
    group_edges,
    resolve_seed,
    make_head_sort,
    tie_thresholds,
    check_null_size,
    check_max_workers,
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
# Estimated bytes of the pools alive at once (:func:`_pool_bytes`): threads
# wait to build a pool until it fits beside the others, so concurrent pools
# hold at most this much, or one pool if it alone is larger (MAX_POOL
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

# Rank code of a NaN key: after every number.
_NAN = np.uint32(0xFFFFFFFF)


def _make_kernels(philox, uniform, jit=lambda f: f):
    """Build the scalar draw and mAP functions around ``philox`` and ``uniform``.

    ``jit`` compiles each part (e.g. ``numba.njit``) or leaves it Python.
    Returns ``(draw, group_map)``.
    """

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

    sort_entries = make_head_sort(_INSERTION, jit)

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

    uniform_pair = make_uniform_pair(philox, uniform, jit)

    @jit
    def next_uniform(c0, c1, blk, spare, k0, k1):
        """Next uniform of draw ``(c0, c1)``'s stream: ``(u, blk, spare)``."""
        if spare >= 0.0:
            return spare, blk, -1.0
        u, spare = uniform_pair(c0, c1, blk, 0, k0, k1)
        return u, blk + 1, spare

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


def _key_table(layouts, similarity, map=map, budget=KEY_TABLE_BUDGET):
    """Return a ``_KeyTable`` of the layouts' pools, or None if it does not pay.

    It is built when its ``4 * U**2`` bytes over the ``U`` profiles of all
    pools fit in ``KEY_TABLE_BUDGET`` and its ``U**2`` pairs are fewer than
    the distinct pools' key matrices hold together.
    """
    pools = {tuple(b.tobytes() for b in lay[2]): lay[2] for lay in layouts}
    refs = [np.concatenate(blocks) for blocks in pools.values()]
    if not refs:
        return None
    profiles = np.unique(np.concatenate(refs))
    u = len(profiles)
    if 4 * u * u > budget or u * u >= sum(len(r) ** 2 for r in refs):
        return None
    return _KeyTable(profiles, similarity, map)


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

    def __init__(self, blocks, similarity, table=None, max_reference_profiles=MAX_POOL):
        check_pool_size(blocks, max_reference_profiles)
        self.blocks = blocks
        self.refs = np.concatenate(blocks).astype(np.int64)
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
        return int(min(2**16, max(256, _CHUNK_ELEMENTS // max(self.n * self.n, 1))))


def _uniforms_np(c0, c1, count, k0, k1):
    """Return the first ``count`` uniforms of each draw's stream, as ``next_uniform``."""
    out = np.empty((len(c0), count + count % 2))
    for blk in range((count + 1) // 2):
        out[:, 2 * blk], out[:, 2 * blk + 1] = uniform_pair(c0, c1, blk, 0, k0, k1)
    return out


def _draws_np(layout, start, size, k0, k1):
    """``pm`` of draws ``start, ..., start + size - 1``, one row each, as ``draw``."""
    n, bm_ptr, bm_slot = layout.n, layout.bm_ptr, layout.bm_slot
    cr_lo, cr_hi = layout.pool.cr_lo, layout.pool.cr_hi
    j = np.arange(start, start + size, dtype=np.uint64)
    ks = np.diff(bm_ptr)
    u = _uniforms_np(j & M32, j >> S32, int((2 * ks - 1).sum()), k0, k1)
    pm = np.empty((size, n), dtype=np.int64)
    rows = np.arange(size)
    i = 0
    for b, k in enumerate(ks):
        p = k + cr_hi[b] - cr_lo[b]
        chosen = np.empty((size, k), dtype=np.int64)
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
        ent = (keys.astype(np.uint64) << np.uint64(32)) | slots.astype(np.uint64)
        ent.sort(axis=1)
        k = ent >> np.uint64(32)
        t = (ent & np.uint64(0xFFFFFFFF)).astype(np.int64)
        # Drawn block profiles ranked strictly before each entry's key.
        block = layout.slot_block[t] == b
        new = np.ones(k.shape, dtype=np.bool_)
        new[:, 1:] = k[:, 1:] != k[:, :-1]
        run = np.maximum.accumulate(np.where(new, np.arange(k.shape[1]), 0), axis=1)
        drawn = np.take_along_axis(np.cumsum(block, axis=1) - block, run, axis=1)
        # The positives, in rank order.
        deg = layout.deg[s]
        cols = np.nonzero(layout.adjm[s][t])[1].reshape(size, deg)
        tp = np.take_along_axis(t, cols, axis=1)
        yp = np.take_along_axis(pm, tp, axis=1)
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
        before = before - np.take_along_axis(drawn, cols, axis=1)
        acc = np.zeros(size)
        for d in range(deg):
            acc += (d + 1) / (d + 1 + before[:, d])
        total += acc / deg
    return total / n


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
    pos_pairs = check_pairs(pos_pairs, "positive pairs")
    neg_pairs = check_pairs(neg_pairs, "negative pairs")
    n_prof = 1 + max(
        int(profile.max(initial=-1)),
        int(pos_pairs.max()) if pos_pairs.size else -1,
        int(neg_pairs.max()) if neg_pairs.size else -1,
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
    check_pair_counts(count, member, num_neg[rows], "negative")
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


def check_resources(
    max_reference_profiles, pool_memory_budget, key_table_memory_budget
):
    """Validate per-call reference/table limits; zero disables the shared key table."""
    values = []
    for name, value, minimum in (
        ("max_reference_profiles", max_reference_profiles, 0),
        ("pool_memory_budget", pool_memory_budget, 1),
        ("key_table_memory_budget", key_table_memory_budget, 0),
    ):
        try:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError
            number = integer_index(value)
            if number < minimum:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be an integer >= {minimum}") from None
        values.append(number)
    return tuple(values)


def check_pool_size(blocks, limit):
    """Check a group's total references before allocating its quadratic tables."""
    size = sum(len(block) for block in blocks)
    if size > limit:
        raise ValueError(
            f"a group pool has {size} reference profiles, exceeding max_reference_profiles={limit}. "
            "Select fewer references or explicitly raise the limit if RAM permits; "
            "table memory grows quadratically. Splitting the same references into blocks "
            "does not reduce their total."
        )


def group_key(seed: int, members) -> tuple[np.uint64, np.uint64]:
    """Philox key of a group's draws: a pure function of ``(seed, members)``."""
    return stream_key(seed, tuple(sorted(int(m) for m in members)))


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
    k0, k1 = group_key(seed, members)
    identity = float(_group_maps_np(layout, np.arange(layout.n)[None])[0])
    thr = float(tie_thresholds(identity))
    count = 0
    chunk = layout.chunk()
    for start, size in chunks(int(null_size), chunk):
        maps = _group_maps_np(layout, _draws_np(layout, start, size, k0, k1))
        if out is not None:
            out[start : start + size] = maps
        count += int((maps >= thr).sum())
    return count, identity, pool


def _pool_bytes(blocks):
    """Estimated bytes of the ``_Pool`` of ``blocks``: its key and rank tables."""
    r = sum(len(b) for b in blocks)
    rows = sum(len(b) * max(len(b) - 1, 0) for b in blocks)  # sorted rows
    ranked = 8 * rows + 4 * r * r if r <= RANK_POOL else 0  # words, rank table
    return 4 * r * r + 4 * rows + ranked


class _SharedPools:
    """The pools of a set of groups, each built once and freed after its last group.

    Groups ranking the same references share one read-only pool, so threads
    can run them concurrently. A pool is built only when its estimated
    bytes fit in the per-call ``budget`` beside the pools alive (or none is alive);
    until then its thread waits for other pools to be freed. Groups must be
    visited pool by pool: a waiting thread then holds no pool, and every
    alive pool's remaining groups have started, so its pools are freed.
    When a run fails, :meth:`abort` stops the waiting threads: the cancelled
    groups never release their pools.
    """

    def __init__(
        self,
        layouts,
        similarity,
        table=None,
        budget=POOL_BUDGET,
        max_reference_profiles=MAX_POOL,
    ):
        self.max_reference_profiles = max_reference_profiles
        self.similarity = similarity
        self.table = table
        self.key = [tuple(b.tobytes() for b in layout[2]) for layout in layouts]
        self.left = Counter(self.key)
        self.pools = {}
        self.bytes = {}
        self.building = {}
        self.lock = threading.Lock()
        self.freed = threading.Condition(self.lock)
        self.budget = budget
        self.used = 0
        self.aborted = False

    def acquire(self, g, blocks):
        """Return group ``g``'s pool (its ``blocks``' references), building it once."""
        key = self.key[g]
        with self.lock:
            building = self.building.setdefault(key, threading.Lock())
        with building:
            if key not in self.pools:
                need = _pool_bytes(blocks)
                with self.freed:
                    while (
                        self.used and self.used + need > self.budget
                    ) and not self.aborted:
                        self.freed.wait()
                    if self.aborted:
                        raise CancelledError("another group of the run failed")
                    self.used += need
                try:
                    pool = _Pool(
                        blocks, self.similarity, self.table, self.max_reference_profiles
                    )
                except BaseException:
                    self._free(need)
                    raise
                with self.lock:
                    self.pools[key], self.bytes[key] = pool, need
            return self.pools[key]

    def abort(self):
        """Make threads waiting for memory, and later pool builds, raise."""
        with self.freed:
            self.aborted = True
            self.freed.notify_all()

    def _free(self, size):
        with self.freed:
            self.used -= size
            self.freed.notify_all()

    def release(self, g):
        """Free group ``g``'s pool if no other group needs it."""
        key = self.key[g]
        with self.lock:
            self.left[key] -= 1
            if self.left[key]:
                return
            self.pools.pop(key, None)
            self.building.pop(key, None)
            size = self.bytes.pop(key, 0)
        self._free(size)


def permutation_pvalues(
    groups, profile, num_pos, num_total, pos_pairs, neg_pairs, similarity,
    null_size, seed, observed=None, progress_bar=False, max_workers=None,
    max_reference_profiles=MAX_POOL, pool_memory_budget=POOL_BUDGET,
    key_table_memory_budget=KEY_TABLE_BUDGET,
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
        Threads that run groups concurrently (default: ``ThreadPoolExecutor``'s).
        Each group's p-value depends only on its own draws, so results do not
        depend on it. The pools they build at once hold at most
        ``pool_memory_budget`` (default 4 GiB) of estimated key and rank tables, or one
        pool if it alone is larger.
    max_reference_profiles : int
        Maximum total references across a group's blocks (default 16,384).
    pool_memory_budget : int
        Positive byte budget for estimated concurrent pool tables (default 4 GiB).
        One oversized pool may run alone. This does not cap total process RAM.
    key_table_memory_budget : int
        Byte limit for the optional shared key table (default 256 MiB); zero
        disables this optimization. Other allocations are not included.

    Returns
    -------
    np.ndarray
        float64 p-values.
    """
    null_size = check_null_size(null_size)
    max_workers = check_max_workers(max_workers)
    seed = resolve_seed(seed)
    max_reference_profiles, pool_memory_budget, key_table_memory_budget = (
        check_resources(
            max_reference_profiles, pool_memory_budget, key_table_memory_budget
        )
    )
    profile = np.asarray(profile, dtype=np.int64)
    num_pos = np.asarray(num_pos, dtype=np.int64)
    num_neg = np.asarray(num_total, dtype=np.int64) - num_pos
    groups = [np.asarray(ix, dtype=np.int64) for ix in groups]
    layouts = group_layouts(groups, profile, num_pos, num_neg, pos_pairs, neg_pairs)
    for _, _, blocks, _ in layouts:
        check_pool_size(blocks, max_reference_profiles)
    if null_size == 0:
        return np.full(len(groups), np.nan)

    def pvalue(g):
        try:
            pool = pools.acquire(g, layouts[g][2])
            count, identity, _ = permutation_null(
                *layouts[g], similarity, null_size, seed, pool
            )
        finally:
            pools.release(g)
        if observed is not None and abs(identity - observed[g]) > 1e-6:
            raise ValueError(
                f"group {g}: mAP {identity} from the pairs differs from "
                f"{observed[g]} in the AP table; the pairs or similarity do not "
                "match the AP scores (keep the AP frame's index: no reset_index)"
            )
        return (count + 1) / (null_size + 1)

    pvals = np.empty(len(groups), dtype=np.float64)
    pools = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        try:
            # Pools ranking overlapping references slice one key table.
            table = _key_table(layouts, similarity, ex.map, key_table_memory_budget)
            pools = _SharedPools(
                layouts, similarity, table, pool_memory_budget, max_reference_profiles
            )
            # Visit groups pool by pool, so few pools are alive at a time.
            order = sorted(range(len(groups)), key=lambda g: pools.key[g])
            done = progress(
                ex.map(pvalue, order), len(order), progress_bar, "permutation null"
            )
            for g, p in zip(order, done):
                pvals[g] = p
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            if pools is not None:  # cancelled groups never release their pools
                pools.abort()
            raise
    return pvals
