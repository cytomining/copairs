"""Philox4x32-10 counter-based random number generator.

Salmon et al. (2011), "Parallel random numbers: as easy as 1, 2, 3". A block
of four 32-bit words is a pure function of a 128-bit counter and a 64-bit key,
so any sample of a null distribution can be generated independently, in any
order, on any backend. Words are carried in ``uint64`` so the same code runs
elementwise on NumPy arrays and on scalars inside Numba kernels; the CUDA
kernel in :mod:`copairs.nulls.cuda` implements the identical function.
"""

import numpy as np

M32 = np.uint64(0xFFFFFFFF)
S32 = np.uint64(32)
PHILOX_M0 = np.uint64(0xD2511F53)
PHILOX_M1 = np.uint64(0xCD9E8D57)
PHILOX_W0 = np.uint64(0x9E3779B9)
PHILOX_W1 = np.uint64(0xBB67AE85)
# Key of the block that derives per-configuration keys ("copa", "irs!").
SALT0 = np.uint64(0x636F7061)
SALT1 = np.uint64(0x69727321)
INV_2_53 = 1.0 / 9007199254740992.0


def philox4x32(c0, c1, c2, c3, k0, k1):
    """Return the Philox4x32-10 block for counter ``(c0..c3)`` and key ``(k0, k1)``.

    All arguments are ``uint64`` values (scalars or arrays) below ``2**32``.
    """
    for r in range(10):
        if r > 0:
            k0 = (k0 + PHILOX_W0) & M32
            k1 = (k1 + PHILOX_W1) & M32
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        c0, c1, c2, c3 = (
            (p1 >> S32) ^ c1 ^ k0,
            p1 & M32,
            (p0 >> S32) ^ c3 ^ k1,
            p0 & M32,
        )
    return c0, c1, c2, c3


def uniform53(a, b):
    """Map two 32-bit words to a double in ``[0, 1)`` with 53 random bits."""
    return (((a >> np.uint64(5)) << np.uint64(26)) | (b >> np.uint64(6))) * INV_2_53


def config_key_arrays(seed: int, num_pos: np.ndarray, total: np.ndarray):
    """Philox keys ``(k0, k1)`` of each ``(num_pos, total)`` null configuration.

    A key depends only on ``(seed, num_pos, total)``, so a configuration's null
    is the same whichever other configurations are computed alongside it.
    """
    if not 0 <= seed < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    num_pos, total = np.asarray(num_pos), np.asarray(total)
    if len(num_pos) and (num_pos.max() >= 2**32 or total.max() >= 2**32):
        raise ValueError("num_pos and total must be < 2**32")
    u = np.uint64
    n = len(num_pos)
    w = philox4x32(
        num_pos.astype(u),
        total.astype(u),
        np.full(n, seed & 0xFFFFFFFF, dtype=u),
        np.full(n, seed >> 32, dtype=u),
        SALT0,
        SALT1,
    )
    return w[0], w[1]
