"""Power of the permutation and joint mAP nulls against the exact null, over n and K.

Design: n replicates and K private controls per group, 50 iid Gaussian features,
cosine similarity; signal groups add strength s along a random unit direction shared by
the replicates. Per (n, K):

exact        the exact null (exact_null.py) from M pure-noise groups.
strength     s at which the exact null detects 50% of signal groups at the strictest
             alpha its smallest p-value reaches (0.001 unless the design is too small).
permutation  copairs' permutation null (null size 20,000) on G signal groups.
joint        copairs' joint null: the smallest mAP with joint p <= alpha on a fine grid
             (null size 1e6); its false-positive rate is measured on the exact null.

Writes one CSV row per (n, K, alpha) to OUT_CSV and each cell's arrays to
``<OUT_CSV stem>_raw/``. Cells share one random stream, so a cell's numbers depend on
the cells run before it.

Usage: python power_sweep.py OUT_CSV [--cells n:K,...] [--null-groups M] [--groups G]
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
from exact_null import D, maps_of, exact_pvalues

from copairs.map import get_map_pvalue, average_precision, mean_average_precision

NULL_SIZE = 20_000
ALPHAS = (0.05, 0.01, 0.001)
CELLS = (
    "2:4,2:10,2:20,2:50,2:100,2:400,4:8,4:20,4:40,4:80,4:200,8:16,8:40,8:80,8:160,"
    "8:400,20:40,20:100,20:200,20:400,20:1000"
)


def strength_for(n, K, null_sorted, alpha, rng, target=0.5):
    """Find s by bisection so the exact null detects ``target`` at ``alpha``."""
    lo, hi = 0.0, 2.0
    for _ in range(10):
        s = (lo + hi) / 2
        power = np.mean(
            exact_pvalues(null_sorted, maps_of(n, K, 1000, s, rng)) <= alpha
        )
        lo, hi = (s, hi) if power < target else (lo, s)
    return (lo + hi) / 2


def permutation_pvalues(n, K, X):
    """Return copairs' mAP and permutation p-value of each group, in group order."""
    N = n + K
    block = np.repeat(np.arange(len(X)), N)
    member = np.tile(np.arange(N) < n, len(X))
    meta = pd.DataFrame(
        {
            "block": block,
            "g": np.where(member, block.astype(str), "ctrl"),
            "ref": np.where(member, -1, np.arange(len(X) * N)),
        }
    )
    ap = average_precision(
        meta,
        X.reshape(-1, D),
        pos_sameby=["g", "ref"],
        pos_diffby=[],
        neg_sameby=["block"],
        neg_diffby=["g", "ref"],
        progress_bar=False,
    )
    ap = ap[ap["g"] != "ctrl"]
    res = mean_average_precision(
        ap, ["g"], NULL_SIZE, 0.05, 1, progress_bar=False, null="permutation"
    )
    res["block"] = res["g"].astype(int)
    return res.sort_values("block")[["mean_average_precision", "p_value"]].to_numpy()


def joint_cutoffs(n, K, lo, hi):
    """Find the smallest mAP on a fine grid with joint p-value <= alpha, per alpha."""
    grid = np.linspace(lo, hi, 4001)
    rows = [(k, m, n - 1, n - 1 + K) for k, m in enumerate(grid) for _ in range(n)]
    f = pd.DataFrame(
        rows, columns=["g", "average_precision", "n_pos_pairs", "n_total_pairs"]
    )
    f["normalized_average_precision"] = 0.0
    p = get_map_pvalue(f, ["g"], 1_000_000, 3, progress_bar=False, null="joint")
    p = p.sort_values("g")["p_value"].to_numpy()
    cut = {a: grid[np.argmax(p <= a)] if (p <= a).any() else np.inf for a in ALPHAS}
    return cut, grid, p


def cell(n, K, M, G, rng, raw_dir):
    """Run one (n, K) cell and return one row per alpha."""
    t0 = time.time()
    null = np.sort(maps_of(n, K, M, 0.0, rng))
    p_top = float(exact_pvalues(null, null[-1]))  # smallest exact p-value possible
    alpha_cal = next((a for a in sorted(ALPHAS) if p_top <= a / 2), max(ALPHAS))
    s = strength_for(n, K, null, alpha_cal, rng)
    keep = []
    sig = maps_of(n, K, G, s, rng, keep=keep)
    perm = permutation_pvalues(n, K, keep[0])
    agree = float(np.abs(perm[:, 0] - sig).max())
    hi = float(max(sig.max(), null.max()))
    cut, grid, joint_p = joint_cutoffs(n, K, float(np.quantile(null, 0.5)), hi)
    exact_p = exact_pvalues(null, sig)
    np.savez_compressed(
        f"{raw_dir}/cell_n{n}_K{K}.npz",
        null=null,
        sig=sig,
        perm_p=perm[:, 1],
        exact_p=exact_p,
        joint_grid=grid,
        joint_p=joint_p,
        strength=s,
    )
    rows = [
        dict(
            n=n,
            K=K,
            r=n / (n + K),
            strength=s,
            alpha_cal=alpha_cal,
            p_top=p_top,
            alpha=a,
            exact_power=float(np.mean(exact_p <= a)),
            perm_power=float(np.mean(perm[:, 1] <= a)),
            joint_power=float(np.mean(sig >= cut[a])),
            joint_fpr=float(np.mean(null >= cut[a])),
            map_agree=agree,
            seconds=time.time() - t0,
        )
        for a in ALPHAS
    ]
    print(
        f"n={n} K={K} s={s:.3f} {time.time() - t0:.0f}s "
        + " ".join(
            f"a={r['alpha']}: exact {r['exact_power']:.3f} "
            f"permutation {r['perm_power']:.3f} joint {r['joint_power']:.3f} "
            f"(false positives {r['joint_fpr']:.4f})"
            for r in rows
        ),
        flush=True,
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_csv")
    parser.add_argument("--cells", default=CELLS, help="comma-separated n:K pairs")
    parser.add_argument("--null-groups", type=int, default=100_000)
    parser.add_argument("--groups", type=int, default=2000)
    args = parser.parse_args()
    raw_dir = os.path.splitext(args.out_csv)[0] + "_raw"
    os.makedirs(raw_dir, exist_ok=True)
    rng = np.random.default_rng(20261015)
    rows = []
    for c in args.cells.split(","):
        n, K = map(int, c.split(":"))
        rows += cell(n, K, args.null_groups, args.groups, rng, raw_dir)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
