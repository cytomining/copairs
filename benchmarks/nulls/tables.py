"""Print the simulation tables of the #151 comment from simulate_nulls.py and power_sweep.py.

SIM_DIR holds detection.csv, calibration.csv and plate.csv; the exact nulls of the
detection design (200,000 pure-noise groups for each n) are simulated on first use and
cached there as exact_null_K100.npz. SWEEP_CSV are power_sweep.py outputs, each with its
``_raw`` directory next to it.

Usage: python tables.py SIM_DIR [SWEEP_CSV ...]
"""

import os
import argparse
from math import comb

import numpy as np
import pandas as pd
from exact_null import D, maps_of, exact_pvalues

NULLS = ("copied", "joint", "permutation")  # "main" is copairs' copied null


def pct(x, digits=1):
    """Format x as a percent with the given decimals."""
    return f"{100 * x:.{digits}f}%"


def exact_nulls(sim_dir):
    """Return sorted exact-null mAPs of the detection design (K = 100), cached."""
    path = f"{sim_dir}/exact_null_K100.npz"
    if not os.path.exists(path):
        rng = np.random.default_rng(778)
        nulls = {
            f"n{n}": np.sort(maps_of(n, 100, 200_000, 0.0, rng)) for n in (4, 8, 20)
        }
        np.savez_compressed(path, **nulls)
    z = np.load(path)
    return {n: z[f"n{n}"] for n in (4, 8, 20)}


def reported(det, nulls):
    """Print what a compound with a given exact p-value is reported as (medians)."""
    print("| replicates | exact p | main | joint | permutation |")
    print("|---:|---:|---:|---:|---:|")
    for n in (4, 8, 20):
        d = det[det.n == n]
        exact = exact_pvalues(nulls[n], d["map"].to_numpy())
        cells = {c: [] for c in NULLS}
        for t in (0.05, 0.01, 0.001):
            sel = d[(exact >= t / 1.25) & (exact <= t * 1.25)]
            for c in NULLS:
                cells[c].append(f"{sel[c].median():.3g}")
        print(
            f"| {n} | 0.05 / 0.01 / 0.001 | "
            + " | ".join(" / ".join(cells[c]) for c in NULLS)
            + " |"
        )


def detected(det, cal, nulls):
    """Print detection and false positives, with the exact null as the target."""
    print(
        "| replicates (signal) | measure | target (exact null) | main | joint | permutation |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for n, s in ((4, 0.4), (8, 0.3), (20, 0.2)):
        d = det[(det.n == n) & (det.strength == s)]
        exact = exact_pvalues(nulls[n], d["map"].to_numpy())
        for i, a in enumerate((0.05, 0.001)):
            label = f"{n} ({s}, mAP {d['map'].mean():.2f})" if i == 0 else ""
            row = [pct((exact <= a).mean())] + [pct((d[c] <= a).mean()) for c in NULLS]
            print(
                f"| {label} | signal groups detected at p ≤ {a} | "
                + " | ".join(row)
                + " |"
            )
        c = cal[(cal.kind == "iid") & (cal.n == n)]
        row = [pct((c[k] <= 0.001).mean(), 2) for k in NULLS]
        print("| | null groups called at p ≤ 0.001 | 0.1% | " + " | ".join(row) + " |")
    print()
    for n in (4, 8, 20):
        d = det[det.n == n]
        m = np.sort(d["map"].to_numpy())
        exact_cut = m[np.argmax(exact_pvalues(nulls[n], m) <= 0.001)]
        joint_cut = d.loc[d.joint <= 0.001, "map"].min()
        fp = np.mean(nulls[n] >= joint_cut - 1e-9)
        print(
            f"n={n}: mAP cutoff for p <= 0.001: exact {exact_cut:.3f}, joint "
            f"{joint_cut:.3f}; exact-null groups above joint's cutoff: {pct(fp, 2)}"
        )


def hubs(cal):
    """Print null groups called with iid and with hub features."""
    print("| features | replicates | measure | target | main | joint | permutation |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for (kind, n), d in cal.groupby(["kind", "n"], sort=False):
        for a, target in ((0.05, "5%"), (0.01, "1%")):
            row = " | ".join(pct((d[c] < a).mean()) for c in NULLS)
            print(
                f"| {kind} | {n} | null groups called at p < {a} | {target} | {row} |"
            )


def unit(x):
    """Scale rows to unit norm, as float32."""
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def same_plate_cosines(taus=(0.0, 0.3, 0.6), plates=2000):
    """Return the mean cosine of two same-plate wells for each plate offset."""
    rng = np.random.default_rng(0)
    out = {}
    for tau in taus:
        X = rng.standard_normal((plates, 13, D)) / np.sqrt(D)
        U = unit(X + tau * unit(rng.standard_normal((plates, 1, D))))
        out[tau] = float(np.einsum("pd,pd->p", U[:, 0], U[:, 1]).mean())
    return out


def plates(pl):
    """Print null and signal groups called at p < 0.05 in the plate layout."""
    print(
        "| plate offset (same-plate cosine) | measure | target | main | joint | permutation |"
    )
    print("|---|---|---:|---:|---:|---:|")
    cosine = same_plate_cosines()
    for (tau, s), d in pl.groupby(["tau", "strength"]):
        label = f"{tau} ({cosine[tau]:.3f})" if s == 0 else ""
        measure, target = (
            ("null groups called", "5%")
            if s == 0
            else (f"signal {s} detected", "higher")
        )
        row = " | ".join(pct((d[c] < 0.05).mean()) for c in NULLS)
        print(f"| {label} | {measure} | {target} | {row} |")


def sweep(paths):
    """Print permutation minus exact, in points of detection, and joint's false positives."""
    d = pd.concat([pd.read_csv(p) for p in paths])
    raw = {}
    for p in paths:
        stem = os.path.splitext(p)[0] + "_raw"
        for n, K in d[["n", "K"]].drop_duplicates().itertuples(index=False):
            f = f"{stem}/cell_n{n}_K{K}.npz"
            if os.path.exists(f):
                raw[(n, K)] = np.load(f)
    print("| replicates | K = 2n | K = 5n | K = 10n | K = 20n | K = 50n |")
    print("|---:|---:|---:|---:|---:|---:|")
    g = d.set_index(["n", "K", "alpha"])
    for n in sorted(d.n.unique()):
        cells = []
        for mult in (2, 5, 10, 20, 50):
            key = (n, n * mult)
            if (n, n * mult, 0.001) not in g.index:
                cells.append("")
                continue
            gap = [
                100
                * (g.loc[key + (a,), "perm_power"] - g.loc[key + (a,), "exact_power"])
                for a in (0.001, 0.01)
            ]
            cells.append(f"{gap[0]:+.1f} [{gap[1]:+.1f}]")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\nPer cell (paired SE of permutation − exact from the raw arrays):\n")
    print(
        "| n | K | relabellings | exact-null min p | p ≤ 0.05 | p ≤ 0.01 | p ≤ 0.001 | joint false positives ÷ α |"
    )
    print("|---:|---:|---:|---:|---|---|---|---|")
    for (n, K), c in d.groupby(["n", "K"]):
        c = c.set_index("alpha")
        z = raw.get((n, K))
        cells = []
        for a in (0.05, 0.01, 0.001):
            text = " / ".join(
                f"{100 * c.loc[a, k]:.1f}"
                for k in ("exact_power", "perm_power", "joint_power")
            )
            if z is not None:
                e, q = z["exact_p"] <= a, z["perm_p"] <= a
                b1, b2 = np.sum(q & ~e), np.sum(e & ~q)
                se = np.sqrt(max(b1 + b2 - (b1 - b2) ** 2 / len(e), 0)) / len(e)
                text += f" (±{100 * se:.1f})"
            cells.append(text)
        fp = " / ".join(f"{c.loc[a, 'joint_fpr'] / a:.2f}" for a in (0.05, 0.01, 0.001))
        print(
            f"| {n} | {K} | {comb(n + K, n):.2g} | {c['p_top'].iloc[0]:.2g} | "
            + " | ".join(cells)
            + f" | {fp} |"
        )
    print("\nCells: exact / permutation / joint detection, % of signal groups.")


def derived():
    """Print the hypergeometric and resolution numbers quoted in the comment."""
    for n, K in ((4, 20), (4, 40), (4, 80)):
        print(f"P(a relabelling keeps >= {n - 1} of {n} replicates), K={K}: "
              f"{(n * K + 1) / comb(n + K, n):.2g}")  # fmt: skip
    for n in (2, 3, 4, 6):
        K = next(k for k in range(1, 10_000) if comb(n + k, n) >= 10 / 0.001)
        print(f"controls for C(n+K, n) >= 10/0.001 with n={n}: {K}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sim_dir")
    parser.add_argument("sweep_csv", nargs="*")
    args = parser.parse_args()
    det = pd.read_csv(f"{args.sim_dir}/detection.csv")
    cal = pd.read_csv(f"{args.sim_dir}/calibration.csv")
    pl = pd.read_csv(f"{args.sim_dir}/plate.csv")
    nulls = exact_nulls(args.sim_dir)
    for title, show in (
        ("Reported p-value for a given exact p", lambda: reported(det, nulls)),
        ("Detection and false positives", lambda: detected(det, cal, nulls)),
        ("Hubs", lambda: hubs(cal)),
        ("Plates", lambda: plates(pl)),
    ):
        print(f"\n## {title}\n")
        show()
    if args.sweep_csv:
        print("\n## Power against the exact null\n")
        sweep(args.sweep_csv)
    print("\n## Derived\n")
    derived()
