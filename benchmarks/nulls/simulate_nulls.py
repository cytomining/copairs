"""Simulated groups scored by copairs' copied ("main"), joint and permutation mAP nulls.

All nulls go through copairs' API (average_precision + mean_average_precision), so they
see the same AP scores. Each part writes ``<out_dir>/<part>.csv`` with one row per group.

detection    #151's design: n replicates share a random unit direction of a given
             strength, ranked by cosine against 100 private controls on 50 iid Gaussian
             features; 4,000 groups per (n, strength), null size 20,000.
calibration  null groups (no signal) of that design ("iid") and of a design whose
             profiles are a common mean plus noise with a per-profile scale drawn from
             lognormal(0, 0.5) ("hub"); n = 4, 8, 20; 10,000 groups each, null size 10,000.
plate        cpg0004's layout: 4 replicates, each on its own plate with 12 controls;
             negatives are same-plate controls only; every well of a plate gets a plate
             offset of norm tau. Null groups and signal groups, 10,000 each, null size
             10,000.

Usage: python simulate_nulls.py {detection,calibration,plate} OUT_DIR
"""

import time
import argparse

import numpy as np
import pandas as pd

from copairs.map import average_precision, mean_average_precision

K, D = 100, 50
NULLS = ("copied", "joint", "permutation")


def unit(x):
    """Scale rows to unit norm, as float32."""
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def block_design(n, X, G):
    """Build metadata and pair spec for G blocks of n members and K controls."""
    N = n + K
    block = np.repeat(np.arange(G), N)
    member = np.tile(np.arange(N) < n, G)
    meta = pd.DataFrame(
        {
            "block": block,
            "g": np.where(member, block.astype(str), "ctrl"),
            "ref": np.where(member, -1, np.arange(G * N)),
        }
    )
    spec = dict(
        pos_sameby=["g", "ref"],
        pos_diffby=[],
        neg_sameby=["block"],
        neg_diffby=["g", "ref"],
    )
    return meta, X.reshape(G * N, D), spec


def pvalues(meta, X, spec, null_size, seed, nulls=NULLS):
    """Return each group's mAP and its p-value under each null."""
    ap = average_precision(meta, X, progress_bar=False, **spec)
    ap = ap[ap["g"] != "ctrl"]
    out = {}
    for null in nulls:
        t = time.time()
        res = mean_average_precision(
            ap, ["g"], null_size, 0.05, seed, progress_bar=False, null=null
        ).set_index("g")
        out[null] = res["p_value"]
        out["map"] = res["mean_average_precision"]
        print(f"  {null}: {time.time() - t:.1f} s", flush=True)
    return pd.DataFrame(out)


def signal_features(n, strength, G, rng):
    """Draw features whose replicates share a random unit direction."""
    v = unit(rng.standard_normal((G, 1, D)))
    X = rng.standard_normal((G, n + K, D)) / np.sqrt(D)
    X[:, :n] += strength * v
    return unit(X)


def hub_features(n, G, rng):
    """Draw a common mean plus noise whose scale varies per profile."""
    sig = rng.lognormal(0.0, 0.5, size=(G, n + K, 1))
    return unit(1.0 + sig * rng.standard_normal((G, n + K, D)))


def detection(out_dir):
    """Simulate the signal groups of #151's detection table."""
    rng = np.random.default_rng(3)
    rows = []
    for n, strengths in [(4, (0.4, 0.6)), (8, (0.3, 0.45)), (20, (0.2, 0.3))]:
        for s in strengths:
            G = 4000
            print(f"detection n={n} s={s}", flush=True)
            meta, X, spec = block_design(n, signal_features(n, s, G, rng), G)
            p = pvalues(meta, X, spec, 20_000, 1)
            p["n"], p["strength"] = n, s
            rows.append(p.reset_index())
    pd.concat(rows).to_csv(f"{out_dir}/detection.csv", index=False)


def calibration(out_dir):
    """Simulate null groups with iid and with hub features."""
    rng = np.random.default_rng(4)
    rows = []
    for kind in ("iid", "hub"):
        for n in (4, 8, 20):
            G = 10_000
            print(f"calibration {kind} n={n}", flush=True)
            if kind == "iid":
                X = signal_features(n, 0.0, G, rng)
            else:
                X = hub_features(n, G, rng)
            meta, X, spec = block_design(n, X, G)
            p = pvalues(meta, X, spec, 10_000, 5)
            p["kind"], p["n"] = kind, n
            rows.append(p.reset_index())
    pd.concat(rows).to_csv(f"{out_dir}/calibration.csv", index=False)


def plate(out_dir, n=4, C=12, G=10_000):
    """Simulate null and signal groups in the plate layout, by plate offset."""
    rng = np.random.default_rng(6)
    rows = []
    P = G * n  # one plate per replicate
    plate_id = np.repeat(np.arange(P), C + 1)
    is_member = np.tile(np.arange(C + 1) == 0, P)
    group = plate_id // n
    meta = pd.DataFrame(
        {
            "plate": plate_id,
            "g": np.where(is_member, group.astype(str), "ctrl"),
            "ref": np.where(is_member, -1, np.arange(len(plate_id))),
        }
    )
    spec = dict(
        pos_sameby=["g", "ref"],
        pos_diffby=[],
        neg_sameby=["plate"],
        neg_diffby=["g", "ref"],
    )
    for tau in (0.0, 0.3, 0.6):
        for s in (0.0, 0.5, 0.8):
            print(f"plate tau={tau} s={s}", flush=True)
            X = rng.standard_normal((P, C + 1, D)) / np.sqrt(D)
            X += tau * unit(rng.standard_normal((P, 1, D)))
            v = unit(rng.standard_normal((G, 1, D)))
            X[:, 0] += s * np.repeat(v[:, 0], n, axis=0)
            p = pvalues(meta, unit(X).reshape(-1, D), spec, 10_000, 7)
            p["tau"], p["strength"] = tau, s
            rows.append(p.reset_index())
    pd.concat(rows).to_csv(f"{out_dir}/plate.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["detection", "calibration", "plate"])
    parser.add_argument("out_dir")
    args = parser.parse_args()
    {"detection": detection, "calibration": calibration, "plate": plate}[args.part](
        args.out_dir
    )
