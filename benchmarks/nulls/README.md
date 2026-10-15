# mAP null simulations

Scripts behind the simulation tables in the comparison of copairs' mAP nulls on
[#151](https://github.com/cytomining/copairs/pull/151). Each null scores the same
simulated APs through copairs' API (`average_precision`, `mean_average_precision`):

- **main**: copairs <= 0.5.5's null (`null="copied"`);
- **joint**: #151's null (`null="joint"`);
- **permutation**: label permutation (`null="permutation"`, the fast method's default).

The **exact** null (`exact_null.py`) is the mAP distribution of pure-noise groups in the
iid design, simulated directly; a test that uses it is calibrated by construction.

## Run

From this directory, in an environment with copairs installed:

```sh
export NUMBA_NUM_THREADS=8
mkdir -p out
python simulate_nulls.py calibration out   # ~6 min
python simulate_nulls.py plate out         # ~2 min
python simulate_nulls.py detection out     # ~4 min
python power_sweep.py out/sweep.csv        # ~11 min
python power_sweep.py out/sweep_n50.csv --cells 50:100,50:250,50:1000 \
    --null-groups 50000 --groups 1000      # ~15 min
python tables.py out out/sweep.csv out/sweep_n50.csv
```

Times are on 8 CPU threads. `tables.py` simulates the exact nulls of the detection
design on first use (200,000 groups for each group size, a few minutes) and caches them
in `out/`. Seeds are fixed: with the same packages, a rerun reproduced the comment's numbers exactly. Other BLAS builds can differ in the last bits of float32 similarities. In `power_sweep.py` the
cells share one random stream, so a cell's numbers depend on the cells run before it.

## Which script gives which table

| table in the comment | script, then `tables.py` section |
|---|---|
| reported p-value for a given exact p | `simulate_nulls.py detection`; "Reported p-value for a given exact p" |
| detection and false positives | `simulate_nulls.py detection` and `calibration`; "Detection and false positives" |
| hubs | `simulate_nulls.py calibration`; "Hubs" |
| plates | `simulate_nulls.py plate`; "Plates" |
| when permutation loses power | `power_sweep.py`; "Power against the exact null" |

The comment's numbers were produced at copairs 4dd3a51.
