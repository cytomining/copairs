 # copairs

`copairs` is a Python package for finding groups of profiles based on metadata and calculate mean Average Precision to assess intra- vs inter-group similarities.

## Getting started

### System requirements
copairs supports Python 3.11 and newer (tested through Python 3.14) and should work with all modern operating systems (tested with MacOS 13.5, Ubuntu 18.04, Windows 10).

### Dependencies
copairs depends on widely used Python packages:
* numpy
* pandas
* tqdm
* statsmodels

### Installation

To install copairs and dependencies, run:
```bash
pip install copairs
```

To also install dependencies for running examples, run:
```bash
pip install copairs[demo]
```

#### Null distributions and the optional Numba backend

The AP and mAP p-value functions (`compute.p_values`,
`map.average_precision.p_values`, `mean_average_precision`,
`mean_average_precision_hierarchical` and `get_map_pvalue`) take
`method="fast"` (default) or `"legacy"`, and a `backend`:

- `"fast"` samples each AP null exactly (Philox counter-based RNG, Vitter's
  Algorithm A) in O(1) memory per sample, instead of permuting a
  `null_size x total` matrix, and streams p-values in bounded memory without a
  disk cache. Sample `j` of a `(num_pos, total)` null depends only on
  `(seed, num_pos, total, j)`, with `seed` in `[0, 2**64)`.
- `"legacy"` is copairs 0.5.5's null, seeded per configuration and cached
  under `cache_dir` (default `~/.copairs`). In the mAP functions it is the
  `"copied"` null (see [mAP null distributions](#map-null-distributions)),
  their default under it; other nulls raise.
- The fast method's null samples come from a different random stream, so
  p-values agree with `"legacy"` within Monte Carlo error. Its mAP p-values
  count null values `>=` the observed mAP, as in the paper; legacy counts `>`,
  which gives too small p-values when the mAP equals an atom of the null
  (e.g. perfect retrieval).
- With the NumPy backend, `"fast"` can be slower than `"legacy"` for
  configurations with many positives at small null sizes (for 1000 positives
  among 100,000 pairs at `null_size=1000`, 3.6 s against 1.6 s; 0.02 s with
  Numba), so install the `numba` extra for such data.

`backend="numpy"` needs nothing extra. `backend="numba"` compiles the same
code with [Numba](https://numba.pydata.org), an optional dependency:
```bash
pip install copairs[numba]
```
In the mAP functions, `backend` applies to each null of the fast method: the
copied null's AP samples, the joint null's samples, and the permutation
null's relabellings and their mAPs. For each, the Numba backend compiles the
scalar functions that define a draw and the NumPy backend computes the same
draws vectorized. The two backends return bitwise-identical nulls and
p-values (tested), so `backend="auto"` (default) uses Numba whenever it
imports; if an installed Numba fails to import, `"auto"` warns once and uses
NumPy. A Numba that imports but cannot compile a kernel raises instead; the
tests pass on Numba 0.60 (the extra's minimum) and 0.68. The
first call of each null under Numba compiles its kernels (a few seconds per
null) and caches the compiled code next to copairs' sources, where later
processes load it in under a second.

Numba starts one thread per host CPU. In a container or job with a CPU quota,
set `NUMBA_NUM_THREADS` to the quota, or kernels oversubscribe the CPUs. The
NumPy backend samples on one thread per CPU the process may run on (its CPU
affinity, which follows `taskset` and Slurm allocations but not a CPU quota),
or on `max_workers` threads where the function takes it. The permutation
null runs groups on `max_workers` threads; under Numba they build the groups'
similarity tables while one kernel at a time runs draws on Numba's threads. If
Numba uses its GNU OpenMP threading layer (the default when it cannot load
TBB), a process forked after copairs ran a Numba kernel stops when it runs one
itself ("Terminating: fork() called from a process already using GNU
OpenMP"). Use the `spawn` or `forkserver` start method, set
`NUMBA_THREADING_LAYER=workqueue` before the first copairs call, or pass
`backend="numpy"` in the children.

### Testing

To run tests, run:
```bash
pip install -e .[test]
pytest
```

## Usage

We provide examples demonstrating how to use copairs for:
- [grouping profiles based on their metadata](https://github.com/cytomining/copairs/blob/main/docs/examples/finding_pairs.ipynb)
- [calculating mAP to assess phenotypic activity of perturbations](https://github.com/cytomining/copairs/blob/main/docs/examples/phenotypic_activity.ipynb)
- [calculating mAP to assess phenotypic consistency of perturbations](https://github.com/cytomining/copairs/blob/main/docs/examples/phenotypic_consistency.ipynb)
- [estimating null size for mAP p-value calculation](https://github.com/cytomining/copairs/blob/main/docs/examples/null_size.ipynb)

## mAP null distributions

`mean_average_precision`, `mean_average_precision_hierarchical` and
`get_map_pvalue` take `null=`, the null of a group's mAP. Under
`method="fast"` (default) all three count null mAPs `>=` the observed mAP, so
an mAP equal to an atom of the null (e.g. perfect retrieval) counts the atom.

- `"permutation"` (the default under `method="fast"`) tests the paper's null hypothesis directly: within
  each block of members ranked against the same references (e.g. one plate's
  replicates and controls), any of the block's profiles could have carried the
  group's label. Draws relabel profiles and recompute the mAP from the
  observed similarities, so whatever dependence the data has between rank
  lists stays in the null. Each group's positives must be its other members;
  for groups of one profile whose positives are in other groups (e.g. one
  profile per guide, positives by gene), use `"joint"`, which there is that
  profile's AP null. It needs the pairs the APs came from, which
  `average_precision` attaches to the frame it returns, so the usual calls
  need nothing extra:

  ```python
  ap = copairs.map.average_precision(meta, feats, ...)
  result = copairs.map.mean_average_precision(
      ap, sameby, null_size=10_000, threshold=0.05, seed=0
  )
  ```

  The pairs (`ap.attrs["copairs_pairs"]`) follow the frame through filtering
  and copies, but not through saving, pickling or concatenating frames of
  different AP calls; such frames raise under this null. Filter out whole
  groups (or controls) only: the members' APs were computed with all of
  their group's members as positives, so dropping some members raises
  under this null; recompute the APs on the kept profiles or use `"joint"`
  or `"copied"`. For frames without pairs, recompute the APs, pass
  `pairs=` (`average_precision(..., return_pairs=True)` also returns them),
  or choose another null. The attached pairs keep the features and
  every positive and negative pair alive with the frame (8 bytes per pair,
  e.g. 800 MB for 10^8 negative pairs), whichever null is used;
  `ap.attrs.pop("copairs_pairs")` drops them. Frames from
  `map.multilabel.average_precision` carry no pairs: use `"joint"` or
  `"copied"` for them. The null recomputes similarities from the features,
  so distances that estimate their scale from each batch of pairs
  (`seuclidean`, `mahalanobis`) raise under it; use `"joint"` or `"copied"`.
  Relabelled profiles keep the group's positive pairs, so with `pos_diffby`
  the references must not change them (e.g. one plate per block of
  references when `pos_diffby=["plate"]`); layouts where they could raise.

  Its smallest p-value is about one over the number of distinct
  relabellings, e.g. C(14, 2) = 91 for 2 replicates among 12 controls on one
  plate. The null assumes the group's profiles are exchangeable with the
  references they are ranked against: replicates that share structure the
  references lack (the same well, clone or batch) make it too narrow, so match
  references to that structure (`neg_sameby`) or test whole units. It is the
  slowest of the three nulls, and its cost grows with group size.
- `"joint"` samples the members' APs together: a positive pair inside the
  group is one random number in both members' rank lists, so the null
  narrows as replicates are added. Other similarities are treated as
  independent, which real profiles are not (members share references, and
  some profiles are close to many others), so on real data it can give too
  small p-values for groups of many replicates. It takes each group's
  positive pairs from the attached pairs (or `pairs=`, `pos_pairs=`), so
  positives in other groups stay private to each member; frames without
  pairs are treated as replicate groups when every member has
  `len(group) - 1` positives.
- `"copied"` is the null model of copairs <= 0.5.5: each member's AP null is
  sampled per `(n_pos_pairs, n_total_pairs)` configuration and the group's
  null averages them, so members with equal pair counts share one draw and
  the null does not narrow with replicates. Use it for results comparable to
  0.5.5: `method="legacy"`, whose only null it is, samples and counts as
  0.5.5 did (null mAPs `>` the observed mAP); `null="copied"` under the fast
  method samples it exactly and counts `>=`. Both agree with 0.5.5 within
  Monte Carlo error, not bitwise: each configuration is now seeded by its
  pair counts.

## Citation
If you find this work useful for your research, please cite our [paper](https://doi.org/10.1038/s41467-025-60306-2):

Kalinin, A.A., Arevalo, J., Serrano, E., Vulliard, L., Tsang, H., Bornholdt, M., Muñoz, A.F., Sivagurunathan, S., Rajwa, B., Carpenter, A.E., Way, G.P. and Singh, S., 2025. A versatile information retrieval framework for evaluating profile strength and similarity. _Nature Communications_ 16, 5181. doi:10.1038/s41467-025-60306-2

BibTeX:
```
@article{kalinin2025versatile,
  author       = {Kalinin, Alexandr A. and Arevalo, John and Serrano, Erik and Vulliard, Loan and Tsang, Hillary and Bornholdt, Michael and Muñoz, Alán F. and Sivagurunathan, Suganya and Rajwa, Bartek and Carpenter, Anne E. and Way, Gregory P. and Singh, Shantanu},
  title        = {A versatile information retrieval framework for evaluating profile strength and similarity},
  journal      = {Nature Communications},
  year         = {2025},
  volume       = {16},
  number       = {1},
  pages        = {5181},
  doi          = {10.1038/s41467-025-60306-2},
  url          = {https://doi.org/10.1038/s41467-025-60306-2},
  issn         = {2041-1723}
}
```
