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

To also install the optional Numba backend (see [Performance](#performance)), run:
```bash
pip install 'copairs[numba]'
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

`backend="numpy"` needs nothing extra; `backend="numba"` compiles the same
code with [Numba](https://numba.pydata.org) (the `numba` extra above). In the
mAP functions, `backend` applies to each null of the fast method: the
copied null's AP samples, the joint null's samples, and the permutation
null's relabellings (moves of a label, for multilabel APs) and their mAPs.
For each, the Numba backend compiles the scalar functions that define a draw
and the NumPy backend computes the same draws vectorized. The two backends
return bitwise-identical nulls and p-values (tested), so `backend="auto"`
(default) uses Numba whenever it imports; if an installed Numba fails to
import, `"auto"` warns once and uses NumPy. A Numba that imports but cannot
compile a kernel raises instead; the tests pass on Numba 0.60 (the extra's
minimum) and 0.68. The first call of each null under Numba compiles its
kernels (a few seconds per null) and caches the compiled code next to
copairs' sources, where later processes load it in under a second. `"auto"`
never picks the GPU (see below).

Numba starts one thread per host CPU. In a container or job with a CPU quota,
set `NUMBA_NUM_THREADS` to the quota, or kernels oversubscribe the CPUs. The
NumPy backend samples on one thread per CPU the process may run on (its CPU
affinity, which follows `taskset` and Slurm allocations but not a CPU quota),
or on `max_workers` threads where the function takes it. The permutation
null runs groups on `max_workers` threads (by default one per CPU, at most
32, and under Numba one per Numba thread), which share a large group's
draws. Under Numba these threads run the draws themselves when they run as
many groups at once as Numba has threads; otherwise the draws run on Numba's
threads, one group at a time, as many of them as the calling thread has
(`numba.set_num_threads` is per thread). If Numba uses its GNU OpenMP threading
layer (the default when it cannot load TBB), a process forked after copairs
ran a Numba kernel stops when it runs one itself ("Terminating: fork() called
from a process already using GNU OpenMP"). Use the `spawn` or `forkserver`
start method, set `NUMBA_THREADING_LAYER=workqueue` before the first copairs
call, or pass `backend="numpy"` in the children.

#### GPU acceleration (optional)

With an NVIDIA GPU and a CUDA 12 driver, the `gpu` extra adds
`backend="cuda"`:
```bash
pip install 'copairs[gpu]'
```
It installs Numba and CuPy's CUDA 12 wheel (`cupy-cuda12x`). CuPy compiles
copairs' kernels at run time, so it also needs the CUDA libraries (NVRTC, and
cuBLAS for `draw_average_precisions`): from a CUDA 12 Toolkit, or with
`pip install 'cupy-cuda12x[ctk]'`.

- AP nulls and p-values (the copied mAP null): `backend="cuda"`, only when
  asked for, samples and counts on the GPU; the joint and permutation nulls
  have no GPU kernels and raise a ValueError under it. It steps long gaps of
  Algorithm A in float32, which is faster on GPUs with few float64 units, and
  its guided gap search uses CUDA's `log1p`, not glibc's. Error bounds keep
  the gaps equal to the CPU's, so samples are NumPy's, but not by identical
  arithmetic, and no GPU run has tested them bit for bit yet; `"auto"` stays
  on the CPU.
- AP stage: `backend="cuda"`, only when asked for, computes pair similarities
  and AP on the GPU, streaming negative pairs, their similarities and keys
  through it in chunks of bounded memory, and runs multilabel matching,
  per-label AP and normalized AP on Numba. Given the same similarities, its APs
  equal `"numba"`'s bit for bit; its similarities sum in another order, so
  they match to float32 rounding. `draw_average_precisions(...,
  backend="cuda")` scores draws of up to 6144 queries on the GPU and larger
  ones on Numba.
- Without CuPy or a usable GPU, `backend="cuda"` raises an error that names
  the extra and the reason, and `copairs.nulls.available_backends()` leaves
  `"cuda"` out (warning only when a GPU is visible but kernels fail to
  compile). The check runs once per process: a process forked after its
  parent used CUDA cannot use the GPU, so `"cuda"` is unavailable there.

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
  lists stays in the null. For multilabel APs
  (`map.multilabel.average_precision`, grouped by the label column), a draw
  moves the group's label to as many profiles of each component of the
  negative-pair graph as hold it there; those profiles keep their other
  labels. Each group's positives must be its other members; for groups of
  one profile whose positives are in other groups (e.g. one profile per
  guide, positives by gene), use `"joint"`, which there is that profile's AP
  null. It needs the pairs the APs came from, which
  `average_precision` and `multilabel.average_precision` attach to the frame
  they return, so the usual calls need nothing extra:

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
  `ap.attrs.pop("copairs_pairs")` drops them. A multilabel group needs every
  AP row of its label, since any profile holding the label could hold it
  under the null, and a label moves among every profile of the
  negative-pair components holding it (e.g. its plates under `neg_sameby`),
  at most 16,384 profiles: a label held on more raises, so use `"joint"`
  or `"copied"` for it. The null recomputes similarities from the features,
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
  `stop_after=10` stops a group's draws at the 10th that reaches its mAP and
  reports 10 / draws (Besag and Clifford's sequential p-value). It spends few
  draws on groups with large p-values, and groups that never reach 10 hits
  run all `null_size` draws and keep their p-value; draw `j` is the same
  either way. It stays calibrated, but p-values near a threshold carry about
  `1 / sqrt(stop_after)` relative error however large `null_size` is, so
  borderline calls depend on the seed: use it to screen, or raise it for
  final calls. Other nulls reject it.
- `"joint"` samples the members' APs together: a positive pair inside the
  group is one random number in both members' rank lists, so the null
  narrows as replicates are added. Other similarities are treated as
  independent, which real profiles are not (members share references, and
  some profiles are close to many others), so on real data it can give too
  small p-values for groups of many replicates. It takes each group's
  positive pairs from the attached pairs (or `pairs=`, `pos_pairs=`), so
  positives in other groups stay private to each member; on multilabel
  frames a pair joins its two profiles' AP rows for the label they share.
  Frames without pairs are treated as replicate groups when every member
  has `len(group) - 1` positives.
- `"copied"` is the null model of copairs <= 0.5.5: each member's AP null is
  sampled per `(n_pos_pairs, n_total_pairs)` configuration and the group's
  null averages them, so members with equal pair counts share one draw and
  the null does not narrow with replicates. Use it for results comparable to
  0.5.5: `method="legacy"`, whose only null it is, samples and counts as
  0.5.5 did (null mAPs `>` the observed mAP); `null="copied"` under the fast
  method samples it exactly and counts `>=`. Both agree with 0.5.5 within
  Monte Carlo error, not bitwise: each configuration is now seeded by its
  pair counts.

## Performance

`map.average_precision`, `map.multilabel.average_precision`,
`matching.find_pairs_multilabel` and `map.normalization.normalize_ap` take
`backend="numpy"` (default) or `"numba"`, and the two AP functions also
`"cuda"` (see [GPU acceleration](#gpu-acceleration-optional)). `"numpy"`
has no optional dependencies and gives the results of copairs <= 0.5.5.
`"numba"` needs the `numba` extra, which is imported only when requested:

```python
ap = copairs.map.average_precision(meta, feats, ..., backend="numba")
```

- Similarities come from per-pair kernels for `cosine`, `abs_cosine`,
  `correlation`, `euclidean`, `manhattan` and `chebyshev`; other distances and
  callables keep the NumPy functions.
- AP is computed by counting, for each negative, the sorted positives it ranks
  behind, without sorting rank lists; normalized AP uses one table of harmonic
  numbers.
- Multilabel pairs come from an inverted label index (falling back to SQL for
  inputs it cannot represent), and per-label APs from one kernel.
- `copairs.fastap.draw_average_precisions` scores many query-vs-reference draws
  in one batched call.

Given the same similarities, APs agree with `"numpy"` to float64 rounding, and
normalized APs bit for bit. The kernels' similarities match the NumPy ones to
float32 rounding, so near-tied pairs can rank differently. They compute in
float64, so float32 profiles whose norms overflow float32 (entries around 1e19
and up) get their actual similarity instead of 0 or NaN.

For Numba's threads, CPU quotas and `fork`, see
[Null distributions and the optional Numba backend](#null-distributions-and-the-optional-numba-backend).
Its `workqueue` threading layer stops the process if AP-stage kernels are
called from several threads at once (the nulls launch theirs one at a time).

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
