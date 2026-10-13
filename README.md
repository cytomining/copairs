 # copairs

`copairs` is a Python package for finding groups of profiles based on metadata and calculate mean Average Precision to assess intra- vs inter-group similarities.

## Getting started

### System requirements
copairs supports Python 3.10 and newer (tested through Python 3.14) and should work with all modern operating systems (tested with MacOS 13.5, Ubuntu 18.04, Windows 10).

For Python 3.9, use the last compatible release: `pip install "copairs==0.5.5"`.

### Dependencies
copairs depends on widely used Python packages:
* numpy
* pandas
* tqdm
* statsmodels
* duckdb
* numba

### Installation

To install copairs and dependencies, run:
```bash
pip install copairs
```

To also install dependencies for running examples, run:
```bash
pip install copairs[demo]
```

#### GPU acceleration (optional)

With [CuPy](https://cupy.dev) installed and a CUDA GPU visible, copairs runs
null distributions, p-values, pair similarities and AP on the GPU
(`backend="auto"` picks it). Install the CuPy wheel matching your driver, one
of the two (the `ctk` extra brings the CUDA runtime and NVRTC, so no CUDA
Toolkit is needed):
```bash
pip install "cupy-cuda12x[ctk]>=14"  # CUDA 12 drivers, any GPU from Maxwell on
pip install "cupy-cuda13x[ctk]>=14"  # CUDA 13 drivers, compute capability >= 7.5
```

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
`get_map_pvalue` take `null=`, the null of a group's mAP:

- `"permutation"` (default of `method="fast"`) tests the paper's null
  hypothesis directly: within each block of members ranked against the same
  references (e.g. one plate's replicates and controls), any of the block's
  profiles could have carried the group's label. Draws relabel profiles and
  recompute the mAP from the observed similarities, so whatever dependence
  the data has between rank lists stays in the null. For multilabel APs
  (`map.multilabel.average_precision`), a draw moves the group's label to
  other profiles, which keep their other labels. Each group's positives must
  be its other members. It needs the pairs the APs came from, which
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
  different AP calls; such frames raise under this null. Recompute them,
  pass `pairs=` (`average_precision(..., return_pairs=True)` also returns
  them), or choose another null. They hold a reference to the features;
  `ap.attrs.pop("copairs_pairs")` drops it.

  Its smallest p-value is about one over the number of distinct
  relabellings, e.g. C(14, 2) = 91 for 2 replicates among 12 controls on one
  plate. Its cost grows with group size; `stop_after=10` stops a group's
  draws at the 10th that reaches its mAP and reports 10 / draws (Besag and
  Clifford's sequential p-value), which spends few draws on groups with
  large p-values and leaves small ones as they were.
- `"joint"` samples the members' APs together: a positive pair inside the
  group is one random number in both members' rank lists, so the null
  narrows as replicates are added. Other similarities are treated as
  independent, which real profiles are not (members share references, and
  some profiles are close to many others), so on real data it can give too
  small p-values for groups of many replicates. It needs no pairs.
- `"copied"` is the null of copairs <= 0.5.5 (and the only null of
  `method="legacy"`): members with equal pair counts share one AP draw, so the
  null does not narrow with replicates, while members with different counts
  get independent draws.

## Performance

`average_precision`, `mean_average_precision` and the p-value functions take
`method="fast"` (default) or `"legacy"`, and a `backend`. `"auto"` (default)
picks `"cuda"` when CuPy can compile kernels for a visible GPU, else `"numba"`.
The null and p-value functions also accept `"numpy"`, a slower reference
implementation; the fast AP stage (`average_precision`, `multilabel`) runs on
`"cuda"` or `"numba"` only, and its NumPy code is `method="legacy"`.

- Null distributions are sampled exactly (Philox counter-based RNG, Vitter's
  Algorithm A) in O(1) memory per sample instead of permuting a
  `null_size x total` matrix, and p-values are streamed in chunks. Sample `j`
  of a `(num_pos, total)` null depends only on `(seed, num_pos, total, j)`, and
  the NumPy, Numba and CUDA backends return identical samples.
- AP is computed by counting, without sorting rank lists, and similarities by
  per-pair kernels; multilabel pairs come from an inverted label index.
- `copairs.fastap.draw_average_precisions` scores many query-vs-reference draws
  in one batched call.

Differences from `method="legacy"` (copairs <= 0.5.5): null samples come from a
different random stream (p-values agree within Monte Carlo error); mAP p-values
count null values `>=` the observed mAP, as in the paper, where legacy counted
`>` and returned too small p-values when the mAP equals an atom of the null
(e.g. perfect retrieval); and similarities from the kernels match the generic
ones to float32 rounding, so near-tied pairs can rank differently. The kernels
compute in float64, so float32 profiles whose norms overflow float32 (entries
around 1e19 and up) get their actual similarity instead of the generic 0 or NaN.

Numba starts one thread per host CPU. In a container or job with a CPU quota,
set `NUMBA_NUM_THREADS` to the quota, or kernels oversubscribe the CPUs.

Multiprocessing: a process forked after copairs ran on the GPU picks a CPU
backend, since CUDA cannot be used after `fork`. If Numba uses its GNU OpenMP
threading layer (the default when TBB is not installed), it terminates a forked
child that runs a parallel kernel once the parent has run one. Use the `spawn`
or `forkserver` start method, or set `NUMBA_THREADING_LAYER=workqueue` (or
install `tbb`) before the first copairs call.

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
