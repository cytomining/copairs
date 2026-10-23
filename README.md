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
  under `cache_dir` (default `~/.copairs`).
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
code with [Numba](https://numba.pydata.org) (the `numba` extra above). The
backends return bitwise-identical nulls and p-values, so `backend="auto"`
(default) uses Numba whenever it is installed; if Numba fails to import,
`"auto"` warns once and uses NumPy. `"auto"` never picks the GPU (see below).

The NumPy backend samples on one thread per CPU the process may run on (its
CPU affinity, which follows `taskset` and Slurm allocations but not a CPU
quota), or on `max_workers` threads where the function takes it. For Numba's
threads and `fork`, see [Performance](#performance).

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

- Nulls and p-values: `backend="cuda"`, only when asked for, samples and
  counts on the GPU. It steps long gaps of Algorithm A in float32, which is
  faster on GPUs with few float64 units, and its guided gap search uses CUDA's
  `log1p`, not glibc's. Error bounds keep the gaps equal to the CPU's, so
  samples are NumPy's, but not by identical arithmetic, and no GPU run has
  tested them bit for bit yet; `"auto"` stays on the CPU.
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

## Performance

`map.average_precision`, `map.multilabel.average_precision`,
`matching.find_pairs_multilabel` and `map.normalization.normalize_ap` take
`backend="numpy"` (default), `"numba"` or `"cuda"` (see
[GPU acceleration](#gpu-acceleration-optional)). `"numpy"` has no optional
dependencies and gives the results of copairs <= 0.5.5. `"numba"` needs the
`numba` extra, which is imported only when requested:

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

Numba starts one thread per host CPU. In a container or job with a CPU quota,
set `NUMBA_NUM_THREADS` to the quota, or kernels oversubscribe the CPUs. If
Numba uses its GNU OpenMP threading layer (the default when it cannot load
TBB), a process forked after a Numba parallel kernel ran stops with
"Terminating: fork() called from a process already using GNU OpenMP" when it
runs one too. Use the `spawn` or `forkserver` start method (`forkserver` is the
Linux default from Python 3.14), set `NUMBA_THREADING_LAYER=workqueue`
before the first copairs call (that layer is fork-safe but stops the process
if Numba parallel kernels are called from several threads at once), or pass
`backend="numpy"` in the children.

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
