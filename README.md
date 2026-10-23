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
  under `cache_dir` (default `~/.copairs`).
- The fast method's null samples come from a different random stream, so
  p-values agree with `"legacy"` within Monte Carlo error. Its mAP p-values
  count null values `>=` the observed mAP, as in the paper; legacy counts `>`,
  which gives too small p-values when the mAP equals an atom of the null
  (e.g. perfect retrieval).

`backend="numpy"` needs nothing extra. `backend="numba"` compiles the same
code with [Numba](https://numba.pydata.org), an optional dependency:
```bash
pip install copairs[numba]
```
The two backends return bitwise-identical nulls and p-values, so
`backend="auto"` (default) uses Numba whenever it is installed; if Numba fails
to import, `"auto"` warns once and uses NumPy.

Numba starts one thread per host CPU. In a container or job with a CPU quota,
set `NUMBA_NUM_THREADS` to the quota, or kernels oversubscribe the CPUs. The
NumPy backend samples on one thread per CPU the process may run on (its CPU
affinity, which follows `taskset` and Slurm allocations but not a CPU quota),
or on `max_workers` threads where the function takes it. If
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
