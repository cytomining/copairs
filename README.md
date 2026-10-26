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
`backend="numpy"` (default) or `"numba"`. `"numpy"` has no optional
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
Linux default from Python 3.14), or set `NUMBA_THREADING_LAYER=workqueue`
before the first copairs call; that layer is fork-safe but stops the process if
Numba parallel kernels are called from several threads at once.

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
