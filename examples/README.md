 # Examples

Example notebooks demostrating the use of `copairs`.

## Installation

To install dependencies for running examples, run:
```bash
pip install copairs[demo]
```

## Running examples

```bash
cd examples
notebook
```

## List of examples

We show how to use copairs for:

- [grouping profiles based on their metadata](./finding_pairs.ipynb)
- [calculating mAP to assess phenotypic activity of perturbations](./phenotypic_activity.ipynb)
- [calculating mAP to assess phenotypic consistency of perturbations](./phenotypic_consistency.ipynb)
- [estimating null size for mAP p-value calculation](./null_size.ipynb)

## Data used

In these examples, we used a single plate of profiles from the dataset "cpg0004" (aka LINCS), which contains Cell Painting images of 1,327 small-molecule perturbations of A549 human cells. The wells on each plate were perturbed with 56 different compounds in six different doses.

> Way, G. P. et al. Morphology and gene expression profiling provide complementary information for mapping cell state. Cell Syst 13, 911–923.e9 (2022).
