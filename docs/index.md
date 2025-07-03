# copairs

Find pairs and compute metrics between them.

copairs is a Python package for finding pairs of profiles based on metadata attributes and computing metrics between them. It's designed for analyzing high-dimensional biological data, particularly in drug discovery and cell profiling applications.

## Installation

```bash
pip install copairs
```

## Key Features

- **Flexible pair finding**: Define pairs based on metadata constraints (same/different attributes)
- **Efficient computation**: Optimized for large-scale profile datasets
- **Mean Average Precision (mAP)**: Assess profile quality and phenotypic activity
- **Replication analysis**: Evaluate consistency across replicates

## Getting Started

Check out the [Phenotypic Activity example](examples/phenotypic_activity.ipynb) to see how copairs can be used to assess phenotypic activity of perturbations using mean average precision (mAP).

## Core Modules

- **[matching](api/matching.md)**: Find pairs based on metadata constraints
- **[compute](api/compute.md)**: Calculate similarity metrics between profiles
- **[map](api/map.md)**: Compute Mean Average Precision scores
- **[replicating](api/replicating.md)**: Analyze replication consistency