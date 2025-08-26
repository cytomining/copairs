# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Testing
- Run all tests: `pytest`
- Run a specific test file: `pytest tests/test_compute.py`
- Run a specific test: `pytest tests/test_compute.py::test_function_name`

### Code Quality
- Format and lint code: `ruff check --fix src/`
- Sort imports by length: `ruff check --select I --fix src/`

### Installation
- Install package in development mode: `pip install -e .`
- Install with test dependencies: `pip install -e .[test]`
- Install with demo dependencies: `pip install -e .[demo]`
- Install with development tools: `pip install -e .[dev]`

## Architecture Overview

copairs is a Python package for computing pairwise similarities and mean Average Precision (mAP) metrics for profile data analysis. The core functionality centers around:

### Key Components

**Matching System** (`src/copairs/matching.py`)
- `Matcher` and `MatcherMultilabel` classes handle profile pairing based on metadata
- Creates pairs with "sameby" (must match) and "diffby" (must differ) column constraints
- Uses DuckDB for efficient SQL-based pair generation
- Supports multilabel matching for complex experimental designs

**Computation Engine** (`src/copairs/compute.py`)
- Distance and similarity calculations between profile feature vectors
- Parallel processing support via ThreadPool for batch operations
- Caching system for computed distances to avoid redundant calculations
- Supports multiple distance metrics via scipy

**MAP Framework** (`src/copairs/map/`)
- `mean_average_precision()` calculates mAP scores and p-values
- `average_precision.py` handles individual AP score computation
- Null distribution generation for statistical significance testing
- Multiple testing correction via statsmodels
- Optional caching of null distributions for performance

### Data Flow

1. Profiles (rows) with metadata columns and feature columns are processed by Matcher
2. Matcher generates valid pairs based on metadata constraints
3. Distance/similarity metrics are computed between paired profiles
4. AP scores are calculated for each query profile
5. mAP scores are aggregated by metadata groups with statistical testing

### Important Implementation Details

- The package uses pandas DataFrames as the primary data structure
- DuckDB provides efficient SQL operations for pair generation without loading all data into memory
- Parallel processing is implemented via ThreadPool/ThreadPoolExecutor for compute-intensive operations
- Cache directories can be specified to store precomputed null distributions
- Progress bars via tqdm are optional but enabled by default for long operations