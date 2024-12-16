 # Unit tests

We use `pytest` package to implement and run unit tests for copairs.

## Getting started

### Installation

To install copairs with test dependencies, check out code locally and install as:
```bash
pip install -e .[test]
```

### Running tests
To execute all tests, run:
```bash
pytest
```

Each individual `test_filename.py` file implements tests for particular features in the corresponding `copairs/filename.py`.

To run tests for a particular source file, specify its test file:
```bash
pytest tests/test_map.py
```
