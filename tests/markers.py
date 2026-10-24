"""Pytest markers for the optional backends, shared by the test modules.

Not in tests.helpers: finding a GPU imports CuPy, and the child
interpreters that check what ``import copairs`` loads import the helpers.
"""

import importlib.util

import pytest

from copairs._backend import available_backends

HAS_CUPY = importlib.util.find_spec("cupy") is not None
HAS_NUMBA = importlib.util.find_spec("numba") is not None
GPU = "cuda" in available_backends()
needs_cupy = pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CuPy and a usable CUDA GPU")
# "cuda" computes similarities on the GPU and the rest as "numba" does.
AP_BACKENDS = ["numba", pytest.param("cuda", marks=needs_gpu)]
