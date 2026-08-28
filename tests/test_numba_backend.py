"""Tests for the optional Numba cosine backend."""

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copairs import compute
from copairs.map import average_precision
from copairs.map.multilabel import average_precision as multilabel_average_precision

PROJECT_ROOT = Path(__file__).parents[1]


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _with_layout(feats: np.ndarray, layout: str) -> np.ndarray:
    if layout == "fortran":
        return np.asfortranarray(feats)
    backing = np.empty((len(feats), feats.shape[1] * 2), dtype=feats.dtype)
    backing[:, ::2] = feats
    return backing[:, ::2]


def test_default_import_does_not_import_numba():
    """The package and default AP backend do not import the optional extra."""
    result = _run_isolated(
        "import sys\n"
        "from copairs import map\n"
        "assert not any(n == 'numba' or n.startswith('numba.') for n in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr


def test_missing_numba_extra_has_actionable_error():
    """Selecting Numba without the extra reports the installation command."""
    result = _run_isolated(
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def blocked_import(name, *args, **kwargs):\n"
        "    if name == 'numba' or name.startswith('numba.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'numba'\", name='numba')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked_import\n"
        "from copairs.map import average_precision\n"
        "try:\n"
        "    average_precision(None, None, [], [], [], [], backend='numba')\n"
        "except ImportError as exc:\n"
        "    assert \"pip install 'copairs[numba]'\" in str(exc)\n"
        "else:\n"
        "    raise AssertionError('missing Numba did not raise ImportError')\n"
    )
    assert result.returncode == 0, result.stderr


def test_numba_transitive_missing_module_is_not_masked():
    """A broken Numba installation retains its original missing-module error."""
    result = _run_isolated(
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def broken_import(name, *args, **kwargs):\n"
        "    if name == 'numba':\n"
        "        raise ModuleNotFoundError(\"No module named 'llvmlite'\", name='llvmlite')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = broken_import\n"
        "from copairs.map import average_precision\n"
        "try:\n"
        "    average_precision(None, None, [], [], [], [], backend='numba')\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.name == 'llvmlite'\n"
        "    assert 'copairs[numba]' not in str(exc)\n"
        "else:\n"
        "    raise AssertionError('transitive import failure was masked')\n"
    )
    assert result.returncode == 0, result.stderr


def test_incompatible_numba_import_error_is_not_masked():
    """An incompatible Numba installation retains its original ImportError."""
    result = _run_isolated(
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def broken_import(name, *args, **kwargs):\n"
        "    if name == 'numba':\n"
        "        raise ImportError('incompatible numba build')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = broken_import\n"
        "from copairs.map import average_precision\n"
        "try:\n"
        "    average_precision(None, None, [], [], [], [], backend='numba')\n"
        "except ImportError as exc:\n"
        "    assert str(exc) == 'incompatible numba build'\n"
        "else:\n"
        "    raise AssertionError('incompatible import failure was masked')\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("api", [average_precision, multilabel_average_precision])
def test_numba_rejects_unsupported_distance_before_import(api):
    """Numba never silently falls back for non-cosine distances."""
    args = (None, None, [], [], [], [])
    if api is multilabel_average_precision:
        args += ("labels",)
    with pytest.raises(ValueError, match="only distance='cosine'"):
        api(*args, distance="euclidean", backend="numba")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_numba_regular_exact_ap_parity_with_ties(dtype, layout):
    """Numba preserves regular AP ranking and exact tie behavior."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    meta = pd.DataFrame(
        {
            "compound": ["a", "a", "b", "b", "c", "c"],
            "plate": ["p1", "p2", "p1", "p2", "p1", "p2"],
        }
    )
    feats = np.tile(
        np.asarray([0.125, -1.5, 2.25, 0.75, -0.0625], dtype=dtype),
        (len(meta), 1),
    )
    kwargs = {
        "meta": meta,
        "feats": _with_layout(feats, layout),
        "pos_sameby": ["compound"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["compound"],
        "progress_bar": False,
        "batch_size": 2,
    }

    numpy_result = average_precision(backend="numpy", **kwargs)
    numba_result = average_precision(backend="numba", **kwargs)

    pd.testing.assert_frame_equal(numba_result, numpy_result, check_exact=True)
    np.testing.assert_array_equal(numba_result["average_precision"], 1.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_numba_multilabel_exact_ap_parity_with_ties(dtype, layout):
    """Numba preserves multilabel grouping, ranking, and exact ties."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    meta = pd.DataFrame(
        {
            "compound": ["a", "b", "c", "d"],
            "target": [["x"], ["x", "y"], ["y"], ["z"]],
        }
    )
    feats = np.tile(
        np.asarray([0.125, -1.5, 2.25, 0.75, -0.0625], dtype=dtype),
        (len(meta), 1),
    )
    kwargs = {
        "meta": meta,
        "feats": _with_layout(feats, layout),
        "pos_sameby": ["target"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["target"],
        "multilabel_col": "target",
        "progress_bar": False,
        "batch_size": 2,
    }

    numpy_result = multilabel_average_precision(backend="numpy", **kwargs)
    numba_result = multilabel_average_precision(backend="numba", **kwargs)

    pd.testing.assert_frame_equal(numba_result, numpy_result, check_exact=True)
    np.testing.assert_array_equal(numba_result["average_precision"], 1.0)


def _assert_exact_cosine_pairs(feats: np.ndarray, pairs: np.ndarray) -> None:
    """Assert bitwise float32 output parity for the two cosine backends."""
    numpy_result = compute.cosine_pairs(feats, pairs, batch_size=11, progress_bar=False)
    numba_result = compute._get_cosine_pairs_fn("numba")(
        feats, pairs, batch_size=11, progress_bar=False
    )
    assert numpy_result.dtype == numba_result.dtype == np.float32
    np.testing.assert_array_equal(
        numba_result.view(np.uint32), numpy_result.view(np.uint32)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_exact_reduction_at_numpy_pairwise_boundaries(dtype):
    """Reduction grouping is exact around NumPy's 8-way and 128-item blocks."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    widths = [
        0,
        1,
        2,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1023,
        1024,
        1025,
    ]
    large = dtype(2**20 if dtype is np.float32 else 2**50)
    for width in widths:
        cancellation = np.resize(
            np.asarray([large, 1, -large, -1, 3, -2, 0.5, -0.5], dtype=dtype),
            width,
        )
        rng = np.random.default_rng(width)
        feats = np.stack(
            [
                np.ones(width, dtype=dtype),
                cancellation,
                rng.standard_normal(width).astype(dtype),
                rng.standard_normal(width).astype(dtype),
            ]
        )
        pairs = np.asarray([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=np.int64)
        _assert_exact_cosine_pairs(feats, pairs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_exact_reduction_for_randomized_wide_vectors(dtype):
    """Wide randomized vectors retain bitwise NumPy cosine output parity."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    for width in [129, 257, 1025, 4097]:
        rng = np.random.default_rng(width + np.dtype(dtype).itemsize)
        feats = rng.standard_normal((24, width)).astype(dtype)
        pairs = rng.integers(0, len(feats), size=(97, 2), dtype=np.int64)
        _assert_exact_cosine_pairs(feats, pairs)


def test_numba_cancellation_sensitive_regular_ap_near_tie():
    """Pairwise grouping preserves a cancellation-sensitive AP ordering."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    width = 129
    values = np.resize(
        np.asarray(
            [1e8, -1e8, 3.0, -2.0, 1.0, 1e-3, -1e-3, 7.0],
            dtype=np.float32,
        ),
        width,
    )
    rng = np.random.default_rng(121)
    feats = np.stack([rng.permutation(values) for _ in range(4)])
    meta = pd.DataFrame({"compound": ["a", "a", "b", "b"]})
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["compound"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["compound"],
        "progress_bar": False,
    }

    normalized = compute.prepare_cosine(feats, np.arange(len(feats)))
    sensitive_pairs = np.asarray([[2, 3], [1, 3]], dtype=np.int64)
    products = normalized[sensitive_pairs[:, 0]] * normalized[sensitive_pairs[:, 1]]
    numpy_sims = np.sum(products, axis=1).astype(np.float32)
    left_fold_sims = np.add.accumulate(products, axis=1)[:, -1].astype(np.float32)
    assert numpy_sims[0] > numpy_sims[1]
    assert left_fold_sims[0] < left_fold_sims[1]
    _assert_exact_cosine_pairs(normalized, sensitive_pairs)

    numpy_result = average_precision(backend="numpy", **kwargs)
    numba_result = average_precision(backend="numba", **kwargs)
    pd.testing.assert_frame_equal(numba_result, numpy_result, check_exact=True)
    np.testing.assert_array_equal(
        numba_result["average_precision"],
        np.asarray([0.5, 1.0, 1.0 / 3.0, 1.0]),
    )


def test_numba_referenced_nonfinite_zero_width_and_empty_pairs():
    """Referenced nonfinite, zero-feature, and empty inputs preserve semantics."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    feats = np.asarray([[0.0, 0.0], [np.inf, 1.0], [1.0, 2.0]])
    with np.errstate(all="ignore"):
        normalized = compute.prepare_cosine(feats, np.arange(len(feats)))
    pairs = np.asarray([[0, 2], [1, 2], [2, 2]], dtype=np.int64)
    numpy_result = compute.cosine_pairs(normalized, pairs, 2, progress_bar=False)
    numba_result = compute._get_cosine_pairs_fn("numba")(
        normalized, pairs, 2, progress_bar=False
    )
    np.testing.assert_array_equal(numba_result, numpy_result)
    assert np.isnan(numba_result[:2]).all()
    assert numba_result[2] == np.float32(1.0)

    zero_width = np.empty((3, 0), dtype=np.float32)
    _assert_exact_cosine_pairs(zero_width, pairs)
    zero_result = compute._get_cosine_pairs_fn("numba")(
        zero_width, pairs, 2, progress_bar=False
    )
    np.testing.assert_array_equal(zero_result.view(np.uint32), np.zeros(3, np.uint32))

    empty_pairs = np.empty((0, 2), dtype=np.int64)
    for dtype in (np.float32, np.float64):
        empty_result = compute._get_cosine_pairs_fn("numba")(
            np.empty((3, 17), dtype=dtype),
            empty_pairs,
            2,
            progress_bar=False,
        )
        assert empty_result.dtype == np.float32
        assert empty_result.shape == (0,)


def test_numba_skips_unpaired_nonfinite_rows():
    """Numba normalizes only paired profiles and keeps the NumPy result layout."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    meta = pd.DataFrame(
        {
            "compound": ["a", "a", "b", "b", "zero", "inf"],
            "cohort": ["main", "main", "main", "main", "zero", "inf"],
        }
    )
    feats = np.asarray(
        [
            [1.0, 0.5],
            [0.75, 1.0],
            [-0.5, 1.0],
            [1.0, -0.25],
            [0.0, 0.0],
            [np.inf, 1.0],
        ]
    )
    kwargs = {
        "meta": meta,
        "feats": feats,
        "pos_sameby": ["compound"],
        "pos_diffby": [],
        "neg_sameby": ["cohort"],
        "neg_diffby": ["compound"],
        "progress_bar": False,
        "batch_size": 2,
    }

    with np.errstate(all="raise"):
        numpy_result = average_precision(backend="numpy", **kwargs)
        numba_result = average_precision(backend="numba", **kwargs)

    pd.testing.assert_frame_equal(numba_result, numpy_result, check_exact=True)


def test_numba_cosine_output_is_float32():
    """The optional backend retains the established cosine output dtype."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")

    feats = np.asarray([[1.0, 2.0], [-2.0, 1.0], [0.5, 0.25]], dtype=np.float64)
    normalized = compute.prepare_cosine(feats, np.arange(len(feats)))
    pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    result = compute._get_cosine_pairs_fn("numba")(
        normalized, pairs, batch_size=1, progress_bar=False
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result, compute.cosine_pairs(normalized, pairs, 1, progress_bar=False)
    )


def test_numba_warms_kernel_before_parallel_map(monkeypatch):
    """Compilation is triggered before the Python thread pool receives work."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    from copairs import _numba

    events = []

    def fake_kernel(feats, pairs, result, start, stop):
        events.append(("kernel", start, stop))
        for i in range(start, stop):
            result[i] = np.sum(feats[pairs[i, 0]] * feats[pairs[i, 1]])

    def fake_parallel_map(par_func, items, progress_bar=True):
        events.append(("parallel", progress_bar))
        for item in items:
            par_func(item)

    monkeypatch.setattr(_numba, "_cosine_pairs_range", fake_kernel)
    monkeypatch.setattr(_numba, "parallel_map", fake_parallel_map)
    result = _numba.cosine_pairs(
        np.eye(3), np.asarray([[0, 0], [1, 1], [2, 2]]), 2, progress_bar=False
    )

    assert events[:2] == [("kernel", 0, 0), ("parallel", False)]
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.ones(3, dtype=np.float32))
