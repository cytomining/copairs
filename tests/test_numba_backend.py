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
from copairs.map.average_precision import build_rank_lists

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


def _assert_exact_rank_lists(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
) -> None:
    """Assert exact rank-list, AP, configuration, and p-value parity."""
    numpy_ranks = build_rank_lists(pos_pairs, neg_pairs, pos_sims, neg_sims)
    numba_ranks = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims, backend="numba"
    )

    for numba_value, numpy_value in zip(numba_ranks, numpy_ranks):
        assert numba_value.dtype == numpy_value.dtype
        assert numba_value.shape == numpy_value.shape
        assert numba_value.tobytes() == numpy_value.tobytes()

    paired_ix, rel_k_list, counts = numba_ranks
    assert paired_ix.dtype == np.result_type(pos_pairs.dtype, neg_pairs.dtype)
    assert rel_k_list.dtype == np.uint32
    assert counts.dtype == np.uint32

    if not len(counts):
        return

    with np.errstate(all="ignore"):
        numpy_ap, numpy_confs = compute.ap_contiguous(numpy_ranks[1], numpy_ranks[2])
        numba_ap, numba_confs = compute.ap_contiguous(rel_k_list, counts)
    assert numba_ap.dtype == numpy_ap.dtype
    assert numba_ap.tobytes() == numpy_ap.tobytes()
    assert numba_confs.dtype == numpy_confs.dtype
    assert numba_confs.tobytes() == numpy_confs.tobytes()

    # Downstream null configurations and scores are identical, so the existing
    # p-value implementation receives exactly the same seeded inputs.
    if np.all(numpy_confs[:, 0] > 0):
        numpy_p = compute.p_values(numpy_ap, numpy_confs, 19, 123, False)
        numba_p = compute.p_values(numba_ap, numba_confs, 19, 123, False)
        np.testing.assert_array_equal(numba_p, numpy_p)


def test_numba_rank_lists_randomized_exact_reference():
    """Random directed multigraphs exactly match NumPy rank construction."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    rng = np.random.default_rng(90421)
    sparse_ids = np.asarray([2, 11, 10_003, 2**20 + 7], dtype=np.int64)

    for _ in range(50):
        n_pos = int(rng.integers(1, 35))
        n_neg = int(rng.integers(0, 35))
        pos_pairs = rng.choice(sparse_ids, size=(n_pos, 2), replace=True)
        neg_pairs = rng.choice(sparse_ids, size=(n_neg, 2), replace=True)

        # A small score pool creates many intra- and inter-class ties while the
        # random values cover ordinary float32 ordering.
        score_pool = np.concatenate(
            [
                rng.standard_normal(12).astype(np.float32),
                np.asarray([-np.inf, -1.0, -0.0, 0.0, 1.0, np.inf, np.nan]),
            ]
        ).astype(np.float32)
        pos_sims = rng.choice(score_pool, size=n_pos, replace=True)
        neg_sims = rng.choice(score_pool, size=n_neg, replace=True)
        _assert_exact_rank_lists(pos_pairs, neg_pairs, pos_sims, neg_sims)


@pytest.mark.parametrize(
    ("pos_dtype", "neg_dtype", "pos_score", "neg_score"),
    [
        (np.float32, np.float64, 1.0, 1.0 + 2**-30),
        (np.float64, np.float32, 1.0 - 2**-30, 1.0),
    ],
)
def test_numba_rank_lists_mixed_float_scores_exact_reference(
    pos_dtype, neg_dtype, pos_score, neg_score
):
    """Mixed float32/float64 scores promote without losing rank distinctions."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    pos_pairs = np.asarray([[0, 1]], dtype=np.int64)
    neg_pairs = np.asarray([[0, 2]], dtype=np.int64)
    pos_sims = np.asarray([pos_score], dtype=pos_dtype)
    neg_sims = np.asarray([neg_score], dtype=neg_dtype)

    _assert_exact_rank_lists(pos_pairs, neg_pairs, pos_sims, neg_sims)
    _, relevance, _ = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims, backend="numba"
    )
    assert relevance[0] == 0


@pytest.mark.parametrize("score_name", ["pos_sims", "neg_sims"])
@pytest.mark.parametrize(
    "bad_scores",
    [
        np.asarray([2**53, 2**53 + 1], dtype=np.int64),
        np.asarray([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64),
        np.asarray([0.5, 1.0], dtype=np.float16),
        np.asarray([0.5, 1.0], dtype=np.complex128),
        np.asarray([0.5, 1.0], dtype=object),
        np.asarray([0.5, 1.0], dtype=np.dtype(np.float32).newbyteorder("S")),
        np.asarray([0.5, 1.0], dtype=np.dtype(np.float64).newbyteorder("S")),
    ],
    ids=[
        "int64-around-2**53",
        "int64-limits",
        "float16",
        "complex128",
        "object",
        "non-native-float32",
        "non-native-float64",
    ],
)
def test_numba_rank_lists_reject_unsupported_score_dtypes(score_name, bad_scores):
    """Unsupported scores fail before JIT rather than silently changing ranks."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    inputs = {
        "pos_pairs": np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        "neg_pairs": np.asarray([[0, 3], [0, 4]], dtype=np.int64),
        "pos_sims": np.asarray([0.5, 1.0], dtype=np.float32),
        "neg_sims": np.asarray([0.5, 1.0], dtype=np.float32),
    }
    inputs[score_name] = bad_scores

    with pytest.raises(TypeError, match=rf"{score_name}.*native float32 or float64"):
        build_rank_lists(**inputs, backend="numba")


def test_numpy_rank_lists_remains_permissive_for_integer_scores():
    """The default backend retains its existing permissive score behavior."""
    pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    scores = np.asarray([2**53, 2**53 + 1], dtype=np.int64)

    paired_ix, relevance, counts = build_rank_lists(pairs, pairs, scores, scores)

    assert paired_ix.dtype == np.int64
    assert relevance.dtype == np.uint32
    assert counts.dtype == np.uint32


@pytest.mark.parametrize(
    ("input_name", "invalid_value", "message"),
    [
        ("pos_pairs", np.asarray([0, 1], dtype=np.int64), "2-D array"),
        ("neg_pairs", np.empty((2, 3), dtype=np.int64), "exactly two columns"),
        ("pos_sims", np.ones((2, 1), dtype=np.float32), "1-D array"),
        ("neg_sims", np.ones((1, 2), dtype=np.float32), "1-D array"),
        ("pos_sims", np.ones(1, dtype=np.float32), "length must equal"),
        ("neg_sims", np.ones(3, dtype=np.float32), "length must equal"),
    ],
)
def test_numba_rank_lists_validates_shapes_and_lengths(
    input_name, invalid_value, message
):
    """Malformed rank inputs raise ValueError before compiled array reads."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    inputs = {
        "pos_pairs": np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        "neg_pairs": np.asarray([[0, 3], [0, 4]], dtype=np.int64),
        "pos_sims": np.asarray([0.5, 1.0], dtype=np.float32),
        "neg_sims": np.asarray([0.5, 1.0], dtype=np.float32),
    }
    inputs[input_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        build_rank_lists(**inputs, backend="numba")


@pytest.mark.parametrize(
    "bad_pairs",
    [
        np.asarray([[0.0, 1.0]], dtype=np.float64),
        np.asarray([[0, 1]], dtype=object),
        np.asarray([[0, 1]], dtype=np.dtype(np.int64).newbyteorder("S")),
    ],
    ids=["float64", "object", "non-native-int64"],
)
def test_numba_rank_lists_rejects_unsupported_pair_dtypes(bad_pairs):
    """Pair indices use native integer arrays supported by the compiled kernel."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    empty_pairs = np.empty((0, 2), dtype=np.int64)
    empty_sims = np.empty(0, dtype=np.float32)

    with pytest.raises(TypeError, match="native signed or unsigned integer dtype"):
        build_rank_lists(
            bad_pairs,
            empty_pairs,
            np.asarray([0.5], dtype=np.float32),
            empty_sims,
            backend="numba",
        )


def _rank_list_case(case: str):
    """Return an adversarial rank-list input named for its reference behavior."""
    empty_pairs = np.empty((0, 2), dtype=np.int32)
    empty_sims = np.empty(0, dtype=np.float32)
    if case == "ties":
        return (
            np.asarray([[7, 11], [7, 13]], dtype=np.int32),
            np.asarray([[7, 17], [7, 19]], dtype=np.int32),
            np.asarray([0.5, -0.0], dtype=np.float32),
            np.asarray([0.5, 0.0], dtype=np.float32),
        )
    if case == "nan-inf":
        return (
            np.asarray([[1, 2], [1, 3]], dtype=np.int32),
            np.asarray([[1, 4], [1, 5]], dtype=np.int32),
            np.asarray([np.nan, np.inf], dtype=np.float32),
            np.asarray([np.nan, -np.inf], dtype=np.float32),
        )
    if case == "duplicates-overlap-sparse-ids":
        sparse = np.asarray([7, 1_000_003, 2**30 + 9], dtype=np.int64)
        return (
            np.asarray(
                [
                    [sparse[0], sparse[1]],
                    [sparse[0], sparse[1]],
                    [sparse[1], sparse[0]],
                    [sparse[2], sparse[2]],
                ],
                dtype=np.int64,
            ),
            np.asarray(
                [
                    [sparse[0], sparse[1]],
                    [sparse[2], sparse[0]],
                    [99_999_991, 99_999_991],
                    [sparse[1], sparse[2]],
                ],
                dtype=np.int64,
            ),
            np.asarray([1.0, np.nan, -np.inf, np.inf], dtype=np.float32),
            np.asarray([1.0, np.nan, -np.inf, np.inf], dtype=np.float32),
        )
    nonempty_pairs = np.asarray([[5, 9], [5, 5]], dtype=np.int32)
    nonempty_sims = np.asarray([np.nan, -np.inf], dtype=np.float32)
    if case == "negative-only":
        return empty_pairs, nonempty_pairs, empty_sims, nonempty_sims
    if case == "positive-only":
        return nonempty_pairs, empty_pairs, nonempty_sims, empty_sims
    return empty_pairs, empty_pairs, empty_sims, empty_sims


@pytest.mark.parametrize(
    "case",
    [
        "ties",
        "nan-inf",
        "duplicates-overlap-sparse-ids",
        "negative-only",
        "positive-only",
        "both-classes-empty",
    ],
)
def test_numba_rank_lists_adversarial_exact_bytes(case):
    """Adversarial rank lists have byte-exact NumPy results and dtypes."""
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    _assert_exact_rank_lists(*_rank_list_case(case))


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
