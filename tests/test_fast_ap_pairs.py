"""The pairs the Numba and CUDA AP stages attach for the permutation null."""

import numpy as np
import pandas as pd
import pytest

numba = pytest.importorskip("numba", reason="Numba optional extra is unavailable")

from copairs import compute
from copairs.map import multilabel, average_precision, mean_average_precision
from copairs.nulls import _numba
from tests.markers import AP_BACKENDS
from copairs._backend import backend_module

NULL = dict(
    null_size=2000, threshold=0.05, seed=3, progress_bar=False, null="permutation"
)


def plate_meta(groups=8, replicates=4, plates=3):
    """Return ``groups`` treatments with ``replicates`` profiles on each of ``plates``."""
    names = [f"g{i}" for i in range(groups)]
    return pd.DataFrame(
        {
            "g": np.tile(np.repeat(names, replicates), plates),
            "plate": np.repeat(np.arange(plates), groups * replicates),
        }
    )


def label_meta(n, labels=5, seed=0):
    """Return ``n`` profiles with some of ``labels`` targets each, plus one of their own."""
    has = np.random.default_rng(seed).random((n, labels)) < 0.3
    return pd.DataFrame(
        {
            "target": [
                [f"L{j}" for j in np.flatnonzero(h)] + [f"own{i}"]
                for i, h in enumerate(has)
            ],
            "plate": np.arange(n) % 3,
        }
    )


def plate_ap(feats, backend, **kwargs):
    """APs of :func:`plate_meta`'s treatments, ranked against the same plate."""
    return average_precision(
        plate_meta(), feats, ["g"], [], ["plate"], ["g"], progress_bar=False,
        backend=backend, **kwargs,
    )  # fmt: skip


def label_ap(feats, backend, **kwargs):
    """Multilabel APs of :func:`label_meta`'s targets, ranked against the same plate."""
    return multilabel.average_precision(
        label_meta(len(feats)), feats, ["target"], [], ["plate"], ["target"],
        multilabel_col="target", progress_bar=False, backend=backend, **kwargs,
    )  # fmt: skip


def shared_labels(ap):
    """Return the rows of ``ap`` whose target other profiles share."""
    return ap[ap["target"].str.startswith("L")]


@pytest.mark.parametrize("backend", AP_BACKENDS)
@pytest.mark.parametrize("distance", ["cosine", "euclidean"])
def test_pairs_recompute_the_ap_stage_similarities(backend, distance):
    """The attached pairs give the AP stage's kernel similarities, bit for bit.

    The permutation null re-ranks profiles by them. NumPy's similarities of
    float32 rows, which it computes in float32, differ in rounding.
    """
    feats = np.random.default_rng(1).standard_normal((96, 24)).astype(np.float32)
    kernel = backend_module(backend).pair_similarity(feats, distance)
    numpy = compute.pairwise_similarity_fn(distance)
    _, rank_pairs = plate_ap(feats, backend, distance=distance, return_pairs=True)
    _, label_pairs = label_ap(feats, backend, distance=distance, return_pairs=True)
    # A label table's positives are pairs of its rows, its negatives of profiles.
    for pairs in (rank_pairs.pos, rank_pairs.neg, label_pairs.neg):
        assert not np.array_equal(
            numpy(feats[pairs[:, 0]], feats[pairs[:, 1]]), kernel(pairs)
        )
        np.testing.assert_array_equal(rank_pairs.similarity(pairs), kernel(pairs))
        np.testing.assert_array_equal(label_pairs.similarity(pairs), kernel(pairs))


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_permutation_null_ranks_by_the_kernel_similarities(backend):
    """Rows whose norms overflow float32 keep the kernels' similarities in the null.

    NumPy computes their cosines in float32, where every norm overflows and
    every similarity comes out 0; re-ranked by those, the null would
    reproduce no AP and raise.
    """
    rng = np.random.default_rng(2)
    feats = (
        rng.standard_normal((96, 8))
        + np.repeat(rng.standard_normal((8, 8)), 4, 0)[np.tile(np.arange(32), 3)]
    )  # a shift per treatment of plate_meta
    feats = (feats * 1e20).astype(np.float32)
    ap = plate_ap(feats, backend)
    got = mean_average_precision(ap, ["g"], backend="numba", **NULL)
    assert 0 < got["below_p"].sum() < len(got)
    labels = shared_labels(label_ap(feats, backend))
    got = mean_average_precision(labels, ["target"], backend="numba", **NULL)
    assert ((got["p_value"] > 0) & (got["p_value"] <= 1)).all()


@pytest.mark.parametrize("backend", AP_BACKENDS)
def test_kernel_ap_stages_give_numpy_pvalues(backend):
    """Permutation p-values after a kernel AP stage equal those after NumPy's.

    Integer features make every euclidean similarity exact in both, so the
    AP tables and the pairs (in the multilabel table, each AP row's label
    and profile) must agree for the p-values to.
    """
    base = np.random.default_rng(4).integers(-3, 4, size=(96, 6)).astype(np.float64)
    plate_signal = plate_meta()["g"].isin(["g0", "g1"]).to_numpy()
    label_signal = np.array(["L0" in t for t in label_meta(96)["target"]])
    for ap_fn, sameby, signal in (
        (plate_ap, "g", plate_signal),
        (label_ap, "target", label_signal),
    ):
        feats = base.copy()
        feats[signal, :2] += 4
        got = []
        for b in (backend, "numpy"):
            ap = ap_fn(feats, b, distance="euclidean")
            if sameby == "target":
                ap = shared_labels(ap)
            m = mean_average_precision(ap, [sameby], backend="numba", **NULL)
            got.append(m.sort_values(sameby).reset_index(drop=True))
        pd.testing.assert_frame_equal(*got)
        assert 0 < got[0]["below_p"].sum() < len(got[0])


@pytest.mark.parametrize("null_backend", ["numba", "numpy"])
def test_null_threads_compute_similarities_on_the_callers_numba_threads(
    null_backend, monkeypatch
):
    """The permutation null's threads launch a Numba AP stage's kernel as the caller set them.

    Numba's thread count is per thread, and the null's pool threads
    recompute similarities with the kernel of a Numba AP stage, under the
    NumPy null too, whose draws run no Numba.
    """
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("needs 2 Numba threads to tell the caller's 1 from the default")
    feats = np.random.default_rng(5).standard_normal((96, 8))
    frames = (
        (plate_ap(feats, "numba"), "g"),
        (shared_labels(label_ap(feats, "numba")), "target"),
    )
    launches = []
    launch = _numba._launch

    def recorded(kernel, *args):
        launches.append((kernel.__name__, numba.get_num_threads()))
        return launch(kernel, *args)

    monkeypatch.setattr(_numba, "_launch", recorded)
    before = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        for ap, sameby in frames:
            mean_average_precision(ap, [sameby], backend=null_backend, **NULL)
    finally:
        numba.set_num_threads(before)
    sims = [threads for name, threads in launches if name == "_kernel_sims"]
    assert sims, launches
    assert {threads for _, threads in launches} == {1}
