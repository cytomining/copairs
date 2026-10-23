"""Tests for the optional CUDA backend (CuPy, ``pip install 'copairs[gpu]'``).

Tests marked ``needs_gpu`` run only where CuPy compiles kernels for a visible
GPU. The others cover machines without one: CuPy missing, CuPy installed
without a usable GPU, a GPU whose kernels fail to compile, and ``"auto"``
staying on the CPU where only an explicit ``"cuda"`` runs the GPU.
"""

import types
import warnings
import importlib.util
import multiprocessing

import numpy as np
import pandas as pd
import pytest

from copairs import nulls, compute
from copairs.map import mean_average_precision
from copairs.nulls import backend, pvalues, sampler
from tests.helpers import run_isolated, without_modules, brute_map_pvalues

HAS_CUPY = importlib.util.find_spec("cupy") is not None
HAS_NUMBA = importlib.util.find_spec("numba") is not None
GPU = "cuda" in nulls.available_backends()
needs_cupy = pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CuPy and a usable CUDA GPU")

CONFS = np.array(
    [[1, 1], [1, 7], [2, 2], [3, 10], [5, 20], [9, 49], [40, 45], [99, 1099], [3, 1000]]
)
# The guided gap search, then float32-stepped gap loops below 2**24 ranks.
# Without its error band the float32 loop changed 671 of 30,000 samples of
# similar loop configurations in a CPU emulation of the kernel.
WIDE = np.array(
    [[2, 5000], [3, 3000], [8, 20000], [4, 70000], [200, 30_000], [1000, 5000]]
)

# Calls that select the CUDA backend, run in a child interpreter: CALLS for the
# nulls, AP_CALLS for the AP stage.
_SETUP = """
import sys
import importlib.util
import numpy as np
import pandas as pd
from copairs import nulls, compute, matching
from copairs.map import average_precision, mean_average_precision, multilabel
from copairs.map.normalization import normalize_ap

FRAME = pd.DataFrame({"g": ["a", "a"], "average_precision": [0.5, 0.6],
                      "normalized_average_precision": 0.0, "n_pos_pairs": 1,
                      "n_total_pairs": 10})
CALLS = [
    "nulls.ap_nulls([[2, 5]], 4, 0, backend='cuda')",
    "nulls.ap_pvalues([0.5], [0], [[2, 5]], 4, 0, backend='cuda')",
    "nulls.map_pvalues([0.5], [0, 1], [0], [1], [[2, 5]], 4, 0, backend='cuda')",
    "compute.get_null_dists(np.array([[2, 10]]), 10, 0, progress_bar=False,"
    " backend='cuda')",
    "compute.p_values(np.array([0.5]), [[2, 10]], 10, 0, progress_bar=False,"
    " backend='cuda')",
    "compute.p_values(np.array([]), np.empty((0, 2)), 10, 0, backend='cuda')",
    "mean_average_precision(FRAME, ['g'], 10, 0.05, 0, progress_bar=False,"
    " backend='cuda')",
    "mean_average_precision(FRAME.iloc[:0], ['g'], 10, 0.05, 0, backend='cuda')",
]
META = pd.DataFrame({"l": list("aabb"), "p": list("xyxy")})
LABELS = META.assign(l=[["a"], ["a", "b"], ["b"], ["b"]])
AP_CALLS = [
    "average_precision(META, np.eye(4), ['l'], ['p'], [], ['l'], backend='cuda')",
    "multilabel.average_precision(LABELS, np.eye(4), ['l'], ['p'], [], ['l'], 'l',"
    " backend='cuda')",
    "matching.find_pairs_multilabel(LABELS, ['l'], ['p'], 'l', backend='cuda')",
    "normalize_ap(np.array([0.5]), 3, 10, backend='cuda')",
]
if importlib.util.find_spec("numba"):
    from copairs import fastap

    AP_CALLS.append(
        "fastap.draw_average_precisions(np.eye(4), [[0, 1]], [[2, 3]], backend='cuda')"
    )


def raises(error, *needles, calls=CALLS + AP_CALLS):
    for call in calls:
        try:
            eval(call)
        except error as exc:
            assert all(n in str(exc) for n in needles), (call, exc)
        else:
            raise AssertionError(f"{call} did not raise {error.__name__}")
"""


def test_cuda_without_cupy_names_the_extra():
    """Without CuPy, "auto" stays on the CPU and "cuda" names the gpu extra."""
    result = run_isolated(
        without_modules("cupy")
        + _SETUP
        + "assert 'cuda' not in nulls.available_backends()\n"
        "assert nulls.resolve_backend('auto') != 'cuda'\n"
        "raises(ImportError, 'CuPy', \"pip install 'copairs[gpu]'\")\n"
        "assert 'cupy' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


@needs_cupy
@pytest.mark.skipif(GPU, reason="a CUDA GPU is usable here")
def test_cuda_without_a_gpu():
    """With CuPy but no usable GPU, "auto" uses the CPU silently; "cuda" raises.

    A missing driver or device is not a misconfiguration, so it warns nothing;
    the error of an explicit "cuda" names the gpu extra and the reason.
    """
    result = run_isolated(
        _SETUP + "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    backends = nulls.available_backends()\n"
        "    auto = nulls.resolve_backend('auto')\n"
        "assert 'cuda' not in backends and auto != 'cuda', (backends, auto)\n"
        "shown = [str(w.message) for w in caught if 'CUDA' in str(w.message)\n"
        "         or 'CuPy' in str(w.message)]\n"
        "assert not shown, shown\n"
        "raises(RuntimeError, 'found none', \"pip install 'copairs[gpu]'\")\n"
    )
    assert result.returncode == 0, result.stderr


def test_auto_skips_a_cupy_that_fails_to_import(tmp_path):
    """An installed CuPy that fails to import is not listed, with one warning.

    ``backend="cuda"`` raises the import error itself.
    """
    fake = tmp_path / "cupy"
    fake.mkdir()
    (fake / "__init__.py").write_text('raise ImportError("simulated broken cupy")\n')
    result = run_isolated(
        f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
        + _SETUP
        + "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    assert 'cuda' not in nulls.available_backends()\n"
        "    assert nulls.resolve_backend('auto') != 'cuda'\n"
        "shown = [str(w.message) for w in caught if 'CuPy' in str(w.message)]\n"
        "assert len(shown) == 1 and 'simulated broken cupy' in shown[0], shown\n"
        "raises(ImportError, 'simulated broken cupy')\n"
    )
    assert result.returncode == 0, result.stderr


@needs_cupy
def test_cuda_ap_stage_without_numba_names_the_extra():
    """The CUDA AP stage runs some steps on Numba, so without it "cuda" names the extra.

    The probe is stubbed to find a usable GPU, so the Numba import is what fails.
    """
    check = (
        "from copairs import _cuda\n"
        "_cuda.require = lambda: None\n"
        "assert len(AP_CALLS) == 4\n"
        "raises(ImportError, 'Numba', \"pip install 'copairs[gpu]'\", calls=AP_CALLS)\n"
    )
    result = run_isolated(without_modules("numba", "llvmlite") + _SETUP + check)
    assert result.returncode == 0, result.stderr


@needs_cupy
def test_import_does_not_load_cupy():
    """Importing copairs and running the default paths leave CuPy unloaded.

    ``"auto"``, the null functions' default, never runs the GPU, so it does
    not import CuPy either; only ``"cuda"`` and ``available_backends`` do.
    """
    result = run_isolated(
        "import sys\n"
        "import numpy as np\n"
        "import copairs, copairs.map, copairs.compute, copairs.nulls\n"
        "import copairs.map.multilabel, copairs.nulls.sampler\n"
        + ("import copairs.fastap\n" if HAS_NUMBA else "")
        + "from copairs import map, nulls\n"
        "from tests.helpers import simulate_random_dframe\n"
        "rng = np.random.default_rng(0)\n"
        "meta = simulate_random_dframe(60, {'p': 3, 'l': 6}, ['l'], ['p'], rng)\n"
        "feats = rng.normal(size=(len(meta), 4))\n"
        "ap = map.average_precision(meta, feats, ['l'], ['p'], [], ['l'],\n"
        "                           progress_bar=False)\n"
        "map.mean_average_precision(ap, ['l'], null_size=10, threshold=0.05,\n"
        "                           seed=0, progress_bar=False)\n"
        "nulls.ap_nulls([[2, 5]], 4, 0, backend='numpy')\n"
        "nulls.ap_nulls([[2, 5]], 4, 0)\n"
        "nulls.ap_pvalues([0.5], [0], [[2, 5]], 4, 0)\n"
        "assert nulls.resolve_backend('auto') != 'cuda'\n"
        + ("nulls.ap_nulls([[2, 5]], 4, 0, backend='numba')\n" if HAS_NUMBA else "")
        + "assert 'cupy' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def cuda_runtime(monkeypatch):
    """Yield :mod:`copairs._cuda` with a fresh probe cache, cleared afterwards."""
    from copairs import _cuda

    _cuda._probe.cache_clear()
    yield _cuda
    monkeypatch.undo()
    _cuda._probe.cache_clear()


@needs_cupy
def test_cuda_unavailable_when_kernels_cannot_compile(cuda_runtime, monkeypatch):
    """A visible GPU whose kernels fail to compile is not offered, with one warning.

    ``backend="cuda"`` then raises a RuntimeError that names the failure.
    """

    def broken(*args, **kwargs):
        raise RuntimeError("NVRTC not found")

    monkeypatch.setattr(cuda_runtime.cp, "RawKernel", broken)
    monkeypatch.setattr(cuda_runtime.cp.cuda.runtime, "getDeviceCount", lambda: 1)
    with pytest.warns(RuntimeWarning, match="fail to compile.*NVRTC not found"):
        assert "cuda" not in nulls.available_backends()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # probed once per process: no second warning
        assert nulls.resolve_backend("auto") != "cuda"
        with pytest.raises(RuntimeError, match=r"fail to compile.*copairs\[gpu\]"):
            nulls.ap_nulls(CONFS, 4, seed=0, backend="cuda")


@needs_cupy
def test_cuda_probed_once_per_process(cuda_runtime, monkeypatch):
    """The probe runs once per process, and again in a forked child (new pid).

    A child forked after its parent initialised CUDA cannot use it, so it
    must not inherit the parent's answer.
    """
    calls = []

    def device_count():
        calls.append(1)
        return 0

    pid = [100]
    monkeypatch.setattr(cuda_runtime.cp.cuda.runtime, "getDeviceCount", device_count)
    monkeypatch.setattr(
        cuda_runtime, "os", types.SimpleNamespace(getpid=lambda: pid[0])
    )
    assert not cuda_runtime.usable() and not cuda_runtime.usable()
    assert len(calls) == 1
    pid[0] = 101
    assert not cuda_runtime.usable()
    assert len(calls) == 2


class _HostCuda:
    """Stands in for :mod:`copairs.nulls.cuda` on the CPU, recording its calls.

    Samples and counts with the NumPy backend, so p-values must equal NumPy's.
    """

    def __init__(self):
        self.calls = 0
        self.cp = np

    def plan_on_device(self, plan):
        return plan

    def host(self, x):
        return np.asarray(x)

    def ap_nulls(self, plan, start, size, dtype):
        self.calls += 1
        return sampler._ap_nulls(plan, size, start, "numpy", dtype)

    class ScoreCounter:
        def __init__(self, thr):
            self.thr = thr
            self._counts = np.zeros(len(thr), dtype=np.int64)

        def __call__(self, null, ptr):
            pvalues._count_ge_host(null, self.thr, ptr, self._counts)

        def counts(self):
            return self._counts

    class GroupCounter:
        def __init__(self, *args):
            self.args = args
            self._counts = np.zeros(len(args[0]) - 1, dtype=np.int64)

        def __call__(self, null):
            pvalues._group_ge_host(null, *self.args, self._counts)

        def counts(self):
            return self._counts


@pytest.fixture
def host_cuda(monkeypatch):
    """Make a GPU look usable and run :class:`_HostCuda` in place of its kernels."""
    fake = _HostCuda()
    cpu = [name for name in nulls.available_backends() if name != "cuda"]
    for module in (backend, sampler, pvalues):
        monkeypatch.setattr(module, "_import_cuda_backend", lambda: fake)
    monkeypatch.setattr(backend, "_cuda_usable", lambda: True)
    monkeypatch.setattr(backend, "available_backends", lambda: ["cuda", *cpu])
    return fake


def test_auto_never_runs_cuda(host_cuda):
    """With a usable GPU, "auto" still runs on the CPU; only "cuda" runs the GPU.

    The CUDA samples match NumPy's by error bounds, not identical arithmetic,
    and no GPU has tested them bit for bit, so "auto" must not pick them.
    """
    assert nulls.resolve_backend("auto") == ("numba" if HAS_NUMBA else "numpy")
    rng = np.random.default_rng(0)
    confs = np.concatenate([CONFS, WIDE[:4]])
    conf_ix = rng.integers(len(confs), size=40)
    scores = rng.random(40)
    ptr, conf_cnt = np.array([0, 3, 5, len(confs)]), np.arange(1, len(confs) + 1)
    group_args = (ptr, np.arange(len(confs)), conf_cnt, confs)
    numpy = {
        "nulls": nulls.ap_nulls(confs, 300, 7, backend="numpy"),
        "ap": nulls.ap_pvalues(scores, conf_ix, confs, 300, 7, backend="numpy"),
        "map": nulls.map_pvalues(scores[:3], *group_args, 300, 7, backend="numpy"),
    }
    for name, on_gpu in (("auto", False), ("cuda", True)):
        host_cuda.calls = 0
        got = {
            "nulls": nulls.ap_nulls(confs, 300, 7, backend=name),
            "ap": nulls.ap_pvalues(
                scores, conf_ix, confs, 300, 7, backend=name, budget_bytes=2**12
            ),
            "map": nulls.map_pvalues(
                scores[:3], *group_args, 300, 7, backend=name, budget_bytes=2**12
            ),
        }
        if on_gpu:
            assert host_cuda.calls > 3, "the small budget samples in chunks"
        else:
            assert not host_cuda.calls, name
        for key, expected in numpy.items():
            np.testing.assert_array_equal(got[key], expected, err_msg=(name, key))
    host_cuda.calls = 0
    frame = pd.DataFrame(
        {
            "g": list("aabbb"),
            "average_precision": [0.5, 0.6, 0.1, 0.2, 0.9],
            "normalized_average_precision": 0.0,
            "n_pos_pairs": [1, 2, 3, 3, 1],
            "n_total_pairs": 10,
        }
    )
    compute.get_null_dists(confs, 50, 1, progress_bar=False)
    compute.p_values(scores[:5], confs[conf_ix[:5]], 200, 1, progress_bar=False)
    mean_average_precision(frame, ["g"], 200, 0.05, 1, progress_bar=False)
    assert not host_cuda.calls


@needs_cupy
def test_float32_gap_band_covers_rounding():
    """The float32 gap loop's band covers its worst-case rounding at every step.

    After ``g`` steps the float32 product is within ``(1 +- 2^-24)^(2g + 1)``
    of the exact product, the float64 one within ``(1 +- 2^-53)^(2g + 1)``,
    and ``uf`` and each threshold round once more. A float32 step (or stop)
    then implies the float64 step (or stop) for every ``g < F32_MAX_GAP``,
    with ``band``, ``1 + band`` and ``1 - band`` rounded as the kernel does.
    Products stay normal floats, since ``u >= 1e-30`` and factors ``> 2^-24``.
    """
    from copairs.nulls.cuda import _AP_NULLS, F32_MAX_GAP

    assert "(float)(2 * gap + 4) * 1.2e-7f" in _AP_NULLS  # the band checked here
    assert "u < 1e-30" in _AP_NULLS and "remaining >= (1LL << 24)" in _AP_NULLS
    assert 1e-30 * 2.0**-24 * (1 - 2.0**-24) ** 3 > np.finfo(np.float32).tiny
    e32, e64 = 2.0**-24, 2.0**-53
    for start in range(0, F32_MAX_GAP, 1 << 20):
        g = np.arange(start, min(F32_MAX_GAP, start + (1 << 20)))
        band = (2 * g + 4).astype(np.float32) * np.float32(1.2e-7)
        hi = (np.float32(1) + band).astype(np.float64)
        lo = (np.float32(1) - band).astype(np.float64)
        n = 2.0 * g + 1
        # log A_g >= 0: a float32 step implies the float64 product exceeds u.
        log_a = np.log(hi) + 2 * np.log1p(-e32) + n * (np.log1p(-e64) - np.log1p(e32))
        assert log_a.min() > 1e-12, g[np.argmin(log_a)]
        # log B_g <= 0: a float32 stop implies the float64 product is <= u
        # (with lo <= 0 the loop never stops in float32).
        ok = lo > 0
        log_b = np.log(lo[ok]) + 2 * np.log1p(e32)
        log_b += n[ok] * (np.log1p(e64) - np.log1p(-e32))
        assert not ok.any() or log_b.max() < -1e-12, g[ok][np.argmax(log_b)]


@needs_gpu
def test_cuda_samples_and_pvalues_bitwise_numpy():
    """CUDA samples and p-values are NumPy's bit for bit.

    The wide configurations step long gaps below 2**24 ranks, where the
    float32 loop runs.
    """
    confs = np.concatenate([CONFS, WIDE])
    for dtype, size in ((np.float64, 4000), (np.float32, 2000)):
        expected = nulls.ap_nulls(confs, size, 3, backend="numpy", dtype=dtype)
        got = nulls.ap_nulls(confs, size, 3, backend="cuda", dtype=dtype)
        assert got.tobytes() == expected.tobytes(), dtype
    rng = np.random.default_rng(1)
    conf_ix = rng.integers(len(confs), size=200)
    scores = nulls.ap_nulls(confs, 1, 11, dtype=np.float64, backend="numpy")[conf_ix, 0]
    kwargs = dict(null_size=5000, seed=2, budget_bytes=2**16)
    a = nulls.ap_pvalues(scores, conf_ix, confs, backend="numpy", **kwargs)
    b = nulls.ap_pvalues(scores, conf_ix, confs, backend="cuda", **kwargs)
    assert a.tobytes() == b.tobytes()


@needs_gpu
def test_cuda_counts_match_host():
    """The GPU counters equal the NumPy counts, NaN samples and ties included."""
    from copairs.nulls import cuda

    rng = np.random.default_rng(3)
    lengths = [0, 1, 37, 200, 3]
    null = rng.integers(0, 50, size=(len(lengths), 3000)) / 50
    null[1, :5] = np.nan
    pad = 4  # the segments start past the first thresholds
    ptr = pad + np.concatenate([[0], np.cumsum(lengths)])
    thr = rng.integers(0, 50, size=ptr[-1]) / 50
    for c in range(len(lengths)):
        thr[ptr[c] : ptr[c + 1]].sort()
    expected = np.zeros(len(thr), dtype=np.int64)
    pvalues._count_ge_host(null, thr, ptr, expected)
    counter = cuda.ScoreCounter(thr)
    counter(cuda.cp.asarray(null), ptr)
    np.testing.assert_array_equal(counter.counts(), expected)

    gptr, conf_ix, conf_cnt = (
        np.array([0, 2, 5]),
        np.array([0, 2, 1, 3, 4]),
        [3, 1, 2, 0, 5],
    )
    n_group = np.add.reduceat(conf_cnt, gptr[:-1])
    gthr = np.array([0.4, 0.55])
    expected = np.zeros(2, dtype=np.int64)
    pvalues._group_ge_host(null, gptr, conf_ix, conf_cnt, n_group, gthr, expected)
    group = cuda.GroupCounter(gptr, conf_ix, conf_cnt, n_group, gthr)
    group(cuda.cp.asarray(null))
    np.testing.assert_array_equal(group.counts(), expected)


@needs_gpu
def test_cuda_map_pvalues_match_brute_force():
    """CUDA mAP p-values equal the brute-force count over materialized nulls."""
    ptr, conf_ix, conf_cnt = [0, 2, 5], [0, 3, 1, 4, 8], [2, 1, 1, 3, 1]
    scores = np.array([0.31, 0.62])
    expected = brute_map_pvalues(scores, ptr, conf_ix, conf_cnt, CONFS, 3000, 4)
    got = nulls.map_pvalues(scores, ptr, conf_ix, conf_cnt, CONFS, 3000, 4, "cuda")
    np.testing.assert_array_equal(got, expected)


def _child_backend():
    # Samples on NumPy: Numba's GNU OpenMP layer, which earlier tests may have
    # started, terminates forked children that run parallel kernels.
    return nulls.available_backends(), nulls.ap_nulls(CONFS, 50, 4, backend="numpy")


@needs_gpu
def test_forked_child_finds_no_gpu():
    """A process forked after CUDA was initialised does not list the GPU."""
    expected = nulls.ap_nulls(CONFS, 50, seed=4, backend="cuda")  # initialises CUDA
    with multiprocessing.get_context("fork").Pool(1) as pool:
        backends, null = pool.apply_async(_child_backend).get(timeout=300)
    assert "cuda" not in backends
    np.testing.assert_array_equal(null, expected)
