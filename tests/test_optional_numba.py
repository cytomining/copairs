"""Tests that Numba stays optional: lazy import, errors, backend checks."""

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from copairs.map import average_precision
from copairs.map.normalization import normalize_ap

PROJECT_ROOT = Path(__file__).parents[1]

# Imports numba lazily in a fresh interpreter whose import of numba fails.
BLOCK_NUMBA = (
    "import builtins\n"
    "original_import = builtins.__import__\n"
    "def blocked_import(name, *args, **kwargs):\n"
    "    if name == 'numba' or name.startswith('numba.'):\n"
    "        raise {error}\n"
    "    return original_import(name, *args, **kwargs)\n"
    "builtins.__import__ = blocked_import\n"
)


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


def test_default_path_does_not_import_numba():
    """Importing copairs and a default AP + mAP run never import numba."""
    result = _run_isolated(
        "import sys\n"
        "import numpy as np\n"
        "import copairs\n"
        "from copairs import map\n"
        "from tests.helpers import simulate_random_dframe\n"
        "rng = np.random.default_rng(0)\n"
        "meta = simulate_random_dframe(60, {'p': 3, 'l': 6}, ['l'], ['p'], rng)\n"
        "feats = rng.normal(size=(len(meta), 4))\n"
        "ap = map.average_precision(meta, feats, ['l'], ['p'], [], ['l'],\n"
        "                           progress_bar=False)\n"
        "map.mean_average_precision(ap, ['l'], null_size=10, threshold=0.05,\n"
        "                           seed=None, progress_bar=False)\n"
        "assert not any(n == 'numba' or n.startswith('numba.') for n in sys.modules)\n"
        "assert 'copairs.fastap' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "call",
    [
        "average_precision(None, None, [], [], [], [], backend='numba')",
        "normalize_ap(0.5, 3, 10, backend='numba')",
    ],
)
def test_missing_numba_extra_has_actionable_error(call):
    """Selecting Numba without the extra reports the installation command."""
    result = _run_isolated(
        BLOCK_NUMBA.format(
            error="ModuleNotFoundError(\"No module named 'numba'\", name='numba')"
        )
        + "from copairs.map import average_precision, multilabel\n"
        "from copairs.map.normalization import normalize_ap\n"
        "try:\n"
        f"    {call}\n"
        "except ImportError as exc:\n"
        "    assert \"pip install 'copairs[numba]'\" in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('missing Numba did not raise ImportError')\n"
    )
    assert result.returncode == 0, result.stderr


def test_numba_transitive_missing_module_is_not_masked():
    """A broken Numba installation keeps its original missing-module error."""
    result = _run_isolated(
        BLOCK_NUMBA.format(
            error="ModuleNotFoundError(\"No module named 'llvmlite'\", name='llvmlite')"
        )
        + "from copairs.map import average_precision\n"
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
    """An incompatible Numba installation keeps its original ImportError."""
    result = _run_isolated(
        BLOCK_NUMBA.format(error="ImportError('incompatible numba build')")
        + "from copairs.map import average_precision\n"
        "try:\n"
        "    average_precision(None, None, [], [], [], [], backend='numba')\n"
        "except ImportError as exc:\n"
        "    assert str(exc) == 'incompatible numba build'\n"
        "else:\n"
        "    raise AssertionError('incompatible import failure was masked')\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["auto", "cuda", "Numba", None])
def test_unknown_backend_raises(backend):
    """Backends other than numpy and numba raise before any work."""
    with pytest.raises(ValueError, match="'numpy' or 'numba'"):
        average_precision(None, None, [], [], [], [], backend=backend)
    with pytest.raises(ValueError, match="'numpy' or 'numba'"):
        normalize_ap(np.array([0.5]), 3, 10, backend=backend)
