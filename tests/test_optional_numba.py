"""Tests that Numba stays optional: lazy import, errors, backend checks."""

import numpy as np
import pytest

from copairs.map import multilabel, average_precision
from tests.helpers import run_isolated
from copairs.matching import find_pairs_multilabel
from copairs.map.normalization import normalize_ap

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


def test_default_path_does_not_import_numba():
    """Importing copairs and default AP, multilabel AP and mAP runs skip numba."""
    result = run_isolated(
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
        "meta['m'] = meta['l'].map(lambda v: [v])\n"
        "map.multilabel.average_precision(meta, feats, ['m'], [], [], ['m'],\n"
        "                                 'm', progress_bar=False)\n"
        "assert not any(n == 'numba' or n.startswith('numba.') for n in sys.modules)\n"
        "assert 'copairs.fastap' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_multilabel_matching_skips_scipy():
    """find_pairs_multilabel on NumPy loads neither SciPy nor Numba, as on 0.5.5.

    ``import copairs`` imports only copairs.matching; resolving the backend
    must not load copairs.compute, which imports SciPy.
    """
    result = run_isolated(
        "import sys\n"
        "import pandas as pd\n"
        "import copairs\n"
        "from copairs.matching import find_pairs_multilabel\n"
        "df = pd.DataFrame({'l': [['a'], ['a', 'b'], ['b'], ['c']],\n"
        "                   'p': ['p', 'q', 'p', 'q']})\n"
        "find_pairs_multilabel(df, ['l'], ['p'], 'l')\n"
        "find_pairs_multilabel(df, [], ['l'], 'l', backend='numpy')\n"
        "for name in ('scipy', 'numba', 'copairs.compute', 'copairs.fastap'):\n"
        "    assert not any(m == name or m.startswith(name + '.')\n"
        "                   for m in sys.modules), name\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "call",
    [
        "average_precision(None, None, [], [], [], [], backend='numba')",
        "multilabel.average_precision(None, None, [], [], [], [], 'l', "
        "backend='numba')",
        "find_pairs_multilabel(None, ['l'], [], 'l', backend='numba')",
        "normalize_ap(0.5, 3, 10, backend='numba')",
    ],
)
def test_missing_numba_extra_has_actionable_error(call):
    """Selecting Numba without the extra reports the installation command."""
    result = run_isolated(
        BLOCK_NUMBA.format(
            error="ModuleNotFoundError(\"No module named 'numba'\", name='numba')"
        )
        + "from copairs.map import average_precision, multilabel\n"
        "from copairs.matching import find_pairs_multilabel\n"
        "from copairs.map.normalization import normalize_ap\n"
        "try:\n"
        f"    {call}\n"
        "except ImportError as exc:\n"
        "    assert \"pip install 'copairs[numba]'\" in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('missing Numba did not raise ImportError')\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "error",
    [
        "ModuleNotFoundError(\"No module named 'llvmlite'\", name='llvmlite')",
        "ImportError('incompatible numba build')",
    ],
    ids=["transitive_missing_module", "incompatible_build"],
)
def test_numba_import_failure_is_not_masked(error):
    """A broken or incompatible Numba installation keeps its original error."""
    result = run_isolated(
        BLOCK_NUMBA.format(error=error) + "from copairs.map import average_precision\n"
        "try:\n"
        "    average_precision(None, None, [], [], [], [], backend='numba')\n"
        "except ImportError as exc:\n"
        f"    expected = {error}\n"
        "    assert type(exc) is type(expected), exc\n"
        "    assert (str(exc), exc.name) == (str(expected), expected.name), exc\n"
        "else:\n"
        "    raise AssertionError('the import failure was masked')\n"
    )
    assert result.returncode == 0, result.stderr


def test_fastap_does_not_import_nulls():
    """The AP kernels stand alone: copairs.nulls and copairs.map stay unloaded.

    copairs.map loads statsmodels.
    """
    pytest.importorskip("numba", reason="Numba optional extra is unavailable")
    result = run_isolated(
        "import sys\n"
        "import copairs.fastap\n"
        "for name in ('copairs.nulls', 'copairs.map', 'statsmodels'):\n"
        "    assert not any(m == name or m.startswith(name + '.')\n"
        "                   for m in sys.modules), name\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["auto", "cuda", "Numba", None])
def test_unknown_backend_raises(backend):
    """Backends other than numpy and numba raise before any work."""
    with pytest.raises(ValueError, match="'numpy' or 'numba'"):
        average_precision(None, None, [], [], [], [], backend=backend)
    with pytest.raises(ValueError, match="'numpy' or 'numba'"):
        multilabel.average_precision(None, None, [], [], [], [], "l", backend=backend)
    with pytest.raises(ValueError, match="'numpy' or 'numba'"):
        find_pairs_multilabel(None, ["l"], [], "l", backend=backend)
    with pytest.raises(ValueError, match="'numpy' or 'numba'"):
        normalize_ap(np.array([0.5]), 3, 10, backend=backend)
