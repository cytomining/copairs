"""Load and verify the explicitly supplied, read-only PR #156 source snapshot."""

import os
import sys
import hashlib
import subprocess
from pathlib import Path

PIN = "eabb22bf87c65ec12219297d7d85fc8d2b58bf9c"


def load_permutation():
    """Verify tracked package bytes against the Git pin before importing them."""
    source = Path(os.environ["COPAIRS_STUDY_SOURCE"]).resolve()
    repository = Path(__file__).resolve().parents[2]
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", PIN, "src/copairs"], cwd=repository, text=True
    )
    for line in tree.splitlines():
        info, name = line.split("\t")
        expected = info.split()[2]
        data = (source / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source differs from {PIN}: {name}")
    sys.path.insert(0, str(source / "src"))
    from copairs.nulls import permutation

    if (
        Path(permutation.__file__).resolve()
        != source / "src/copairs/nulls/permutation.py"
    ):
        raise RuntimeError("Another copairs installation was imported first")
    return permutation
