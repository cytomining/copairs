"""Helper functions for testing."""

import pandas as pd

from copairs.matching import Matcher, find_pairs

SEED = 42

# Cols are c, p and w
sameby = ["c", "w"]
diffby = []
n_compounds = 150000
n_replicates = 20
plate_size = 384

def simulate_plates(n_compounds, n_replicates, plate_size):
    """Round robin creation of platemaps."""
    total = n_compounds * n_replicates

    compounds = []
    plates = []
    wells = []
    for i in range(total):
        compound_id = i % n_compounds
        well_id = i % plate_size
        plate_id = i // plate_size
        compounds.append(f"c{compound_id}")
        plates.append(f"p{plate_id}")
        wells.append(f"w{well_id}")

    dframe = pd.DataFrame({"c": compounds, "p": plates, "w": wells})
    return dframe

# Gen data
dframe = simulate_plates(n_compounds, n_replicates, plate_size)

# Load matcher
matcher = Matcher(dframe, dframe.columns, seed=SEED)

"""
#Evaluate correctness
pairs = find_pairs(dframe,sameby, diffby)
for id1, id2, *_ in pairs.fetchall():
    for col in sameby:
        assert dframe.loc[id1, col] == dframe.loc[id2, col]
    for col in diffby:
        assert dframe.loc[id1, col] != dframe.loc[id2, col]
"""

"""
# Current
%timeit matcher.get_all_pairs(sameby, diffby)
1500:   862 ms ± 4.33 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
15000:  8.7 s ± 20.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
150000: 1min 30s ± 312 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Duckdb
%timeit find_pairs(dframe,sameby, diffby)
1500:  13.8 ms ± 67.5 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
15000: 27.8 ms ± 468 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
150000: 147 ms ± 937 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
"""
