"""Helper functions for testing."""

import pandas as pd
import duckdb

Seed = 0
# Cols are c, p and w
sameby = ["c"]
diffby = []

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
        
def find_pairs(dframe, sameby, diffby):
    # Assumes sameby or diffby is not empty
    df = dframe.reset_index()
    with duckdb.connect("main"):
        pos_suffix = [f"AND A.{x} = B.{x}" for x in sameby[1:]]
        neg_suffix = [f"AND NOT A.{x} = B.{x}" for x in diffby]
        string = (
        # f"SELECT list(index),list(index_1),{','.join('first(' + x +')' for x in sameby)} FROM ("
        f"SELECT {','.join(['CAST(A.' + x +') AS ' for x in sameby])},A.index,B.index "
        'FROM df A '
        'JOIN df B '
        f"ON A.{sameby[0]} = B.{sameby[0]}"
        f" {' '.join((*pos_suffix, *neg_suffix))} "
        # f" {' '.join((*pos_suffix, *neg_suffix))})"
        # f"GROUP BY {','.join(sameby)}"
        )
        tmp = duckdb.sql(string)

    dframe = pd.DataFrame({"c": compounds, "p": plates, "w": wells})
    return dframe

# Gen data
%timeit dframe = simulate_plates(n_compounds=15000, n_replicates=20, plate_size=384)

# Load matcher
%timeit matcher = Matcher(dframe, dframe.columns, seed=SEED)

# Evaluate
# pairs_dict = matcher.get_all_pairs(sameby, diffby)
# %timeit pairs_dict2 = find_pairs(dframe,sameby, diffby)


# Compounds = 15000
#%timeit pairs_dict = matcher.get_all_pairs(sameby, diffby)
# 428 ms ± 2.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# %timeit pairs_dict2 = find_pairs(dframe,sameby, diffby)
# 30.1 ms ± 1.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# Compounds = 150000
# %timeit pairs_dict = matcher.get_all_pairs(sameby, diffby)
# 4.59 s ± 10.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# %timeit pairs_dict2 = find_pairs(dframe,sameby, diffby)
# 151 ms ± 612 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
