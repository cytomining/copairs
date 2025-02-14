"""Helper functions for testing."""

import pandas as pd
import duckdb

Seed = 0
# Cols are c, p and w
sameby = ["c"]
diffby = []
n_compounds = 15000
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
def find_pairs(dframe, sameby, diffby):
    # Assumes sameby or diffby is not empty
    df = dframe.reset_index()
    with duckdb.connect("main"):
        pos_suffix = [f"AND A.{x} = B.{x}" for x in sameby[1:]]
        neg_suffix = [f"AND NOT A.{x} = B.{x}" for x in diffby]
        string = (
        f"SELECT {','.join(['A.' + x for x in sameby])},A.index,B.index "
        'FROM df A '
        'JOIN df B '
        f"ON A.{sameby[0]} = B.{sameby[0]}"
        f" {' '.join((*pos_suffix, *neg_suffix))} "
        )
        # tmp = duckdb.sql(f"SELECT * WHERE(c = c0) FROM ({string})")
        tmp = duckdb.sql(string)
        return tmp

    return None

# Gen data
dframe = simulate_plates(n_compounds, n_replicates, plate_size)

# Load matcher
matcher = Matcher(dframe, dframe.columns, seed=SEED)

# Evaluate
# %timeit pairs_dict = matcher.get_all_pairs(sameby, diffby)
duckdb_results = find_pairs(dframe,sameby, diffby)
"""Assert the pairs are valid."""
for _, id1, id2 in duckdb_results.fetchall():
    for col in sameby:
        assert dframe.loc[id1, col] == dframe.loc[id2, col]
    for col in diffby:
        assert dframe.loc[id1, col] != dframe.loc[id2, col]


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
