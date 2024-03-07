 # copairs

Find pairs and compute metrics between them.

## Installation

```bash
pip install git+https://github.com/cytomining/copairs.git@v0.4.1
```

## Usage

### Data

Say you have a dataset with 20 samples taken in 3 plates `p1, p2, p3`,
each plate is composed of 5 wells `w1, w2, w3, w4, w5`, and each well 
has one or more labels (`t1, t2, t3, t4`) assigned.

```python
import pandas as pd
import random

random.seed(0)
n_samples = 20
dframe = pd.DataFrame({
    'plate': [random.choice(['p1', 'p2', 'p3']) for _ in range(n_samples)],
    'well': [random.choice(['w1', 'w2', 'w3', 'w4', 'w5']) for _ in range(n_samples)],
    'label': [random.choice(['t1', 't2', 't3', 't4']) for _ in range(n_samples)]
})
dframe = dframe.drop_duplicates()
dframe = dframe.sort_values(by=['plate', 'well', 'label'])
dframe = dframe.reset_index(drop=True)
```

|    | plate   | well   | label   |
|---:|:--------|:-------|:--------|
|  0 | p1      | w2     | t4      |
|  1 | p1      | w3     | t2      |
|  2 | p1      | w3     | t4      |
|  3 | p1      | w4     | t1      |
|  4 | p1      | w4     | t3      |
|  5 | p2      | w1     | t1      |
|  6 | p2      | w2     | t1      |
|  7 | p2      | w3     | t1      |
|  8 | p2      | w3     | t2      |
|  9 | p2      | w3     | t3      |
| 10 | p2      | w4     | t2      |
| 11 | p2      | w5     | t1      |
| 12 | p2      | w5     | t3      |
| 13 | p3      | w1     | t3      |
| 14 | p3      | w1     | t4      |
| 15 | p3      | w4     | t2      |
| 16 | p3      | w5     | t2      |
| 17 | p3      | w5     | t4      |

### Getting valid pairs

To get pairs of samples that share the same `label` but comes from different
`plate`s at different `well` positions: 

```python
from copairs import Matcher
matcher = Matcher(dframe, ['plate', 'well', 'label'], seed=0)
pairs_dict = matcher.get_all_pairs(sameby=['label'], diffby=['plate', 'well'])
```

`pairs_dict` is a `label_id: pairs` dictionary containing the list of valid
pairs for every unique value of `labels`

```
{'t4': [(0, 17), (0, 14), (17, 2), (2, 14)],
 't2': [(1, 16), (1, 10), (1, 15), (8, 16), (8, 15), (10, 16)],
 't1': [(3, 11), (3, 5), (3, 6), (3, 7)],
 't3': [(9, 4), (9, 13), (13, 4), (13, 12), (4, 12)]}
```

### Getting valid pairs from a multilabel column

For eficiency reasons, you may not want to have duplicated rows. You can
group all the labels in a single row and use `MatcherMultilabel` to find the
corresponding pairs:

```python
dframe_multi = dframe.groupby(['plate', 'well'])['label'].unique().reset_index()
```

|    | plate   | well   | label              |
|---:|:--------|:-------|:-------------------|
|  0 | p1      | w2     | ['t4']             |
|  1 | p1      | w3     | ['t2', 't4']       |
|  2 | p1      | w4     | ['t1', 't3']       |
|  3 | p2      | w1     | ['t1']             |
|  4 | p2      | w2     | ['t1']             |
|  5 | p2      | w3     | ['t1', 't2', 't3'] |
|  6 | p2      | w4     | ['t2']             |
|  7 | p2      | w5     | ['t1', 't3']       |
|  8 | p3      | w1     | ['t3', 't4']       |
|  9 | p3      | w4     | ['t2']             |
| 10 | p3      | w5     | ['t2', 't4']       |

```python
from copairs import MatcherMultilabel
matcher_multi = MatcherMultilabel(dframe_multi,
                                  columns=['plate', 'well', 'label'],
                                  multilabel_col='label',
                                  seed=0)
pairs_multi = matcher_multi.get_all_pairs(sameby=['label'],
                                          diffby=['plate', 'well'])
```

`pairs_multi` is also a `label_id: pairs` dictionary with the same
structure discussed before:

```
{'t4': [(0, 10), (0, 8), (10, 1), (1, 8)],
 't2': [(1, 10), (1, 6), (1, 9), (5, 10), (5, 9), (6, 10)],
 't1': [(2, 7), (2, 3), (2, 4), (2, 5)],
 't3': [(5, 2), (5, 8), (8, 2), (8, 7), (2, 7)]}
```
