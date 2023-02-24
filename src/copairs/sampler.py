'''
Sample pairs with given column restrictions
'''
from collections import defaultdict
from math import comb
from typing import Collection, Sequence, Set, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from copairs import logger


class UnpairedException(Exception):
    '''Exception raised when a row can not be paired with any other row in the
    data'''


class Sampler():
    '''Class to get pair of rows given contraints in the columns'''

    def __init__(self, dframe: pd.DataFrame,
                 columns: Union[Sequence[str], pd.Index], seed: int):
        values = dframe[columns].to_numpy(copy=True)
        reverse = [defaultdict(set)
                   for _ in range(len(columns))]  # type: list[dict]
        # Create a reverse index to locate rows containing particular values
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                if pd.isna(val):
                    continue
                mapper = reverse[j]
                mapper[val].add(i)

        # Create a column order based on the number of potential row matches
        # Useful to solve queries with more than one groupby
        n_pairs = []
        for mapper in reverse:
            curr = 0
            for rows in mapper.values():
                curr += comb(len(rows), 2)
            n_pairs.append(curr)
        self.col_order = np.argsort(n_pairs)

        self.values = values
        self.reverse = reverse
        self.rng = np.random.default_rng(seed)
        self.frozen_valid = frozenset(range(len(self.values)))
        self.col_to_ix = {c: i for i, c in enumerate(columns)}
        self.columns = columns
        self.n_pairs = n_pairs
        self.rand_iter = iter([])

    def _null_sample(self, diffby: Collection[str]):
        '''
        Sample a pair from the frame.
        '''
        valid = set(self.frozen_valid)
        id1 = self.integers(0, len(valid) - 1)
        valid.remove(id1)
        diffby_ix = [self.col_to_ix[col] for col in diffby]
        valid = self._filter_diffby(id1, diffby_ix, valid)

        if len(valid) == 0:
            # row1 = self.values[id1]
            # assert np.any(row1 == self.values, axis=1).all()
            raise UnpairedException(f'{id1} has no pairs')
        id2 = self.choice(list(valid))
        return id1, id2

    def sample_null_pair(self, diffby: Collection[str], n_tries=5):
        '''Sample pairs from the data. It tries multiple times before raising an error'''
        for _ in range(n_tries):
            try:
                return self._null_sample(diffby)
            except UnpairedException:
                pass
        raise ValueError(
            'Number of tries exhusted. Could not find a valid pair')

    def rand_next(self):
        try:
            value = next(self.rand_iter)
        except StopIteration:
            rands = self.rng.uniform(size=int(1e6))
            self.rand_iter = iter(rands)
            value = next(self.rand_iter)
        return value

    def integers(self, min_val, max_val):
        return int(self.rand_next() * (max_val - min_val + 1) + min_val)

    def choice(self, items):
        min_val, max_val = 0, len(items) - 1
        pos = self.integers(min_val, max_val)
        return items[pos]

    def get_all_pairs(self, groupby: Union[str, Collection[str]],
                      diffby: Collection[str]):
        '''
        Get all pairs with given params
        '''
        if isinstance(groupby, str):
            groupby = [groupby]
        if set(groupby) & set(diffby):
            raise ValueError('groupby and diffby must be disjoint lists')
        if len(groupby) == 1:
            key = next(iter(groupby))
            return self._get_all_pairs_single(key, diffby)

        # Multiple groupby. Ordering by minimum number of posible pairs
        groupby_ix = [self.col_to_ix[col] for col in groupby]
        groupby_ix = sorted(groupby_ix, key=lambda x: self.col_order[x])
        pairs = defaultdict(list)
        candidates = self._get_all_pairs_single(groupby_ix[0], diffby)
        for key, indices in candidates.items():
            for id1, id2 in indices:
                row1 = self.values[id1]
                row2 = self.values[id2]
                if np.all(row1[groupby_ix[1:]] == row2[groupby_ix[1:]]):
                    pairs[(key, *row1[groupby_ix[1:]])].append((id1, id2))
        return dict(pairs)

    def _get_all_pairs_single(self, groupby: Union[str, int],
                              diffby: Collection[str]):
        '''
        Get all valid pairs for a single column. It considers up to 5000
        samples per each value in the column to avoid memleaks.
        '''
        max_nunique = 5000  # Elements to pair require a limit to avoid memleak.
        if isinstance(groupby, str):
            groupby = self.col_to_ix[groupby]
        diffby_ix = [self.col_to_ix[col] for col in diffby]
        index = self.reverse[groupby]
        pairs = defaultdict(list)
        for key, rows in index.items():
            processed = set()
            if len(rows) >= max_nunique:
                column = self.columns[groupby]
                logger.warning(
                    f'Sampling {max_nunique} values from {key} in column {column}.'
                )
                rows = set(self.rng.choice(np.array(list(rows)), max_nunique))

            for id1 in rows:
                valid = set(rows)
                processed.add(id1)
                valid -= processed
                valid = self._filter_diffby(id1, diffby_ix, valid)
                if valid:
                    pairs[key].extend([(id1, id2) for id2 in valid])
        return dict(pairs)

    def _filter_diffby(self, idx: int, diffby: Collection[int],
                       valid: Set[int]):
        '''
        Remove from valid rows that have matches with idx in any of the diffby columns
        :idx: index of the row to be compared
        :diffby: indices of columns that should have different values
        :valid: candidate rows to be evaluated
        :returns: subset of valid after removing indices

        '''
        row = self.values[idx]
        for col in diffby:
            val = row[col]
            if pd.isna(val):
                continue
            mapper = self.reverse[col]
            valid = valid - mapper[val]
        return valid


class SamplerMultilabel():

    def __init__(self, dframe: pd.DataFrame, columns: Union[Sequence[str],
                                                            pd.Index],
                 multilabel_col: str, seed: int):
        dframe = dframe.explode(multilabel_col)
        dframe = dframe.reset_index(names='__original_index')
        self.original_index = dframe['__original_index']
        self.sampler = Sampler(dframe, columns, seed)

    def get_all_pairs(self, groupby: Union[str, Collection[str]],
                      diffby: Collection[str]):
        pairs = self.sampler.get_all_pairs(groupby, diffby)
        for key, values in pairs.items():
            values = np.asarray(values)
            pairs[key] = list(
                zip(self.original_index[values[:, 0]],
                    self.original_index[values[:, 1]]))

        return pairs

    def sample_null_pair(self, diffby: Collection[str], n_tries=5):
        null_pair = self.sampler.sample_null_pair(diffby, n_tries)
        id1, id2 = self.original_index[list(null_pair)].values
        return id1, id2

    def get_null_pairs(self, diffby: Collection[str], size: int, n_tries=5):
        null_pairs = []
        for _ in tqdm(range(size)):
            null_pairs.append(self.sampler.sample_null_pair(diffby))
        null_pairs = np.array(null_pairs)
        null_pairs[:, 0] = self.original_index[null_pairs[:, 0]].values
        null_pairs[:, 1] = self.original_index[null_pairs[:, 1]].values
        return null_pairs
