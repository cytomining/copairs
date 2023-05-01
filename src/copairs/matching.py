'''
Sample pairs with given column restrictions
'''
import itertools
from collections import namedtuple
import logging
from math import comb
from typing import Sequence, Set, Union, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger('copairs')
ColumnList = Union[Sequence[str], pd.Index]


def reverse_index(col: pd.Series) -> pd.Series:
    '''Build a reverse_index for a given column in the DataFrame'''
    return pd.Series(col.groupby(col).indices, name=col.name)


class UnpairedException(Exception):
    '''Exception raised when a row can not be paired with any other row in the
    data'''


class Matcher():
    '''Class to get pair of rows given contraints in the columns'''

    def __init__(self,
                 dframe: pd.DataFrame,
                 columns: ColumnList,
                 seed: int,
                 max_size: Optional[int] = None):
        '''
        max_size: max number of rows to consider from the same value.
        '''
        rng = np.random.default_rng(seed)
        dframe = dframe[columns].reset_index(drop=True).copy()
        dframe.index.name = '__copairs_ix'

        def clip_list(elems: np.ndarray) -> np.ndarray:
            if max_size and elems.size > max_size:
                logger.warning(f'Sampling {max_size} out of {elems.size}')
                elems = rng.choice(elems, max_size)
            return elems

        mappers = [
            reverse_index(dframe[col]).apply(clip_list) for col in dframe
        ]

        # Create a column order based on the number of potential row matches
        # Useful to solve queries with more than one sameby
        n_pairs = {}
        for mapper in mappers:
            n_combs = mapper.apply(lambda x: comb(len(x), 2)).sum()
            n_pairs[mapper.name] = n_combs
        col_order = sorted(n_pairs, key=n_pairs.get)
        self.col_order = {column: i for i, column in enumerate(col_order)}

        self.values = dframe[columns].values
        self.reverse = {
            mapper.name: mapper.apply(set).to_dict()
            for mapper in mappers
        }
        self.rng = rng
        self.frozen_valid = frozenset(range(len(self.values)))
        self.col_to_ix = {c: i for i, c in enumerate(columns)}
        self.columns = columns
        self.n_pairs = n_pairs
        self.rand_iter = iter([])

    def _null_sample(self, diffby: ColumnList):
        '''
        Sample a pair from the frame.
        '''
        valid = set(self.frozen_valid)
        id1 = self.integers(0, len(valid) - 1)
        valid.remove(id1)
        valid = self._filter_diffby(id1, diffby, valid)

        if len(valid) == 0:
            # row1 = self.values[id1]
            # assert np.any(row1 == self.values, axis=1).all()
            raise UnpairedException(f'{id1} has no pairs')
        id2 = self.choice(list(valid))
        return id1, id2

    def sample_null_pair(self, diffby: ColumnList, n_tries=5):
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

    def get_all_pairs(self, sameby: Union[str, ColumnList],
                      diffby: Union[str, ColumnList]):
        '''
        Get all pairs with given params
        '''
        if isinstance(diffby, str):
            diffby = [diffby]
        if isinstance(sameby, str):
            sameby = [sameby]
        if set(sameby) & set(diffby):
            raise ValueError('sameby and diffby must be disjoint lists')
        if not sameby and not diffby:
            raise ValueError('sameby, diffby: at least one should be provided')
        if not sameby:
            return self._only_diffby(diffby)
        if len(sameby) == 1:
            key = next(iter(sameby))
            return self._get_all_pairs_single(key, diffby)

        ComposedKey = namedtuple('ComposedKey', sameby)
        # Multiple sameby. Ordering by minimum number of posible pairs
        sameby = sorted(sameby, key=self.col_order.get)
        candidates = self._get_all_pairs_single(sameby[0], diffby)
        col_ix = [self.col_to_ix[col] for col in sameby[1:]]

        pairs = dict()
        for key, indices in candidates.items():
            for id1, id2 in indices:
                row1 = self.values[id1]
                row2 = self.values[id2]
                if np.all(row1[col_ix] == row2[col_ix]):
                    vals = key, *row1[col_ix]
                    key_tuple = ComposedKey(**dict(zip(sameby, vals)))
                    pair = (id1, id2)
                    pairs.setdefault(key_tuple, list()).append(pair)
        return pairs

    def _get_all_pairs_single(self, sameby: str, diffby: ColumnList):
        '''Get all valid pairs for a single column.'''
        mapper = self.reverse[sameby]
        pairs = dict()
        for key, rows in mapper.items():
            processed = set()
            for id1 in rows:
                valid = set(rows)
                processed.add(id1)
                valid -= processed
                valid = self._filter_diffby(id1, diffby, valid)
                for id2 in valid:
                    pair = (id1, id2)
                    pairs.setdefault(key, list()).append(pair)
        return pairs

    def _only_diffby(self, diffby: ColumnList):
        '''Generate a dict with single NaN key containing all of the pairs
        with different values in the column list'''
        diffby = sorted(diffby, key=self.col_order.get)

        # Cartesian product for one of the diffby columns
        mapper = self.reverse[diffby[0]]
        keys = list(mapper.keys())
        pairs = []
        for key_a, key_b in itertools.combinations(keys, 2):
            pairs.extend(itertools.product(mapper[key_a], mapper[key_b]))
        pairs = np.array(pairs)
        num_pairs = len(pairs)
        col_ix = [self.col_to_ix[col] for col in diffby[1:]]
        vals_a = self.values[pairs[:, 0], col_ix].reshape([num_pairs, -1])
        vals_b = self.values[pairs[:, 1], col_ix].reshape([num_pairs, -1])
        valid = vals_a != vals_b
        valid = np.all(valid, axis=1)

        pairs = np.unique(pairs[valid], axis=0)
        return {None: pairs.tolist()}

    def _filter_diffby(self, idx: int, diffby: ColumnList, valid: Set[int]):
        '''
        Remove from valid rows that have matches with idx in any of the diffby columns
        :idx: index of the row to be compared
        :diffby: indices of columns that should have different values
        :valid: candidate rows to be evaluated
        :returns: subset of valid after removing indices

        '''
        row = self.values[idx]
        for col in diffby:
            val = row[self.col_to_ix[col]]
            if pd.isna(val):
                continue
            mapper = self.reverse[col]
            valid = valid - mapper[val]
        return valid


class MatcherMultilabel():

    def __init__(self, dframe: pd.DataFrame, columns: ColumnList,
                 multilabel_col: str, seed: int):
        dframe = dframe.explode(multilabel_col)
        dframe = dframe.reset_index(names='__original_index')
        self.original_index = dframe['__original_index']
        self.matcher = Matcher(dframe, columns, seed)

    def get_all_pairs(self, sameby: Union[str, ColumnList],
                      diffby: ColumnList):
        pairs = self.matcher.get_all_pairs(sameby, diffby)
        for key, values in pairs.items():
            values = np.asarray(values)
            pairs[key] = list(
                zip(self.original_index[values[:, 0]],
                    self.original_index[values[:, 1]]))

        return pairs

    def sample_null_pair(self, diffby: ColumnList, n_tries=5):
        null_pair = self.matcher.sample_null_pair(diffby, n_tries)
        id1, id2 = self.original_index[list(null_pair)].values
        return id1, id2

    def get_null_pairs(self, diffby: ColumnList, size: int, n_tries=5):
        null_pairs = []
        for _ in tqdm(range(size)):
            null_pairs.append(self.matcher.sample_null_pair(diffby, n_tries))
        null_pairs = np.array(null_pairs)
        null_pairs[:, 0] = self.original_index[null_pairs[:, 0]].values
        null_pairs[:, 1] = self.original_index[null_pairs[:, 1]].values
        return null_pairs
