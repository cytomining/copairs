'''Test functions for Matcher'''
from string import ascii_letters

import numpy as np
import pandas as pd
import pytest

from copairs import Matcher, MatcherMultilabel
from tests.helpers import (create_dframe, simulate_plates,
                           simulate_random_dframe, simulate_random_plates)

SEED = 0


def run_stress_sample_null(dframe, num_pairs):
    '''Assert every generated null pair does not match any column'''
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    for _ in range(num_pairs):
        id1, id2 = matcher.sample_null_pair(dframe.columns)
        row1 = dframe.loc[id1]
        row2 = dframe.loc[id2]
        assert (row1 != row2).all()


def test_null_sample_large():
    '''Test Matcher guarantees elements with different values'''
    dframe = create_dframe(32, 10000)
    run_stress_sample_null(dframe, 5000)


def test_null_sample_small():
    '''Test Sample with small set'''
    dframe = create_dframe(3, 10)
    run_stress_sample_null(dframe, 100)


def test_null_sample_nan_vals():
    '''Test NaN values are ignored'''
    dframe = create_dframe(4, 15)
    rng = np.random.default_rng(SEED)
    nan_mask = rng.random(dframe.shape) < 0.5
    dframe[nan_mask] = np.nan
    run_stress_sample_null(dframe, 1000)


def get_naive_pairs(dframe: pd.DataFrame, sameby, diffby):
    '''Compute valid pairs using cross product from pandas'''
    cross = dframe.reset_index().merge(dframe.reset_index(),
                                       how='cross',
                                       suffixes=('_x', '_y'))
    index = True
    for col in sameby:
        index = (cross[f'{col}_x'] == cross[f'{col}_y']) & index
    for col in diffby:
        index = (cross[f'{col}_x'] != cross[f'{col}_y']) & index

    pairs = cross.loc[index, ['index_x', 'index_y']]
    # remove rows that pair themselves
    pairs = pairs[pairs['index_x'] != pairs['index_y']]
    pairs = pairs.sort_values(['index_x', 'index_y']).reset_index(drop=True)
    return pairs


def check_naive(dframe, matcher: Matcher, sameby, diffby):
    '''Check Matcher and naive generate same pairs'''
    gt_pairs = get_naive_pairs(dframe, sameby, diffby)
    vals = matcher.get_all_pairs(sameby, diffby)
    vals = sum(vals.values(), [])
    vals = pd.DataFrame(vals, columns=['index_x', 'index_y'])
    vals = vals.sort_values(['index_x', 'index_y']).reset_index(drop=True)
    vals = set(vals.apply(frozenset, axis=1))
    gt_pairs = set(gt_pairs.apply(frozenset, axis=1))
    assert gt_pairs == vals


def check_simulated_data(length, vocab_size, sameby, diffby, rng):
    '''Test sample of valid pairs from a simulated dataset'''
    dframe = simulate_random_dframe(length, vocab_size, sameby, diffby, rng)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    check_naive(dframe, matcher, sameby, diffby)


def test_stress_simulated_data():
    '''Run multiple tests using simulated data'''
    rng = np.random.default_rng(SEED)
    num_cols_range = [2, 6]
    vocab_size_range = [5, 10]
    length_range = [100, 500]
    for _ in range(50):
        num_cols = rng.integers(*num_cols_range)
        length = rng.integers(*length_range)
        cols = ascii_letters[:num_cols]
        sizes = rng.integers(*vocab_size_range, size=num_cols)
        vocab_size = dict(zip(cols, sizes))
        ndiffby = np.clip(rng.integers(num_cols), 1, num_cols - 2)
        diffby = list(cols[:ndiffby])
        sameby = list(cols[ndiffby:])
        check_simulated_data(length, vocab_size, sameby, diffby, rng)


def test_empty_sameby():
    '''Test query without sameby'''
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    check_naive(dframe, matcher, sameby=[], diffby=['w', 'c'])
    check_naive(dframe, matcher, sameby=[], diffby=['w'])


def test_empty_diffby():
    '''Test query without diffby'''
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    matcher.get_all_pairs(['c'], [])
    check_naive(dframe, matcher, sameby=['c'], diffby=[])
    check_naive(dframe, matcher, sameby=['w', 'c'], diffby=[])


def test_raise_distjoint():
    '''Test check for disjoint sameby and diffby'''
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    with pytest.raises(ValueError, match='must be disjoint lists'):
        matcher.get_all_pairs('c', ['w', 'c'])


def test_raise_no_params():
    '''Test check for at least one of sameby and diffby'''
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    with pytest.raises(ValueError, match='at least one should be provided'):
        matcher.get_all_pairs([], [])


def assert_sameby_diffby(dframe: pd.DataFrame, pairs_dict: dict, sameby,
                         diffby):
    '''Assert the pairs are valid'''
    for _, pairs in pairs_dict.items():
        for id1, id2 in pairs:
            for col in sameby:
                assert dframe.loc[id1, col] == dframe.loc[id2, col]
            for col in diffby:
                assert dframe.loc[id1, col] != dframe.loc[id2, col]


def test_simulate_plates_mult_sameby_large():
    '''Test matcher successfully complete analysis of a large dataset.'''
    dframe = simulate_plates(n_compounds=15000,
                             n_replicates=20,
                             plate_size=384)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    sameby = ['c', 'w']
    diffby = ['p']
    pairs_dict = matcher.get_all_pairs(sameby, diffby)
    assert_sameby_diffby(dframe, pairs_dict, sameby, diffby)


def test_multilabel_column_sameby():
    '''Check the index generated by multilabel implementation is same as Matcher'''
    sameby = ['c']
    diffby = ['p', 'w']
    dframe = simulate_random_plates(n_compounds=4,
                                    n_replicates=5,
                                    plate_size=5,
                                    sameby=sameby,
                                    diffby=diffby)

    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    pairs_dict = matcher.get_all_pairs(sameby, diffby)
    check_naive(dframe, matcher, sameby, diffby)
    assert_sameby_diffby(dframe, pairs_dict, sameby, diffby)

    dframe_multi = dframe.groupby(diffby)['c'].unique().reset_index()
    matcher_multi = MatcherMultilabel(dframe_multi,
                                      dframe_multi.columns,
                                      multilabel_col='c',
                                      seed=SEED)
    pairs_dict_multi = matcher_multi.get_all_pairs(sameby=sameby,
                                                   diffby=diffby)

    for pairs_id, pairs in pairs_dict.items():
        assert pairs_id in pairs_dict_multi
        pairs_multi = pairs_dict_multi[pairs_id].copy()

        values_multi = set()
        for i, j in pairs_multi:
            row_i = dframe_multi.iloc[i][diffby]
            row_j = dframe_multi.iloc[j][diffby]
            value_multi = row_i.tolist() + row_j.tolist()
            values_multi.add(tuple(sorted(value_multi)))

        values = set()
        for i, j in pairs:
            row_i = dframe.iloc[i][diffby]
            row_j = dframe.iloc[j][diffby]
            value = row_i.tolist() + row_j.tolist()
            values.add(tuple(sorted(value)))

        assert values_multi == values


def test_multilabel_column_diffby():
    '''Check the index generated by multilabel implementation is same as Matcher'''
    diffby = ['c']
    sameby = ['p', 'w']
    dframe = simulate_random_plates(n_compounds=4,
                                    n_replicates=5,
                                    plate_size=5,
                                    sameby=sameby,
                                    diffby=diffby)

    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    pairs_dict = matcher.get_all_pairs(sameby, diffby)
    check_naive(dframe, matcher, sameby, diffby)
    assert_sameby_diffby(dframe, pairs_dict, sameby, diffby)

    dframe_multi = dframe.groupby(sameby)['c'].unique().reset_index()
    matcher_multi = MatcherMultilabel(dframe_multi,
                                      dframe_multi.columns,
                                      multilabel_col='c',
                                      seed=SEED)
    pairs_dict_multi = matcher_multi.get_all_pairs(sameby=sameby,
                                                   diffby=diffby)

    for pairs_id, pairs in pairs_dict.items():
        if pairs_id in pairs_dict_multi:
            pairs_multi = pairs_dict_multi[pairs_id].copy()
        elif pairs_id[::-1] in pairs_dict_multi:
            pairs_multi = pairs_dict_multi[pairs_id[::-1]].copy()
        else:
            raise AssertionError('Missing pairs for {pairs_id}')

        values_multi = set()
        for i, j in pairs_multi:
            row_i = dframe_multi.iloc[i][sameby]
            row_j = dframe_multi.iloc[j][sameby]
            value_multi = row_i.tolist() + row_j.tolist()
            values_multi.add(tuple(sorted(value_multi)))

        values = set()
        for i, j in pairs:
            row_i = dframe.iloc[i][sameby]
            row_j = dframe.iloc[j][sameby]
            value = row_i.tolist() + row_j.tolist()
            values.add(tuple(sorted(value)))

        assert values_multi == values
