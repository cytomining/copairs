from numpy.random import default_rng

from copairs.replicating import corr_between_replicates, correlation_test, corr_from_pairs
from copairs.sampler import Sampler

from tests.helpers import create_dframe

SEED = 0


def test_corr_between_replicates():
    rng = default_rng(SEED)
    num_samples = 10
    X = rng.normal(size=[num_samples, 6])
    meta = create_dframe(5, num_samples)
    corr_dist, median_num_repl = corr_between_replicates(X,
                                                         meta,
                                                         sameby=['c'],
                                                         diffby=['p', 'w'])


def test_correlation_test():
    rng = default_rng(SEED)
    num_samples = 10
    X = rng.normal(size=[num_samples, 6])
    meta = create_dframe(5, num_samples)
    result = correlation_test(X, meta, sameby=['c'], diffby=['p', 'w'])
    print(result.percent_score_left())


def test_corr_from_pairs():
    num_samples = 10
    sameby = ['c']
    diffby = ['p', 'w']
    rng = default_rng(SEED)
    X = rng.normal(size=[num_samples, 6])
    meta = create_dframe(5, num_samples)
    sampler = Sampler(meta, sameby + diffby, seed=0)
    pairs = sampler.get_all_pairs(sameby, diffby)
    corr_from_pairs(X, pairs, sameby)
