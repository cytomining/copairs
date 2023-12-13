import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from copairs.compute import compute_ap, compute_ap_contiguous, random_binary_matrix
from copairs.map import average_precision
from copairs.map.multilabel import average_precision as multilabel_average_precision
from tests.helpers import simulate_random_dframe

SEED = 0


def test_random_binary_matrix():
    rng = np.random.default_rng(SEED)
    # Test with n=3, m=4, k=2
    A = random_binary_matrix(3, 4, 2, rng)
    assert A.shape == (3, 4)
    assert np.all(np.sum(A, axis=1) == 2)
    assert np.all((A >= 0) | (A <= 1))

    # Test with n=5, m=6, k=3
    B = random_binary_matrix(5, 6, 3, rng)
    assert B.shape == (5, 6)
    assert np.all(np.sum(B, axis=1) == 3)
    assert np.all((B == 0) | (B <= 1))


def test_compute_ap():
    num_pos, num_neg, num_perm = 5, 6, 100
    total = num_pos + num_neg

    y_true = np.zeros((num_perm, total), dtype=int)
    y_true[:, :num_pos] = 1
    y_pred = np.random.uniform(0, 1, [num_perm, total])
    df = pd.DataFrame({
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
    })
    rel_k = df['y_pred'].apply(lambda x: np.argsort(x)[::-1]).apply(
        lambda x: np.array(df.y_true[0])[x])
    rel_k = np.stack(rel_k)
    ap = compute_ap(rel_k)

    ap_sklearn = df.apply(
        lambda x: average_precision_score(x['y_true'], x['y_pred']), axis=1)

    assert np.allclose(ap_sklearn, ap)


def test_compute_ap_contiguous():
    num_pos_range = [2, 9]
    num_neg_range = [10, 20]
    num_samples_range = [5, 30]
    rng = np.random.default_rng(SEED)
    for _ in range(30):
        num_samples = rng.integers(*num_samples_range)
        counts, rel_k_list = [], []
        ground_truth = []
        null_confs_gt = np.empty((num_samples, 2), dtype=int)
        for j in range(num_samples):
            num_pos = rng.integers(*num_pos_range)
            num_neg = rng.integers(*num_neg_range)
            total = num_pos + num_neg
            y_true = np.zeros(total, dtype=int)
            y_true[:num_pos] = 1
            y_pred = np.random.uniform(0, 1, total)
            ap_score = average_precision_score(y_true, y_pred)
            ground_truth.append(ap_score)

            rel_k = y_true[np.argsort(y_pred)[::-1]]
            rel_k_list.append(rel_k)
            counts.append(total)
            null_confs_gt[j] = [num_pos, total]

        rel_k_list = np.concatenate(rel_k_list)
        counts = np.asarray(counts)
        ap_scores, null_confs = compute_ap_contiguous(rel_k_list, counts)
        assert np.allclose(null_confs_gt, null_confs)
        assert np.allclose(ap_scores, ground_truth)


def test_pipeline():
    length = 10
    vocab_size = {'p': 5, 'w': 3, 'l': 4}
    n_feats = 5
    pos_sameby = ['l']
    pos_diffby = ['p']
    neg_sameby = []
    neg_diffby = ['l']
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby,
                                  rng)
    feats = rng.uniform(size=(length, n_feats))
    average_precision(meta, feats, pos_sameby, pos_diffby, neg_sameby,
                      neg_diffby)


def test_pipeline_multilabel():
    '''Check the multilabel implementation with for mAP calculation'''
    length = 10
    vocab_size = {'p': 3, 'w': 5, 'l': 4}
    n_feats = 8
    multilabel_col = 'l'
    pos_sameby = ['l']
    pos_diffby = []
    neg_sameby = []
    neg_diffby = ['l']
    rng = np.random.default_rng(SEED)
    meta = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby,
                                  rng)
    meta = meta.groupby(['p', 'w'])['l'].unique().reset_index()
    length = len(meta)
    feats = rng.uniform(size=(length, n_feats))

    multilabel_average_precision(meta, feats, pos_sameby, pos_diffby,
                                 neg_sameby, neg_diffby, multilabel_col)
