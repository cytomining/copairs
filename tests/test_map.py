import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from copairs.map import random_binary_matrix, compute_ap

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
