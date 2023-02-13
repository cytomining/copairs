import numpy as np
from tqdm.auto import tqdm
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    TF_ENABLED = True
except ImportError:
    TF_ENABLED = False


def corrcoef_rowwise(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlation between two matrices in a paired row-wise
    fashion. `x_sample` and `y_sample` must be of the same shape.
    '''
    x_mean = x_sample.mean(axis=1, keepdims=True)
    y_mean = y_sample.mean(axis=1, keepdims=True)

    x_center = x_sample - x_mean
    y_center = y_sample - y_mean

    numer = (x_center * y_center).sum(axis=1)

    denom = (x_center**2).sum(axis=1) * (y_center**2).sum(axis=1)
    denom = np.sqrt(denom)

    corrs = numer / denom
    return corrs


def corrcoef_indexed_tf(feats: np.ndarray,
                        pairs: np.ndarray,
                        batch_size: int = 20000):
    '''Compute pairwise correlation using tensorflow'''
    featstf = tf.constant(feats)

    def get_pair(ids):
        feat_x = tf.gather(featstf, ids[:, 0])
        feat_y = tf.gather(featstf, ids[:, 1])
        return feat_x, feat_y

    dataset = tf.data.Dataset.from_tensor_slices(pairs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(get_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    corrs = []
    for x_sample, y_sample in tqdm(dataset, leave=False):
        corr = tfp.stats.correlation(x_sample,
                                     y_sample,
                                     sample_axis=1,
                                     event_axis=None)
        corrs.append(corr.numpy())

    corrs = np.concatenate(corrs)
    assert len(corrs) == len(pairs)
    return corrs


def corrcoef_indexed_np(feats: np.ndarray,
                        pair_ix: np.ndarray,
                        batch_size: int = 50000):
    '''Get pairwise correlation using a list of paired indices'''
    num_pairs = len(pair_ix)

    corrs = []
    for i in tqdm(range(0, num_pairs, batch_size), leave=False):
        x_sample = feats[pair_ix[i:i + batch_size, 0]]
        y_sample = feats[pair_ix[i:i + batch_size, 1]]
        corr = corrcoef_rowwise(x_sample, y_sample)
        corrs.append(corr)

    corrs = np.concatenate(corrs)
    assert len(corrs) == num_pairs
    return corrs


if TF_ENABLED:
    corrcoef_indexed = corrcoef_indexed_tf
else:
    corrcoef_indexed = corrcoef_indexed_np
