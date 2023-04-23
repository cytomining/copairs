from typing import Callable

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def pairwise_indexed(feats: np.ndarray, pairs: np.ndarray,
                     batch_pairwise_op: Callable[[tf.Tensor, tf.Tensor],
                                                 tf.Tensor], batch_size):
    '''Compute pairwise operation'''
    featstf = tf.constant(feats)

    def get_pair(ids):
        feat_x = tf.gather(featstf, ids[:, 0])
        feat_y = tf.gather(featstf, ids[:, 1])
        return feat_x, feat_y

    dataset = tf.data.Dataset.from_tensor_slices(pairs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(get_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    result = []
    for x_sample, y_sample in tqdm(dataset, leave=False):
        output = batch_pairwise_op(x_sample, y_sample)
        result.append(output.numpy())

    result = np.concatenate(result)
    assert len(result) == len(pairs)
    return result


def pairwise_corr(x_sample: tf.Tensor, y_sample: tf.Tensor) -> tf.Tensor:
    import tensorflow_probability as tfp
    return tfp.stats.correlation(x_sample,
                                 y_sample,
                                 sample_axis=1,
                                 event_axis=None)


def pairwise_cosine(x_sample: tf.Tensor, y_sample: tf.Tensor) -> tf.Tensor:
    x_sample = tf.linalg.l2_normalize(x_sample, axis=1)
    y_sample = tf.linalg.l2_normalize(y_sample, axis=1)
    c_dist = tf.reduce_sum(x_sample * y_sample, axis=1)
    return c_dist
