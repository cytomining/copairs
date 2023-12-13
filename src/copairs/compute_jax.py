import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
from functools import partial
import numpy as np

import jax
from jax import numpy as jnp


def _cosine_indexed(pair, feats):
    print('compiling')
    i, j = pair
    x, y = feats[i], feats[j]
    return x @ y / (jnp.linalg.norm(x) * jnp.linalg.norm(y))


batch_cosine_indexed = jax.vmap(_cosine_indexed, in_axes=[0, None])
par_cosine_indexed = jax.pmap(batch_cosine_indexed, in_axes=(0, None))


def _mask(pairs, size):
    pairs = pairs
    diff = size - len(pairs)
    if not diff:
        return pairs
    pad = np.zeros_like(pairs, shape=[diff, 2])
    pairs = np.concatenate([pairs, pad])
    return pairs



def cosine_indexed(feats: np.ndarray, pairs: np.ndarray, num_cores) -> np.ndarray:
    batch_size = pairs.shape[0] // num_cores + int(pairs.shape[0] % num_cores > 0)
    total = len(pairs)
    pairs = [
        pairs[i:i + batch_size] for i in range(0, pairs.shape[0], batch_size)
    ]
    pairs[-1] = _mask(pairs[-1], batch_size)
    pairs = np.stack(pairs)
    dist = par_cosine_indexed(pairs, feats)
    dist = dist.ravel()[:total]
    return dist

if __name__ == '__main__':
    feats = np.random.uniform(size=[10, 300])
    pairs = np.random.randint(0, len(feats), size=[250000, 2])
    cosine_indexed(feats, pairs, num_cores=8)
