import numpy as np
from tqdm.auto import tqdm
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from copairs import compute_tf as backend
    TF_ENABLED = True
except ImportError:
    TF_ENABLED = False
    from copairs import compute_np as backend


def corrcoef_indexed(feats: np.ndarray,
                     pairs: np.ndarray,
                     batch_size: int = 20000):
    '''Compute pairwise correlation'''
    return backend.pairwise_indexed(feats, pairs, backend.pairwise_corr,
                                    batch_size)


def cosine_indexed(feats: np.ndarray,
                   pairs: np.ndarray,
                   batch_size: int = 20000):
    '''Compute pairwise cosine'''
    return backend.pairwise_indexed(feats, pairs, backend.pairwise_cosine,
                                    batch_size)
