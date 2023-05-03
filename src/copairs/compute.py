import numpy as np
try:
    import tensorflow
    TF_ENABLED = True
    from copairs import compute_tf
except ImportError:
    TF_ENABLED = False
    from copairs import compute_np

try:
    import tensorflow_probability as tfp
    from copairs import compute_tf
    TFP_ENABLED = True
except ImportError:
    from copairs import compute_np
    TFP_ENABLED = False


def corrcoef_indexed(feats: np.ndarray, pairs: np.ndarray,
                     batch_size) -> np.ndarray:
    '''Compute pairwise correlation'''
    backend = compute_tf if TFP_ENABLED else compute_np
    return backend.pairwise_indexed(feats, pairs, backend.pairwise_corr,
                                    batch_size)


def cosine_indexed(feats: np.ndarray, pairs: np.ndarray,
                   batch_size) -> np.ndarray:
    '''Compute pairwise cosine'''
    backend = compute_tf if TF_ENABLED else compute_np
    return backend.pairwise_indexed(feats, pairs, backend.pairwise_cosine,
                                    batch_size)
