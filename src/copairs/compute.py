import numpy as np
import pandas as pd
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
                     batch_size: int = 20000) -> np.ndarray:
    '''Compute pairwise correlation'''
    return backend.pairwise_indexed(feats, pairs, backend.pairwise_corr,
                                    batch_size)


def cosine_indexed(feats: np.ndarray,
                   pairs: np.ndarray,
                   batch_size: int = 20000) -> np.ndarray:
    '''Compute pairwise cosine'''
    return backend.pairwise_indexed(feats, pairs, backend.pairwise_cosine,
                                    batch_size)


def compute_similarities(feats: np.ndarray, dict_pairs: dict,
                         sameby: str) -> pd.DataFrame:
    dist_df = []
    for same_id, pairs in dict_pairs.items():
        sub_df = pd.DataFrame(pairs, columns=['ix1', 'ix2'])
        if isinstance(same_id, str):
            sub_df[sameby] = same_id
        else:
            sub_df[sameby] = '_'.join(same_id)
        dist_df.append(sub_df)
    dist_df = pd.concat(dist_df)
    pairs_ix = np.vstack(list(dict_pairs.values()))
    dists = cosine_indexed(feats, pairs_ix)
    dist_df['dist'] = dists
    return dist_df
