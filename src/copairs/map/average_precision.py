"""Functions to compute average precision."""

import itertools
import logging
from typing import List

import numpy as np
import pandas as pd

from copairs import compute
from copairs.matching import UnpairedException, find_pairs

from .filter import evaluate_and_filter, flatten_str_list, validate_pipeline_input

logger = logging.getLogger("copairs")


def build_rank_lists(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
):
    """Build rank lists for calculating average precision.

    This function processes positive and negative pairs along with their similarity scores
    to construct rank lists and determine unique profile indices with their associated counts.

    Parameters
    ----------
    pos_pairs : np.ndarray
        Array of positive pair indices, where each pair is represented as a pair of integers.

    neg_pairs : np.ndarray
        Array of negative pair indices, where each pair is represented as a pair of integers.

    pos_sims : np.ndarray
        Array of similarity scores for positive pairs.

    neg_sims : np.ndarray
        Array of similarity scores for negative pairs.

    Returns
    -------
    paired_ix : np.ndarray
        Unique indices of profiles that appear in the rank lists.

    rel_k_list : np.ndarray
        Array of relevance labels (1 for positive pairs, 0 for negative pairs) sorted by
        decreasing similarity within each profile.

    counts : np.ndarray
        Array of counts indicating how many times each profile index appears in the rank lists.
    """
    # Combine relevance labels: 1 for positive pairs, 0 for negative pairs
    labels = np.concatenate(
        [
            np.ones(pos_pairs.size, dtype=np.uint32),
            np.zeros(neg_pairs.size, dtype=np.uint32),
        ]
    )

    # Flatten positive and negative pair indices for ranking
    ix = np.concatenate([pos_pairs.ravel(), neg_pairs.ravel()])

    # Expand similarity scores to match the flattened pair indices
    sim_all = np.concatenate([np.repeat(pos_sims, 2), np.repeat(neg_sims, 2)])

    # Sort by similarity (descending) and then by index (lexicographical order)
    # `1 - sim_all` ensures higher similarity values appear first, prioritizing
    # pairs with stronger similarity scores for ranking.
    # `ix` acts as a secondary criterion, ensuring consistent ordering of pairs
    # with equal similarity scores by their indices (lexicographical order).
    ix_sort = np.lexsort([1 - sim_all, ix])

    # Create the rank list of relevance labels sorted by similarity and index
    rel_k_list = labels[ix_sort]

    # Find unique profile indices and count their occurrences in the pairs
    paired_ix, counts = np.unique(ix, return_counts=True)

    return paired_ix, rel_k_list, counts.astype(np.uint32)


def average_precision(
    meta: pd.DataFrame,
    feats: pd.DataFrame,
    pos_sameby: List[str],
    pos_diffby: List[str],
    neg_sameby: List[str],
    neg_diffby: List[str],
    batch_size: int = 20000,
    distance: str = "cosine",
) -> pd.DataFrame:
    """Calculate average precision (AP) scores for pairs of profiles based on their similarity.

    This function identifies positive and negative pairs of profiles using  metadata
    rules, computes their similarity scores, and calculates average precision
    scores for each profile. The results include the number of positive and total pairs
    for each profile.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata of the profiles, including columns used for defining pairs.
        This DataFrame should include the columns specified in `pos_sameby`,
        `pos_diffby`, `neg_sameby`, and `neg_diffby`.

    feats : np.ndarray
        Feature matrix representing the profiles, where rows correspond to profiles
        and columns to features.

    pos_sameby : list
        Metadata columns used to define positive pairs. Two profiles are considered a
        positive pair if they belong to the same group that is not a control group.
        For example, replicate profiles of the same compound are positive pairs and
        should share the same value in a column identifying compounds.

    pos_diffby : list
        Metadata columns used to differentiate positive pairs. Positive pairs do not need
        to differ in any metadata columns, so this is typically left empty. However,
        if necessary (e.g., to account for batch effects), you can specify columns
        such as batch identifiers.

    neg_sameby : list
        Metadata columns used to define negative pairs. Typically left empty, as profiles
        forming a negative pair (e.g., a compound and a DMSO/control) do not need to
        share any metadata values. This ensures comparisons are made without enforcing
        unnecessary constraints.

    neg_diffby : list
        Metadata columns used to differentiate negative pairs. Two profiles are considered
        a negative pair if one belongs to a compound group and the other to a DMSO/
        control group. They must differ in specified metadata columns, such as those
        identifying the compound and the treatment index, to ensure comparisons are
        only made between compounds and DMSO controls (not between different compounds).

    batch_size : int
        The batch size for similarity computations to optimize memory usage.
        Default is 20000.

    distance : str
        The distance function used for computing similarities. Default is "cosine".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'average_precision': The calculated average precision score for each profile.
        - 'n_pos_pairs': The number of positive pairs for each profile.
        - 'n_total_pairs': The total number of pairs for each profile.
        - Additional metadata columns from the input.

    Raises
    ------
    UnpairedException
        If no positive or negative pairs are found in the dataset.

    Notes
    -----
    - Positive Pair Rules:
        * Positive pairs are defined by `pos_sameby` (profiles share these metadata values)
          and optionally differentiated by `pos_diffby` (profiles must differ in these metadata values if specified).
    - Negative Pair Rules:
        * Negative pairs are defined by `neg_diffby` (profiles differ in these metadata values)
          and optionally constrained by `neg_sameby` (profiles share these metadata values if specified).
    """
    # Combine all metadata columns needed for pair definitions
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    # Validate and filter metadata to ensure the required columns are present and usable
    meta, columns = evaluate_and_filter(meta, columns)
    validate_pipeline_input(meta, feats, columns)

    # Get the distance function for similarity calculations (e.g., cosine)
    similarity_fn = compute.get_similarity_fn(distance)

    # Reset metadata index for consistent indexing
    meta = meta.reset_index(drop=True).copy()

    logger.info("Indexing metadata...")

    # Identify positive pairs based on `pos_sameby` and `pos_diffby`
    logger.info("Finding positive pairs...")
    pos_pairs = find_pairs(meta, sameby=pos_sameby, diffby=pos_diffby)
    if len(pos_pairs) == 0:
        raise UnpairedException("Unable to find positive pairs.")

    # Identify negative pairs based on `neg_sameby` and `neg_diffby`
    logger.info("Finding negative pairs...")
    neg_pairs = find_pairs(meta, sameby=neg_sameby, diffby=neg_diffby)
    if len(neg_pairs) == 0:
        raise UnpairedException("Unable to find negative pairs.")

    # Compute similarities for positive pairs
    logger.info("Computing positive similarities...")
    pos_sims = similarity_fn(feats, pos_pairs, batch_size)

    # Compute similarities for negative pairs
    logger.info("Computing negative similarities...")
    neg_sims = similarity_fn(feats, neg_pairs, batch_size)

    # Build rank lists for calculating average precision
    logger.info("Building rank lists...")
    paired_ix, rel_k_list, counts = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims
    )

    # Compute average precision scores and associated configurations
    logger.info("Computing average precision...")
    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)

    # Add AP scores and pair counts to the metadata DataFrame
    logger.info("Creating result DataFrame...")
    meta["n_pos_pairs"] = 0
    meta["n_total_pairs"] = 0
    meta.loc[paired_ix, "average_precision"] = ap_scores
    meta.loc[paired_ix, "n_pos_pairs"] = null_confs[:, 0]
    meta.loc[paired_ix, "n_total_pairs"] = null_confs[:, 1]

    logger.info("Finished.")
    return meta


def p_values(dframe: pd.DataFrame, null_size: int, seed: int) -> np.ndarray:
    """Compute p-values for average precision scores based on a null distribution.

    This function calculates the p-values for each profile in the input DataFrame,
    comparing their average precision scores (`average_precision`) against a null
    distribution generated for their specific configurations (number of positive
    and total pairs). Profiles with no positive pairs are excluded from the p-value calculation.

    Parameters
    ----------
    dframe : pd.DataFrame
        A DataFrame containing the following columns:
        - `average_precision`: The AP scores for each profile.
        - `n_pos_pairs`: Number of positive pairs for each profile.
        - `n_total_pairs`: Total number of pairs (positive + negative) for each profile.
    null_size : int
        The number of samples to generate in the null distribution for significance testing.
    seed : int
        Random seed for reproducibility of the null distribution.

    Returns
    -------
    np.ndarray
        An array of p-values for each profile in the DataFrame. Profiles with no positive
        pairs will have NaN as their p-value.
    """
    # Create a mask to filter profiles with at least one positive pair
    mask = dframe["n_pos_pairs"] > 0

    # Initialize the p-values array with NaN for all profiles
    pvals = np.full(len(dframe), np.nan, dtype=np.float32)

    # Extract the average precision scores and null configurations for valid profiles
    scores = dframe.loc[mask, "average_precision"].values
    null_confs = dframe.loc[mask, ["n_pos_pairs", "n_total_pairs"]].values

    # Compute p-values for profiles with valid configurations using the null distribution
    pvals[mask] = compute.p_values(scores, null_confs, null_size, seed)

    # Return the array of p-values, including NaN for invalid profiles
    return pvals
