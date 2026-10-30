"""Functions to compute average precision."""

import logging
from uuid import uuid4
from typing import List, Callable
from dataclasses import field, dataclass

import numpy as np
import pandas as pd

from copairs import compute
from copairs.matching import UnpairedException, find_pairs

from .filter import flatten_str_list, evaluate_and_filter, validate_pipeline_input
from .normalization import normalize_ap

logger = logging.getLogger("copairs")


# Key of the pairs an AP function attaches to the frame it returns.
PAIRS_ATTR = "copairs_pairs"
ROW_ID = "_copairs_row_id"


class _AttachedPairs(dict):
    """A frame's pairs, carried in ``DataFrame.attrs``.

    pandas deep-copies ``attrs`` on every operation and writes them to parquet
    as JSON. This handle is shared by copies, serializes as ``{}`` and pickles
    to an empty handle, so frames stay cheap to copy and safe to save, and a
    reloaded frame has no pairs. Handles are equal only to themselves, so
    pandas drops them when concatenating frames of different AP calls.
    """

    __slots__ = ("pairs",)

    def __init__(self, pairs=None):
        super().__init__()
        self.pairs = pairs

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_AttachedPairs, ())

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other


def attach_pairs(frame: pd.DataFrame, pairs) -> pd.DataFrame:
    """Attach ``pairs`` to ``frame`` (in place) and return it."""
    frame.attrs[PAIRS_ATTR] = _AttachedPairs(pairs)
    return frame


def attached_pairs(frame: pd.DataFrame):
    """Return the pairs an AP function attached to ``frame``, or None."""
    handle = frame.attrs.get(PAIRS_ATTR)
    return handle.pairs if isinstance(handle, _AttachedPairs) else None


class _FeatureSimilarity:
    """``pairs -> similarities`` of an AP call's features, as its AP stage computed them.

    Batches of ``batch_size`` pairs go through the same pairwise function as
    in :func:`copairs.compute.get_similarity_fn`, one after another in the
    calling thread: the nulls call it from their own worker threads.
    """

    def __init__(self, feats, distance, batch_size):
        self.feats = feats
        self.distance = distance
        self.batch_size = batch_size
        self._fn = compute.pairwise_similarity_fn(distance)

    def __call__(self, pairs):
        pairs = np.asarray(pairs)
        out = np.empty(len(pairs), dtype=np.float32)
        for lo in range(0, len(pairs), self.batch_size):
            batch = pairs[lo : lo + self.batch_size]
            out[lo : lo + len(batch)] = self._fn(
                self.feats[batch[:, 0]], self.feats[batch[:, 1]]
            )
        return out


def check_recomputable(similarity) -> None:
    """Raise if ``similarity`` cannot recompute its AP stage's similarities.

    The permutation null ranks profiles by similarities it recomputes in its
    own batches of pairs; a metric that estimates its scale from each batch
    gives other values there than in the AP stage.
    """
    if (
        isinstance(similarity, _FeatureSimilarity)
        and isinstance(similarity.distance, str)
        and similarity.distance in compute.BATCH_DEPENDENT_METRICS
    ):
        raise ValueError(
            f"distance={similarity.distance!r} estimates its scale from each "
            "batch of pairs, so the permutation null cannot recompute the "
            "similarities the APs were ranked by. Pass null='joint' or "
            "null='copied', which use the APs as computed."
        )


@dataclass(frozen=True, eq=False)
class RankPairs:
    """The pairs behind an AP table, for mAP nulls that re-rank profiles.

    Attached to the frame :func:`average_precision` returns (and returned with
    it by ``return_pairs=True``); indices are rows of that frame. Instances
    compare and hash by identity, since their fields are arrays.

    Attributes
    ----------
    pos, neg : np.ndarray
        ``(n, 2)`` positive and negative pairs.
    similarity : callable
        ``pairs -> similarities`` (NumPy), the function the APs were computed
        with.
    row_ids : np.ndarray or None
        Private row identities assigned by ``average_precision``. Keep the
        ``_copairs_row_id`` column and original index when filtering/copying.
        None is for caller-supplied pairs on external tables, whose source
        alignment the caller must establish.
    source_metadata : pd.DataFrame or None
        Original contributors' metadata, indexed by source row. Shared across
        handles/copies and used to validate whole-group filtering for arbitrary
        grouping columns. Required for permutation; do not mutate this snapshot.

    Notes
    -----
    Copies share the backing pair arrays and features. Do not mutate them
    between AP calculation and testing; this frozen container does not make
    NumPy arrays immutable. Custom similarities must be independent of batch
    size and contents.
    """

    pos: np.ndarray
    neg: np.ndarray
    similarity: Callable[[np.ndarray], np.ndarray]
    row_ids: np.ndarray | None = None
    source_metadata: pd.DataFrame | None = None


@dataclass(frozen=True, eq=False, kw_only=True)
class LabelPairs(RankPairs):
    """Pairs behind a multilabel AP table, whose rows are (profile, label).

    ``pos`` pairs expanded AP rows; ``neg`` pairs source profiles. ``profile``
    and ``row_key`` map each original AP row to its profile and label in
    ``keys``. The joint null uses expanded AP rows; permutation uses source
    profiles and the lazy pairs sharing only the tested label. Preserve the
    AP frame's original index and complete labels: this handle does not track
    source identities or original group membership. Storage is shared under
    the same no-mutation and results-only persistence contract as RankPairs.
    """

    profile: np.ndarray
    row_key: np.ndarray
    label_col: str
    keys: np.ndarray
    _lone: Callable[[], tuple[np.ndarray, np.ndarray]] | None = field(
        default=None, repr=False
    )


def _columns(cols) -> list:
    """Column name or sequence of column names as a list."""
    return [cols] if isinstance(cols, str) else list(cols)


def check_row_identity(frame: pd.DataFrame, pairs: RankPairs | None) -> None:
    """Validate persistent row tokens against the index and selected pair source.

    Tokens survive ordinary DataFrame operations even when attrs are lost.
    They are provenance markers, not a replacement for saved pairs/features.
    External tables without tokens require explicit, caller-aligned pairs.
    """
    if ROW_ID not in frame:
        if pairs is not None and pairs.row_ids is not None:
            raise ValueError(
                f"missing row identities: preserve {ROW_ID} or recompute AP"
            )
        return
    tokens = frame[ROW_ID]
    expected_positions = frame.index.astype(str).to_numpy()
    text = tokens.astype("string")
    if not text.str.fullmatch(r"[0-9a-f]{32}:[0-9]+").fillna(False).all():
        raise ValueError(
            "invalid AP row identities; preserve the original tokens or recompute AP"
        )
    parts = text.str.rsplit(":", n=1)
    if parts.str[0].nunique() > 1:
        raise ValueError(
            "AP row identities mix incompatible sources; recompute AP or test each source separately"
        )
    positions = parts.str[-1]
    if not np.array_equal(positions.to_numpy(), expected_positions):
        raise ValueError(
            "AP row identities do not match the index; keep original rows (no reset_index)"
        )
    if pairs is not None:
        ids = pairs.row_ids
        index = frame.index.to_numpy()
        if (
            ids is None
            or (len(index) and index.max() >= len(ids))
            or not np.array_equal(tokens.to_numpy(), ids[index])
        ):
            raise ValueError(
                "pairs do not match the row identities in the AP table; use the original pairs"
            )


def check_complete_groups(frame, sameby, pairs, tested_index):
    """Validate original group membership independently of positive connectivity.

    Known grouping columns must retain their original values. A new grouping
    column can be used only if every original contributor is still present, so
    its complete membership can be established without guessing missing values.
    """
    source = pairs.source_metadata
    if source is None:
        raise ValueError(
            "permutation needs original source metadata to validate group membership; recompute AP"
        )
    if not tested_index.isin(source.index).all():
        raise ValueError(
            "AP rows do not match the original source metadata; recompute AP"
        )
    extra = [column for column in sameby if column not in source]
    if extra:
        if not source.index.isin(frame.index).all():
            raise ValueError(
                "cannot validate group membership for new grouping columns after filtering; "
                "include those columns in the metadata before recomputing AP"
            )
        source = source.assign(
            **{column: frame.loc[source.index, column] for column in extra}
        )
    for column in sameby:
        original = source.loc[tested_index, column].astype(object)
        current = frame.loc[tested_index, column].astype(object)
        equal = original.eq(current).fillna(False) | (original.isna() & current.isna())
        if not equal.all():
            raise ValueError(
                f"grouping metadata {column!r} changed since AP calculation; recompute AP"
            )
    group_ids = source.groupby(sameby, observed=True).ngroup()
    expected = group_ids.value_counts()
    actual = group_ids.loc[tested_index].value_counts()
    if not actual.eq(expected.reindex(actual.index)).all():
        raise ValueError(
            "incomplete permutation group: original members were dropped or excluded; "
            "drop whole groups only or recompute AP on the kept profiles"
        )


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

    # Sort by index (lexicographical order) and then by similarity (descending)
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
    progress_bar: bool = True,
    return_pairs: bool = False,
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

    distance : str or callable
        The distance function used for computing similarities. Default is "cosine".
        A custom similarity must return the same value for a pair regardless
        of batch size/contents; ensuring this is the caller's responsibility.

    progress_bar : bool
        Whether or not to show tqdm's progress bar.

    return_pairs : bool
        Also return the :class:`RankPairs` the APs were computed from, which
        ``null="permutation"`` in :func:`copairs.map.mean_average_precision`
        needs. They are attached to the returned frame either way, in
        ``frame.attrs["copairs_pairs"]``, and follow it through filtering
        and copies (not saving, pickling or concatenating with other AP
        calls' frames). The permutation null accepts filtering out whole
        groups, not some of a group's members: their APs were computed with
        the dropped members as positives. They keep ``feats`` and every
        positive and negative
        pair (8 bytes per pair) alive with the frame, whichever null is used;
        drop them with ``frame.attrs.pop("copairs_pairs")``.

    Returns
    -------
    pd.DataFrame
        A DataFrame (with ``return_pairs``, a ``(DataFrame, RankPairs)``
        tuple) containing the following columns:
        - 'average_precision': The calculated average precision score for each profile.
        - 'normalized_average_precision': The normalized AP score (scale-independent).
        - 'n_pos_pairs': The number of positive pairs for each profile.
        - 'n_total_pairs': The total number of pairs for each profile.
        - '_copairs_row_id': Private provenance marker; preserve it and the index.
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
    if ROW_ID in meta:
        raise ValueError(
            f"{ROW_ID} is reserved for AP row identities; remove it before recomputing AP"
        )
    # Combine all metadata columns needed for pair definitions
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    # Validate and filter metadata to ensure the required columns are present and usable
    meta, columns = evaluate_and_filter(meta, columns)
    validate_pipeline_input(meta, feats, columns)

    # Get the distance function for similarity calculations (e.g., cosine)
    similarity_fn = compute.get_similarity_fn(distance, progress_bar=progress_bar)

    # Reset metadata index for consistent indexing
    meta = meta.reset_index(drop=True).copy()
    source_columns = list(meta.columns)

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

    # Compute normalized AP scores
    logger.info("Computing normalized average precision...")
    meta["normalized_average_precision"] = np.nan
    # Compute normalized scores for profiles with pairs
    M = null_confs[:, 0]  # n_pos_pairs
    L = null_confs[:, 1]  # n_total_pairs
    N = L - M  # n_neg_pairs
    normalized_scores = normalize_ap(ap_scores, M, N)
    meta.loc[paired_ix, "normalized_average_precision"] = normalized_scores

    logger.info("Finished.")
    # A source nonce plus original position detects resets even for identical
    # scores/metadata. This randomness is provenance only, never a null seed.
    source = uuid4().hex
    row_ids = np.array([f"{source}:{i}" for i in range(len(meta))], dtype=object)
    row_ids.setflags(write=False)
    meta[ROW_ID] = row_ids
    source_metadata = meta.loc[
        (meta["n_pos_pairs"] > 0) & meta["average_precision"].notna(), source_columns
    ].copy(deep=True)
    source_metadata.attrs.clear()
    pairs = RankPairs(
        pos_pairs,
        neg_pairs,
        _FeatureSimilarity(feats, distance, batch_size),
        row_ids,
        source_metadata,
    )
    attach_pairs(meta, pairs)
    return (meta, pairs) if return_pairs else meta


def p_values(
    dframe: pd.DataFrame, null_size: int, seed: int, progress_bar: bool = True
) -> np.ndarray:
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
    progress_bar : bool
        Whether or not to show tqdm's progress bar.

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
    pvals[mask] = compute.p_values(scores, null_confs, null_size, seed, progress_bar)

    # Return the array of p-values, including NaN for invalid profiles
    return pvals
