"""Functions to compute average precision."""

import logging
import threading
from typing import List, Literal, Callable
from dataclasses import field, dataclass

import numpy as np
import pandas as pd

from copairs import compute
from copairs._backend import backend_module, import_backend
from copairs.matching import UnpairedException, find_pairs

from .filter import flatten_str_list, evaluate_and_filter, validate_pipeline_input
from .normalization import normalize_ap

logger = logging.getLogger("copairs")


# Key of the pairs an AP function attaches to the frame it returns.
PAIRS_ATTR = "copairs_pairs"


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


# Held while a null's thread builds a similarity kernel, so that the null's
# other threads wait for it instead of building their own copies.
_BUILD = threading.Lock()

# SciPy metrics whose cdist estimates a scale (variances, covariance) from
# each batch of pairs, so a pair's similarity depends on the batch it is in.
BATCH_DEPENDENT_METRICS = ("seuclidean", "mahalanobis")


class _FeatureSimilarity:
    """``pairs -> similarities`` of an AP call's features, as its AP stage computed them.

    With ``kernel_backend``, the backend (``"numba"`` or ``"cuda"``) whose
    kernel computed the AP stage's similarities, the same kernel: built on
    first use, with its own copy of the features (on the GPU under
    ``"cuda"``), and kept until :meth:`release`, which
    :func:`copairs.map.get_map_pvalue` calls when its null returns. The
    nulls call it from their worker threads: under ``"numba"`` it runs the
    kernel's serial form on the calling thread (the same bits; many small
    parallel launches from several threads cost more than they gain), and
    under ``"cuda"`` its launches run one at a time, as the nulls' parallel
    kernels' do (:func:`copairs.nulls._numba._launch`). Otherwise batches
    of ``batch_size`` pairs go through the same pairwise function as in
    :func:`copairs.compute.get_similarity_fn`, one after another in the
    calling thread.
    """

    def __init__(self, feats, distance, batch_size, kernel_backend=None):
        self.feats = feats
        self.distance = distance
        self.batch_size = batch_size
        self.kernel_backend = kernel_backend
        self._fn = compute.pairwise_similarity_fn(distance)
        self._kernel = None

    def __call__(self, pairs):
        if self.kernel_backend == "numba":
            return self._kernel_sims(pairs)
        if self.kernel_backend == "cuda":
            nulls = import_backend("copairs.nulls._numba", self.kernel_backend)
            return nulls._launch(self._kernel_sims, pairs)
        pairs = np.asarray(pairs)
        out = np.empty(len(pairs), dtype=np.float32)
        for lo in range(0, len(pairs), self.batch_size):
            batch = pairs[lo : lo + self.batch_size]
            out[lo : lo + len(batch)] = self._fn(
                self.feats[batch[:, 0]], self.feats[batch[:, 1]]
            )
        return out

    def _kernel_sims(self, pairs):
        """Return the kernel's similarities of ``pairs``, built by the first call."""
        # Looked up on every call: under "cuda" this checks the GPU (and that
        # this process was not forked from one that used it) before a launch.
        kernels = backend_module(self.kernel_backend)
        kernel = self._kernel  # one read: release() may clear it meanwhile
        if kernel is None:
            with _BUILD:  # one build (one copy of the features) at a time
                kernel = self._kernel
                if kernel is None:
                    kernel = self._kernel = self._build(kernels)
        return kernel(pairs)

    def _build(self, kernels):
        if self.kernel_backend == "numba":
            return kernels.pair_similarity(self.feats, self.distance, parallel=False)
        return kernels.pair_similarity(self.feats, self.distance)

    def release(self) -> None:
        """Drop the kernel and its copy of the features; a later call rebuilds them."""
        self._kernel = None


def similarity_fns(fastap, backend, feats, distance, batch_size, progress_bar):
    """Return an AP stage's similarity function and the nulls' recompute of it.

    The function is ``fastap``'s kernel for ``distance`` where ``fastap``
    (the AP stage's kernels for ``backend``, None for NumPy) has one, else
    :func:`copairs.compute.get_similarity_fn`'s; the recompute is the
    :class:`_FeatureSimilarity` that gives the same similarities.
    """
    generic = compute.get_similarity_fn(distance, progress_bar=progress_bar)
    fn = generic if fastap is None else fastap.similarity_fn(feats, distance, generic)
    kernel_backend = None if fn is generic else backend
    return fn, _FeatureSimilarity(feats, distance, batch_size, kernel_backend)


def release_kernel(similarity) -> None:
    """Drop the kernel a null built in ``similarity`` (a :class:`_FeatureSimilarity`)."""
    if isinstance(similarity, _FeatureSimilarity):
        similarity.release()


def check_recomputable(similarity) -> None:
    """Raise if ``similarity`` cannot recompute its AP stage's similarities.

    The permutation null ranks profiles by similarities it recomputes in its
    own batches of pairs; a metric that estimates its scale from each batch
    gives other values there than in the AP stage.
    """
    if (
        isinstance(similarity, _FeatureSimilarity)
        and isinstance(similarity.distance, str)
        and similarity.distance in BATCH_DEPENDENT_METRICS
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
    pos_diffby : np.ndarray or None
        ``(profiles, columns)`` integer codes of the ``pos_diffby`` columns of
        each row, which the permutation null checks its draws keep; None
        when there are none (or unknown, then unchecked).
    """

    pos: np.ndarray
    neg: np.ndarray
    similarity: Callable[[np.ndarray], np.ndarray]
    pos_diffby: np.ndarray | None = None


@dataclass(frozen=True, eq=False, kw_only=True)
class LabelPairs(RankPairs):
    """The pairs behind a multilabel AP table, whose rows are (profile, label).

    Attached to the frame :func:`copairs.map.multilabel.average_precision`
    returns (and returned with it by ``return_pairs=True``). ``pos`` pairs
    rows of that frame, as in :class:`RankPairs`: the two rows of a positive
    pair hold the label the pair shares, and a pair sharing several labels
    appears once per label. ``neg`` and the rows of ``pos_diffby`` index
    profiles (rows of the metadata); ``profile`` and ``row_key`` map AP rows
    to profiles and labels. The permutation null also needs the pairs that
    share exactly one label and meet the negative rules on the other
    columns (negatives but for that label); they are found when it asks,
    not kept with the frame.

    Attributes
    ----------
    profile : np.ndarray
        Profile of each AP row, by the AP frame's index.
    row_key : np.ndarray
        Label (index into ``keys``) of each AP row, by the AP frame's index.
    label_col : str
        The multilabel column.
    keys : np.ndarray
        The labels with positive pairs.
    """

    profile: np.ndarray
    row_key: np.ndarray
    label_col: str
    keys: np.ndarray
    # () -> (pairs, key): the pairs sharing exactly one label, keys[key],
    # that meet the negative rules on the other columns; None when the
    # label column is not one of the negative pairs' diffby columns.
    _lone: Callable[[], tuple[np.ndarray, np.ndarray]] | None = field(
        default=None, repr=False
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
    backend: Literal["numpy", "numba", "cuda"] = "numpy",
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

    distance : str
        The distance function used for computing similarities. Default is "cosine".

    progress_bar : bool
        Whether to show progress bars for similarity computation.

    backend : {"numpy", "numba", "cuda"}
        The default ``"numpy"`` has no optional dependencies and gives the
        results of copairs <= 0.5.5. ``"numba"`` (install
        ``copairs[numba]``) computes similarities with parallel kernels for
        cosine, abs_cosine, correlation, euclidean, manhattan and chebyshev
        (other distances use the NumPy functions), AP by counting the negatives
        between sorted positives instead of sorting rank lists, and normalized
        AP from a harmonic-number table. Given the same similarities, its APs
        match ``"numpy"`` to float64 rounding. The kernels' similarities match
        to float32 rounding, so near-ties can rank differently, and they are
        computed in float64, so float32 rows whose norms overflow float32 get
        their actual similarity. The kernels take all pairs at once:
        ``batch_size`` and ``progress_bar`` apply only to the other distances.
        ``"cuda"`` (install ``copairs[gpu]``; needs a CUDA GPU, else
        RuntimeError) computes the kernels' similarities and AP on the GPU,
        streaming the negative pairs within 1 GiB of GPU memory, and
        normalized AP on Numba. Its APs equal ``"numba"``'s bit for bit given
        the same similarities; its similarities sum in another order, so they
        match to float32 rounding.

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
        drop them with ``frame.attrs.pop("copairs_pairs")``. The permutation
        null recomputes similarities with the AP stage's kernel when it had
        one, which keeps its own copy of ``feats`` (on the GPU for
        ``"cuda"``) until the null returns.

    Returns
    -------
    pd.DataFrame
        A DataFrame (with ``return_pairs``, a ``(DataFrame, RankPairs)``
        tuple) containing the following columns:
        - 'average_precision': The calculated average precision score for each profile.
        - 'normalized_average_precision': The normalized AP score (scale-independent).
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
    fastap = backend_module(backend)

    # Combine all metadata columns needed for pair definitions
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    # Validate and filter metadata to ensure the required columns are present and usable
    meta, columns = evaluate_and_filter(meta, columns)
    validate_pipeline_input(meta, feats, columns)

    # Get the distance function for similarity calculations (e.g., cosine)
    similarity_fn, similarity = similarity_fns(
        fastap, backend, feats, distance, batch_size, progress_bar
    )

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

    if fastap is not None:
        # Count each positive's preceding negatives instead of sorting
        logger.info("Computing average precision...")
        paired_ix, ap_scores, null_confs = fastap.ap_from_pairs(
            pos_pairs, neg_pairs, pos_sims, neg_sims, n=len(meta)
        )
    else:
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
    normalized_scores = normalize_ap(ap_scores, M, N, backend=backend)
    meta.loc[paired_ix, "normalized_average_precision"] = normalized_scores

    logger.info("Finished.")
    pairs = RankPairs(
        pos_pairs,
        neg_pairs,
        similarity,
        _diffby_codes(meta, pos_diffby),
    )
    attach_pairs(meta, pairs)
    return (meta, pairs) if return_pairs else meta


def _columns(cols) -> list:
    """``cols`` (a column name or a list of them) as a list."""
    return [cols] if isinstance(cols, str) else list(cols)


def _diffby_codes(meta: pd.DataFrame, pos_diffby) -> np.ndarray | None:
    """``(profiles, columns)`` integer codes of ``meta``'s ``pos_diffby`` columns, or None."""
    diffby = _columns(pos_diffby)
    if not diffby:
        return None
    return np.stack([pd.factorize(meta[c])[0] for c in diffby], axis=1)


def p_values(
    dframe: pd.DataFrame,
    null_size: int,
    seed: int,
    progress_bar: bool = True,
    method: str = "fast",
    backend: str = "auto",
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
    method : str
        ``"fast"`` (default) or ``"legacy"`` null sampler, see
        :func:`copairs.compute.p_values`.
    backend : str
        Backend of the fast method: ``"auto"`` (default), ``"cuda"``,
        ``"numba"`` or ``"numpy"``, see :func:`copairs.compute.get_null_dists`.

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
    pvals[mask] = compute.p_values(
        scores, null_confs, null_size, seed, progress_bar, method, backend
    )

    # Return the array of p-values, including NaN for invalid profiles
    return pvals
