# Migrating to copairs 0.6.0 (unreleased)

0.6.0 is a breaking release: label permutation becomes the default mAP null,
including for multilabel APs. Numerical results change; reproducing historical
results requires the earlier software environment, not a legacy mode in 0.6.0.

## Choose the statistical model

`null="joint"` and `null="copied"` remain explicit alternatives. Missing information
or unsupported structure raises rather than selecting another null.

Multi-member joint groups require actual positive pairs: counts alone do not
identify their graph. Singleton joint groups can use counts alone. Copied sampling
uses counts, with its existing AP-draw cache.

Single-label permutation retains the observed positive graph, including
non-complete graphs. Identical reference sets define blocks; each draw preserves
the member count per block. Positive/negative metadata rules construct the graph
and pools but are not reapplied during draws. The positive-only `pos_diffby` guard
is removed. Positive edges outside the group, negatives inside it and
overlapping-but-different reference sets within a group remain unsupported.

## Multilabel permutation

Multilabel AP frames now attach pair dictionaries, also available with
`return_pairs=True`. Positive pairs index expanded (profile, label) AP rows;
negative pairs index source profiles. Joint sampling uses those positive AP-row
edges directly.

Permutation moves one label within connected components of the original negative
graph, preserving its member count per component and its positive-graph roles.
Other labels stay fixed. Negatives are the original negatives plus pairs that
shared only the moved label, minus the sampled receivers; counts can vary by draw.
This supports overlapping-but-different references that the single-label model
rejects. Group by the label column, which must appear in `neg_diffby` and not
`neg_sameby`.

Both permutation models assume exchangeability within their pools. For
multilabel tests, component connectivity alone is insufficient; account for
plate/batch structure with the pair rules. Overlap with other labels can change,
so the test concerns label placement rather than an effect unique to that label.

## Pair dictionaries and input lifecycle

```python
ap, pairs = copairs.map.average_precision(meta, feats, ..., return_pairs=True)
kept = ap.loc[whole_group_mask].copy()  # preserve the original AP index
result = copairs.map.mean_average_precision(
    kept, ["group"], null_size=10_000, threshold=0.05, pairs=pairs
)
```

Pairs are plain dictionaries with `"pos"` and `"neg"` arrays of shape `(n, 2)`
and a `"similarity"` callable. Use dictionary access such as `pairs["pos"]`, not
the proposed `RankPairs`/`LabelPairs` classes. Explicit `pairs=` overrides attached
information. Raw `pos_pairs=` is joint-only and must match the selected dictionary
when one is available.

Multilabel dictionaries also contain `"profile"` and `"row_key"` arrays mapping
AP rows to source profiles and labels in `"keys"`, the `"label_col"` column name,
and a lazy `"_lone"` function returning pairs sharing only one label. `"_lone"`
is `None` when negative rules do not support permutation; `"label_col"` selects
the multilabel path.

Keep original AP indices, matching pairs and complete tested groups/labels.
Validation checks indices, counts and layout, not source identity or original
membership. It can miss index resets, mixed AP sources or omitted members,
including disconnected components and label holders without positive AP rows.
Filter whole groups; recompute AP when removing members of a tested group.

Copies share the dictionary, arrays and features; keep them unchanged between
AP calculation and testing. Saved tables contain results only. Later permutation
tests require the original pair dictionary or AP recomputation. Concatenating
frames from different AP calls drops their attached pairs.

Rows with missing AP, no positives or missing group keys are excluded with logged
counts; retain complete tested groups through filtering. No usable groups,
invalid scores/counts and unsupported layouts raise. Counts support NumPy,
nullable pandas and Arrow integers, plus integral floating columns. They are
normalized to int64 without rounding integer inputs through float64; fractional,
nonfinite and out-of-range counts in tested rows raise.

## Scores-only calls and correction

For all three nulls, `null_size=0` validates inputs and layout, skips sampling and
correction, and returns:

| Columns | Scores-only values |
| --- | --- |
| `mean_average_precision`, `mean_normalized_average_precision` | Computed scores |
| `p_value`, `corrected_p_value` | `NaN` |
| `below_p`, `below_corrected_p` | `pd.NA`, nullable `boolean` dtype |
| Hierarchical `stage1_p_value`, `stage1_corrected_p_value` | `NaN` |
| Hierarchical `stage1_significant` | `pd.NA`, nullable `boolean` dtype |

Significance is unknown, rather than `False`. Correction helpers exclude missing
tests and preserve their missing results.

Grouping columns must be nonempty, distinct string names present in the input;
`hierarchical_by` must be a nonempty proper subset of `sameby`. `max_workers`
must be `None` or a positive integer, excluding booleans. These checks apply to
zero draws and joint sampling too.

Positive draw counts use inclusive, dtype-aware ties, float64 means reported in
AP dtype, observed-mAP consistency checks and `(hits + 1)/(draws + 1)`. All requested
Monte Carlo draws run; there is no automatic enumeration or `stop_after` parameter.

## Seeds, resources and execution

`get_map_pvalue`, `mean_average_precision` and
`mean_average_precision_hierarchical` default to `seed=0`. Explicit `None` resolves
fresh entropy once per run. Joint/permutation use unsigned-64-bit seeds; copied
accepts any non-negative integer and leaves fresh runs uncached. Supply
`hierarchical_by=[...]` explicitly, preferably by keyword.

All three functions accept keyword-only settings:

```python
max_reference_profiles=16_384
pool_memory_budget=4 * 1024**3
key_table_memory_budget=256 * 1024**2
```

- The non-negative reference cap counts all reference profile rows across one
  group's blocks, before quadratic pool-table allocation. For multilabel APs,
  it counts the component union minus the label's members; memory estimates
  include the full pool.
- The positive pool byte budget schedules estimated concurrent key/rank tables.
  One oversized pool may run alone.
- The non-negative key-table budget limits cross-pool reuse; zero disables it.
- Budgets cover tables, not total process RAM. Features, layouts and temporary
  arrays need additional memory. Per-call settings leave global limits unchanged.
- `max_workers` controls permutation group/key work and copied p-value evaluation,
  not joint sampling, DuckDB, the original AP thread pool or BLAS.

With fixed inputs, row identities, seed and software environment, worker/chunk
settings leave results unchanged. Equivalent joint signatures share draws and
correlated Monte Carlo errors.

Custom similarities must return the same value for a pair regardless of batch
size or other pairs in the batch. Batch-estimated `seuclidean` and `mahalanobis`
are unsupported for permutation testing.

## Release scope

0.6.0 combines #156 and #157 after main-branch validation. Adaptive stopping,
#155 acceleration and richer persistence are separate work. Other follow-ups:
calibration (#162), fixed variance/covariance distances (#164), joint/permutation
caching (#165), and metadata-reapplied zero-positive assignments (#161).
