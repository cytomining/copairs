# Migrating to copairs 0.6.0 (unreleased)

This is a breaking release preparation. Historical numerical results and null
baselines are not guaranteed. No legacy-reproduction mode is planned.

**Integration status:** this #157 branch combines the reviewed #156 core with
Alex's component-based multilabel model. #156 itself keeps its single-label
block model; multilabel implementation and integration adjustments belong in
#157. Combined validation is required before release. Package version metadata
is 0.6.0; this does not mean that the release is complete or published.

## Choose the statistical model explicitly when needed

The default mAP null is label permutation. `null="joint"` and `null="copied"`
remain explicit alternatives. Missing data or unsupported structure raises;
there is no warning-based substitution of another null.

Multi-member joint groups require actual positive pairs. Pair counts cannot
establish a clique, even when every member has `group_size - 1` positives.
Singleton joint groups can use their individual AP-null distributions from counts.
Copied sampling uses counts and does not need pair identities, but it is not a
historical reproduction mode. Its existing AP-draw cache remains supported.

Single-label permutation retains the observed positive graph, including non-complete graphs.
Initial positive/negative metadata rules determine the graph and reference pools.
Draws move profiles into those roles without reapplying fixed metadata rules on
either side. Identical reference sets define automatic blocks; each draw keeps
member counts per block. The additional positive-only `pos_diffby` guard is removed.
Structural restrictions remain, including positive edges crossing out of a
single-label tested group, negatives inside it, and overlapping-but-different
reference sets within a group. The selected exchangeability hypothesis remains
an assumption, not a general calibration guarantee.

## Multilabel permutation uses connected components

Multilabel AP frames now attach pair dictionaries, also available with
`return_pairs=True`. Positive pairs index expanded (profile, label) AP rows;
negative pairs index source profiles. The default permutation null moves one
label throughout components of the original negative graph, preserving its
member count per component and its positive-graph structure. Other labels stay
fixed. Negatives are the original negatives plus pairs that shared only the
moved label, minus the sampled receivers; counts can vary by draw.

This explicitly approved model supports overlapping-but-different references
that remain unsupported for single-label APs. Group by the label column and
keep all its holders and original AP indices. Multilabel handles do not track
provenance or original membership. Callers must retain the correct pairs and
complete labels; existing pair/layout checks cannot detect every missing holder,
including some without positive AP rows. The label must be in `neg_diffby` and
not `neg_sameby`. Fixed metadata restrictions are not generally reapplied to
positive roles. Other-label overlaps can change; holding other labels fixed does not
isolate an effect unique to the tested label or establish exchangeability.

The zero-draw, seed and resource contracts below apply to multilabel tests too.
Joint reads their actual positive AP-row graph; copied remains explicitly
available. No `stop_after` parameter is included.

## Pair dictionaries and unchanged backing data

```python
ap, pairs = copairs.map.average_precision(meta, feats, ..., return_pairs=True)
kept = ap.loc[whole_group_mask].copy()  # preserve the original AP index
result = copairs.map.mean_average_precision(
    kept, ["group"], null_size=10_000, threshold=0.05, pairs=pairs
)
```

Pairs are plain dictionaries with `"pos"` and `"neg"` arrays of shape `(n, 2)`
and a `"similarity"` callable mapping pairs to similarities. Use `pairs["pos"]`,
`pairs["neg"]` and `pairs["similarity"]`; the proposed `RankPairs`/`LabelPairs`
dataclasses and attribute access are removed. Explicit `pairs=` overrides attached
information. Raw `pos_pairs=` is joint-only and must agree with the selected
dictionary when one is available.

Multilabel dictionaries also contain `"profile"` and `"row_key"` arrays mapping
AP rows to source profiles and labels in `"keys"`, the `"label_col"` column name,
and a lazy `"_lone"` function returning pairs sharing only one label. Its value
is None when negative rules do not support permutation. The `"label_col"` entry
selects multilabel permutation; no class hierarchy is used.

Keep the original AP index, correct pair dictionary and complete tested groups.
There are no private row tokens, source snapshots or original-membership checks.
Existing index/count/pair/layout validation cannot detect every misalignment or
omission, including equal-degree index resets, mixed AP sources and dropped
disconnected components. These are caller responsibilities; recompute AP when
removing members of a tested group.

Copies share the pair dictionary, arrays and feature storage. Do not mutate them
between AP calculation and testing. A small pandas-attachment wrapper keeps this
runtime state shared across copies and out of ordinary CSV/Parquet/pickle saves.
Saved tables contain results only; later permutation tests require the original
pair dictionary or AP recomputation. Concatenating incompatible AP sources must
not be used to manufacture a new graph.

Missing AP/no-positive rows and missing group keys are excluded with logged
counts. No usable groups, invalid scores/counts and unsupported layouts raise.
Ensure exclusions do not leave incomplete tested groups; structural checks cannot
detect every omission.
Integer-valued counts are normalized to int64 without rounding integer inputs
through float64. NumPy integers, nullable pandas integers, Arrow integers and
integral floating columns are supported. Fractional, nonfinite and out-of-range
counts in tested rows raise useful validation errors.

## Scores-only calls and downstream corrections

For all three null models, `null_size=0` retains scores after applicable input and
layout validation, skips sampling/correction and returns:

| Columns | Scores-only values |
| --- | --- |
| `mean_average_precision`, `mean_normalized_average_precision` | Computed scores |
| `p_value`, `corrected_p_value` | `NaN` |
| `below_p`, `below_corrected_p` | `pd.NA`, nullable `boolean` dtype |
| Hierarchical `stage1_p_value`, `stage1_corrected_p_value` | `NaN` |
| Hierarchical `stage1_significant` | `pd.NA`, nullable `boolean` dtype |

Grouping columns must be nonempty, distinct string names present in the input;
`hierarchical_by` must be a nonempty proper subset of `sameby`. `max_workers`
must be None or a positive integer (not a boolean), even for zero draws or joint
sampling. These checks run before sampling/executor construction.

Unknown significance is not `False`. Correction helpers exclude missing tests
and preserve their missing results without contaminating tested rows. This change
does not define zero-positive assignments under a future metadata-reapplied null.

Positive draws still use inclusive ties with dtype-aware tolerance, float64 means
reported in AP dtype, observed-mAP consistency checks and `(hits + 1)/(draws + 1)`.
Sampling is Monte Carlo; no automatic exact enumeration or adaptive stopping runs.

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

- The reference cap is a non-negative integer counting all reference profile rows
  across one tested group's blocks. It is checked before quadratic pool-table
  allocation. For multilabel APs, references are the union of the group's
  components minus its members; sum across all components. Multilabel tables
  include members, so the byte estimate uses the full pool size. Splitting
  unchanged references into more blocks does not help.
- The pool budget is a positive byte count for scheduling estimated concurrent
  pool key/rank tables. One oversized pool may run alone.
- The key-table budget is non-negative; zero disables optional cross-pool reuse.
- These limits do not cap total process RAM. Features, layouts, temporaries and
  other allocations are not all counted. Per-call settings do not mutate globals.
- `max_workers` controls permutation group/key work and copied p-value evaluation;
  it does not control joint sampling, DuckDB, the original AP thread pool or BLAS.

Identical inputs, identities and seed give the same results across supported
worker/chunk settings in the same supported environment. Equivalent joint
signatures share draws and correlated Monte Carlo errors. Arbitrary future
software-version identity is not promised.

Custom similarities must be independent of batch size and contents. The observed
mAP check cannot prove every callable meets that contract. Batch-estimated
`seuclidean` and `mahalanobis` remain unsupported for permutation testing.

## Remaining integration and separate work

- Ship #156 and #157 together in 0.6.0 after combined validation, keeping their
  implementation work in separate PRs.
- Adaptive stopping belongs in a separate PR; its original #157 implementation
  has been removed. No stopping PR has been created by this work.
- #155 was reviewed without identifying a required standalone correctness fix.
  Its streamed sampler/backend work has not been imported; NumPy remains first
  and no legacy-reproduction mode is planned.
- Run combined integration checks before treating the release as ready.

Calibration (#162), fixed variance/covariance distances (#164), persistent
joint/permutation caching (#165), metadata-reapplied zero-positive assignments
(#161), and a persistence framework remain separate work.
