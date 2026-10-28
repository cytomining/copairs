# PR #156: synthetic negative-rule null-layout study

**Analysis only, not a production change or a new null recommendation.** This is
follow-up evidence for retaining the current pool-based behavior for now while
quantifying differences worth considering later. See [results/REPORT.md](results/REPORT.md)
for measured findings and [results/results.json](results/results.json) for per-dataset
summaries. The tested PR #156 source is pinned to
`eabb22bf87c65ec12219297d7d85fc8d2b58bf9c`; this branch starts from main `310e1ce`
and does not include PR #156's production diff.

## Precisely what is compared

All scenarios have one queried group with **complete positive pairs**; no
`pos_diffby` restriction is introduced or relaxed. Observed members have group
`query`, role `is_control=False`; all other profiles have distinct control group
labels and `is_control=True`. Positive rules are `pos_sameby=[group]`,
`pos_diffby=[]`. Negative rules always include `neg_diffby=[is_control]`, plus
`neg_sameby=[plate]` for `same` scenarios or `neg_diffby=[plate]` for `different`
scenarios. For clarity the study constructs these observed pairs explicitly;
separate tests rebuild real metadata and check the public AP pipeline.

Production `group_layouts` constructs the original blocks. A and both B models use
**the same assignments**: independently choose the observed number of members from
each original block pool. Block counts are fixed, not necessarily physical plate
counts. Selected profiles become query/non-control; every other profile in the
original group-specific union becomes a control. Plate and unit IDs stay attached
to profiles. Similarities never change. Newly selected original controls do not
remain controls, and unselected original members do become references.

- **A (production):** each slot ranks all other labelled profiles against its
  original block's unlabelled pool. Negative metadata constraints are not reapplied.
- **B_local (filtering comparator):** same assignments and positives; filter A's
  negative list by the declared fixed plate rule. No outside-block reference can
  be added. AP, and its upper tail at the shared identity, cannot decrease.
- **B_global (global-reapplication comparator):** same assignments and positives;
  rebuild negatives using all currently unlabelled profiles in the **original
  union**, applying the fixed plate rule. This can add references from other
  original blocks as well as remove forbidden ones.
- **C_conditioned_A (separate conditioning diagnostic):** retain A's statistic but
  restrict its uniform assignment law to assignments where *every* A negative
  edge still obeys the fixed plate rule. This is not either B statistic. Small or
  identity-only support is reported as such, not used as a replacement null.

Identity mAP must agree across A/B_local/B_global. Changed reference counts mean
B models are **different statistics** and need not test the same hypothesis as A.
When a slot has positives but zero negatives, we explicitly retain it with AP=1;
this is not an implicit request to drop it as an AP-frame filtering pipeline
might do. The report records zero-reference fractions. AP without positives is
unsupported. No empty support is converted into a fabricated tail probability.

## Bounded synthetic design

`SCENARIOS` in `study.py` is the authoritative matrix, also saved with each result:

- Shared pool: 3 members / 5 controls, no fixed restriction.
- Same-plate: member counts `(2,1)`, control counts `(3,4)`, effects 0 and 4.
- Cross-plate balanced: `(2,2)` members / `(3,3)` controls, effects 0, 0.75, 4.
  Here “balanced” means equal counts across plates, not equal members and controls.
- Cross-plate imbalanced: `(3,1)` members / `(2,5)` controls, same three effects.
- Complete role-swap: `(2,0)` members / `(0,2)` controls, effect 4.
- Reference-exhaustion: `(3,0)` members / `(0,2)` controls, effect 4; some B slots
  have no negatives.
- Clustered cross-plate: balanced counts above, effects 0 and 4, unit SD 2.5.
- Three-plate cross-reference overlap: `(1,1,1)` / `(2,2,2)`, effect 4;
  intentionally unsupported by A, with no alternative substituted.

Each profile has 8 features. With plate index `p` and feature index `j`,
`X[i,j] = epsilon[i,j] + plate_effect*p + cluster_sd*u[unit(i),j]`, with independent
standard-normal epsilon and unit effects. Units pair consecutive profiles within
a plate (last profile is a singleton if needed), with members ordered before
controls. Unit is a source of feature dependence, **not a declared negative
restriction or an additional permutation block**. There is no query-specific
feature effect. Data seeds are 7, 19, 41; datasets of the same shape share residual
noise across effect levels. Rows with shared seeds/controls are not treated as
independent Bernoulli trials. No calibration or rejection rates are reported.

Similarity is negative squared Euclidean distance, cast to float32; the AP rank
key is float32 `1 - similarity`. Tied positives precede tied negatives, matching
the pinned production rank convention. Exact distributions enumerate uniform
blockwise subsets; complete positives make within-block slot order irrelevant.
Finite-support upper-tail probabilities include the identity and use
`hits/support`, with production's float64 tie tolerance. Distribution SD is the
population SD; quantiles use NumPy's default linear interpolation. W1 is measured
in mAP units; the ECDF distance is the maximum absolute ECDF gap (no KS p-value).

The production sampler also runs with Philox seed 156 and 2,048 draws per dataset.
Its output is checked against evaluation on the identical `_draws_np` assignments;
both B statistics use those very assignments. Sampled p-values use
`(1+hits)/(1+draws)`. JSON reports approximate binomial Monte Carlo SEs and paired
SEs for Δp; these are **not calibration intervals** or simultaneous bounds. Main
reported comparisons are exact conditional on the features, so there is no
Monte Carlo uncertainty in those comparisons. Three seeds do not establish
population-general effect sizes or exchangeability.

## Reproduce (from this repository's root)

Python 3.11.13 was used. This creates an isolated temporary snapshot/environment;
it does not modify another worktree or install PR #156 into the main environment.
`source.py` verifies every tracked `src/copairs` file against the Git pin before
loading it. The Git object must be available locally (fetch it if needed).

```bash
PIN=eabb22bf87c65ec12219297d7d85fc8d2b58bf9c
git cat-file -e "$PIN^{commit}" || git fetch origin "$PIN"
STUDY_TMP=$(mktemp -d /tmp/copairs-pr156-study.XXXXXX)
mkdir "$STUDY_TMP/source"
git archive "$PIN" | tar -x -C "$STUDY_TMP/source"
uv venv "$STUDY_TMP/venv" --python 3.11
uv pip install --python "$STUDY_TMP/venv/bin/python" \
  -r analysis/pr156_null_layouts/requirements.txt -e "$STUDY_TMP/source[test]"
export COPAIRS_STUDY_SOURCE="$STUDY_TMP/source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$STUDY_TMP/venv/bin/python"
RUFF="$STUDY_TMP/venv/bin/ruff"

# Small default: exact supports plus 2,048 production draws per dataset.
"$PY" analysis/pr156_null_layouts/study.py
"$PY" -m pytest -q analysis/pr156_null_layouts
"$RUFF" check analysis/pr156_null_layouts
"$RUFF" format --check analysis/pr156_null_layouts

# Relevant pinned production regression tests (slower than the small study).
"$PY" -m pytest -q "$COPAIRS_STUDY_SOURCE/tests/test_nulls_permutation.py" \
  -k 'matches_relabelled_average_precision or hits_match_scalar_draws or vectorized_matches_scalar_draws or pos_diffby'

# Optional larger sampler/seeds; do not overwrite the recorded first slice.
"$PY" analysis/pr156_null_layouts/study.py --draws 20000 \
  --seeds 7 19 41 73 101 --output "$STUDY_TMP/larger-results"
```

Study tests are opt-in via `COPAIRS_STUDY_SOURCE`, so normal repository pytest
collection without that variable does not require an external research snapshot.
No production dependencies/configuration are changed. The study uses one Python
process and bounded small arrays, with BLAS/OpenMP limited to one thread. See
[VALIDATION.md](VALIDATION.md) for actual commands, outcomes, runtime and memory.
Only compact summaries are saved: no raw draws, features, environment or copied
production source is committed.
