# Validation record

Run on 2026-10-28 UTC, Linux, CPython 3.11.13. Production source:
`eabb22bf87c65ec12219297d7d85fc8d2b58bf9c`. Dependency versions are pinned in
`requirements.txt`; Ruff is **0.15.20**. All paths below are relative to the
repository root except the intentionally external source/environment. `PY`,
`RUFF`, and `COPAIRS_STUDY_SOURCE` are established by the README bootstrap.

| Command/check | Observed outcome |
|---|---|
| `"$PY" analysis/pr156_null_layouts/study.py` | Exit 0; main matrix unchanged: 39 supported datasets plus 3 unsupported overlap cases, exact supports 6–168, 79,872 production draws. Added matched public-guard diagnostic: negative-only accepted (32 draws), matched positive+negative rejected. |
| Same default command, repeated | `results.json` and `REPORT.md` byte-identical. |
| `"$PY" -m pytest -q analysis/pr156_null_layouts` | **28 passed in 8.00s** after the policy-consistency follow-up. |
| `"$PY" -m pytest -q "$COPAIRS_STUDY_SOURCE/tests/test_nulls_permutation.py" -k 'matches_relabelled_average_precision or hits_match_scalar_draws or vectorized_matches_scalar_draws or pos_diffby'` | **37 passed, 16 deselected in 162.57s**. No claim that the entire production suite was run. |
| `"$PY" -m pytest -q "$COPAIRS_STUDY_SOURCE/tests/test_nulls_permutation.py" -k pos_diffby` | Follow-up guard regression: **1 passed, 52 deselected in 3.77s**. The broader 37-test run above was completed in the initial study; pinned production source is unchanged. |
| `"$RUFF" check analysis/pr156_null_layouts` | All checks passed. |
| `"$RUFF" format --check analysis/pr156_null_layouts` | 5 files already formatted. |
| `env -u COPAIRS_STUDY_SOURCE "$PY" -m pytest --collect-only -q analysis/pr156_null_layouts` | No tests collected (expected pytest exit 5); research tests do not affect ordinary repository collection. |
| `git diff --check` | No whitespace errors. |

## Independent and semantic checks

- Matched six-profile plate rules have identical observed positive/negative pairs
  and mAP, yet the public permutation entry point accepts negative-only restrictions
  and rejects the version with `pos_diffby=[plate]`. The full rejection is saved.
  No guard is bypassed. All nine assignments are retained in a metadata-only
  diagnostic: eight violate fixed negative restrictions, four would have no
  positives under fixed positive reapplication. No relabelled mAP or p-value is
  computed for that rejected specification; undefined assignments are not dropped.
- All 42 original quantitative/unsupported result records equal those from
  commit `250f385` exactly (`old['results'] == new['results']`). The matched-guard
  record is separate; the main A/B_local/B_global matrix has not expanded.

- 3,528 exact A assignment values, across all supported datasets, agree with
  independent direct positive/negative list sorting: maximum absolute error 0.
- Observed-identity mAP agrees across A, B_local and B_global for every dataset.
- Hand-calculated AP/tie examples check the independent evaluator, not just
  agreement between two implementations of the same assumptions.
- Explicit control-role swap test: original controls become members; original
  members become negatives. Mixed-label references are checked by hand.
- Explicit multi-block example: B_global adds other-block references that
  B_local cannot include.
- Public metadata/AP oracle rebuilds `group` and `is_control` for every small
  assignment under no/same/different plate restrictions, then compares B_global
  against the real AP pipeline (without the study's reference helper).
- Exact and production assignments preserve block counts; within-block slot
  permutations leave complete-positive statistics unchanged.
- Invariance, local-filter monotonicity, zero-negative slots, identity-only
  no-reference pools, undefined no-positive layouts, empty support reporting,
  and overlapping-reference rejection are covered.
- Production `permutation_null` output equals evaluation on its extracted Philox
  assignments, draw for draw; its plus-one p agrees with the reported sampled p.
  Maximum paired sampler SE(Δp) was 0.011043 and maximum absolute sampled/exact
  Δp difference was 0.016428. Main comparisons use exact finite distributions.

## Resource measurement and reproducibility

Updated default study including the public guard diagnostic: **7.354 s wall**,
6.714 s user, 0.864 s system; **181,876 KiB maximum RSS (~177.6 MiB)**,
one study process. Initial negative-only matrix: 5.880 s and ~132.4 MiB. Environment:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Python's
`time.monotonic` and `resource.getrusage(resource.RUSAGE_CHILDREN)` measured a
`subprocess.run([sys.executable, 'analysis/pr156_null_layouts/study.py'])` child.
An initial `/usr/bin/time` attempt could not run because that executable was
absent; the Python measurement above replaced it. No production test failure
was encountered. The new study test initially failed on incidental negative-pair
list ordering; canonical sorting of recorded pair lists fixed the diagnostic and
its reproducibility, without changing the pairs or production code. Final study
tests pass. The optional larger study was documented, not run.

SHA-256 after two default runs:

```text
11966d3fb3e2321c58011d4798933066f0272b3d8cdbea49a7e812f8a878cd64  results/results.json
1a5ee4efeb81855c8d0d22286f272aa818ae747540e7cf507ae3bde3f08051fa  results/REPORT.md
```

Only analysis files are changed. Apart from the six explicitly declared profiles
in the tiny guard example, no feature matrices are saved. No raw permutation
draws, external environments, copied production source, private logs or
decision/session files are included. Limits: small synthetic complete-positive groups, three
feature seeds, custom fixed negative squared-Euclidean similarities, no
calibration proof, and no guarantee for future PR #156 source revisions.
