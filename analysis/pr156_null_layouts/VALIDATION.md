# Validation record

Run on 2026-10-28 UTC, Linux, CPython 3.11.13. Production source:
`eabb22bf87c65ec12219297d7d85fc8d2b58bf9c`. Dependency versions are pinned in
`requirements.txt`; Ruff is **0.15.20**. All paths below are relative to the
repository root except the intentionally external source/environment. `PY`,
`RUFF`, and `COPAIRS_STUDY_SOURCE` are established by the README bootstrap.

| Command/check | Observed outcome |
|---|---|
| `"$PY" analysis/pr156_null_layouts/study.py` | Exit 0; 39 supported dataset/scenario combinations plus 3 explicitly unsupported overlap cases; exact supports 6–168; 2,048 production draws per supported dataset (79,872 total). |
| Same default command, repeated | `results.json` and `REPORT.md` byte-identical. |
| `"$PY" -m pytest -q analysis/pr156_null_layouts` | **27 passed in 6.73s**. |
| `"$PY" -m pytest -q "$COPAIRS_STUDY_SOURCE/tests/test_nulls_permutation.py" -k 'matches_relabelled_average_precision or hits_match_scalar_draws or vectorized_matches_scalar_draws or pos_diffby'` | **37 passed, 16 deselected in 162.57s**. No claim that the entire production suite was run. |
| `"$RUFF" check analysis/pr156_null_layouts` | All checks passed. |
| `"$RUFF" format --check analysis/pr156_null_layouts` | 5 files already formatted. |
| `env -u COPAIRS_STUDY_SOURCE "$PY" -m pytest --collect-only -q analysis/pr156_null_layouts` | No tests collected (expected pytest exit 5); research tests do not affect ordinary repository collection. |
| `git diff --check` | No whitespace errors. |

## Independent and semantic checks

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

Final default study: **5.880 s wall**, 5.579 s user, 0.252 s system;
**135,568 KiB maximum RSS (~132.4 MiB)**, one study process. Environment:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Python's
`time.monotonic` and `resource.getrusage(resource.RUSAGE_CHILDREN)` measured a
`subprocess.run([sys.executable, 'analysis/pr156_null_layouts/study.py'])` child.
An initial `/usr/bin/time` attempt could not run because that executable was
absent; the Python measurement above replaced it. No production test failure
was encountered. The optional larger study was documented, not run.

SHA-256 after two default runs:

```text
c0b3860e363baab4caa7e635f0eb7ffd22d397103497eac6e4bcc08cb1dd4c8d  results/results.json
8c82e0dced15e435dfc889a143b07751adcb64ae170387f4fde154891f8e31cc  results/REPORT.md
```

Only analysis files are changed. No feature matrices, raw permutation draws,
external environments, copied production source, private logs or decision/session
files are included. Limits: small synthetic complete-positive groups, three
feature seeds, custom fixed negative squared-Euclidean similarities, no
calibration proof, and no guarantee for future PR #156 source revisions.
