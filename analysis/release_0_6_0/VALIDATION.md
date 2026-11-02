# Validation receipts

Recorded 2026-11-01 for the 0.6.0 release-preparation PR.

- All 12 downloaded inputs passed their SHA256 checks.
- Ten workflows × three nulls × seeds 0/19/41: **90 successful comparisons**,
  each requesting 100,000 draws per tested group; no fallback or rejected layout.
- **1,913 group tests per seed**, with per-group evidence for all methods.
  These overlapping workflows are not independent calibration experiments.
- mAP and normalized mAP matched across methods. The report separately checks
  mAP across seeds, exact group membership of changed decisions, and
  gained/lost-count arithmetic.
- Matched modalities have identical compound/target membership. The blocked
  A549 case has exactly two members on each of three plates for every guide.
- Independent six-well enumeration and perfect-pair analytical probabilities
  agree with production sampling. The latter checks joint/copied at all seeds.
- Fixed one-million-draw ES2/copied follow-up: 45 calls at each of the three
  seeds, compared with 45–59 at the primary 100,000-draw budget.
- All 17 installed library files and all 17 worktree library files match the
  previously validated main candidate `742f2d8bdd24528cfbbd181ba46b940297324fd5`.
- Ruff lint/format checks pass for all 44 tracked Python files.
- `uv lock --check`, package builds and strict MkDocs build pass.
- The built wheel's library files match the candidate hashes. The sdist includes
  `docs/release-0.6.0.md` and excludes the research scripts/results.
- The release-page table matches the generated table exactly.

## Final review and reproduction checks (2026-11-02)

- Fixed two analysis-validation gaps: Cell Health preprocessing now consumes
  only hash-verified manifest inputs, and report generation checks both score
  columns and group identities across every method and seed.
- Regression evidence: the previous report accepted altered normalized scores
  and group membership; both cases now raise. Three report tests pass.
- Rebuilt all ten prepared datasets with an extra, invalid, unlisted Cell Health
  file present. Every prepared frame and case/preprocessing record matches the
  archived inputs exactly; the unlisted file is ignored.
- Regenerated reports and the six-well enumeration/sampling example. All output
  files are byte-identical to the committed results; no table counts changed.
- Independently recomputed BH correction with SciPy for all 90 comparisons and
  three precision runs. Every decision agrees, including 45 ES2/copied calls at
  each precision seed. Round-trip CSV parsing preserves the original precision;
  default pandas parsing introduces only last-bit corrected-p differences.
- All 309 archived artifact checksums pass.
- Python 3.11 source suite: **391 passed** (388 library tests plus 3 report tests).
- Python 3.14 installed-wheel suite from the fresh sdist: **388 passed**, with
  `PYTHONPATH` unset and optional Arrow tests enabled (DuckDB 1.4.4).
- A separate installed-wheel smoke check passes with unpinned current dependencies,
  including DuckDB 2.0.0, for single-label/multilabel data, all three nulls,
  default/explicit permutation equivalence, and zero-draw unknown significance.
- Lint, formatting, lock, strict documentation, build and strict Twine checks pass.
  All 17 library files in source, wheel and sdist still match candidate `742f2d8`.

This PR changes analysis and documentation, not the runtime library. It does
not claim a new full-suite test run from the local analysis environment;
the PR's normal CI runs the package tests. It creates no tag and publishes no
GitHub or PyPI release.
