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

This PR changes analysis and documentation, not the runtime library. It does
not claim a new full-suite test run from the local analysis environment;
the PR's normal CI runs the package tests. It creates no tag and publishes no
GitHub or PyPI release.
