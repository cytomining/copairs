# Real-data comparison for the copairs 0.6.0 release

Compare `permutation`, `joint`, and `copied` on ten fixed real-data workflows.
The candidate is `742f2d8bdd24528cfbbd181ba46b940297324fd5` (#156 + #157).
`candidate-library.json` checks all 17 installed library files against that
candidate. This directory does not change the library or release workflow.

## Results

- [PR-ready table](results/table.md)
- [Per-group scores, p-values and decisions](results/group-comparison.csv)
- [Per-case counts and timing records](results/case-summary.csv)
- [Groups whose decisions change across seeds](results/seed-stability.csv)
- [Two illustrative groups](results/illustrative-groups.csv)
- [Independent perfect-pair calculation](results/perfect-pair-oracle.json)
- [Six real wells: every label assignment](results/six-well-assignments.csv)
- [Six-well exact and sampled probabilities](results/six-well-summary.json)
- [Higher-precision ES2 follow-up](results/precision/summary.csv)

Every method receives the same AP table within a comparison. Sampling uses
100,000 draws per group and seeds 0, 19 and 41, with BH-FDR correction at 0.05
within each comparison. Observed mAP and normalized mAP are identical across
methods and seeds. No unsupported case is silently switched to another null.
These are model-sensitivity/workflow checks, not a calibration benchmark.

## Datasets and fixed selection

All datasets appear in [Kalinin et al., Nature Communications (2025)](https://doi.org/10.1038/s41467-025-60306-2).
Exact input URLs, hashes and byte sizes are in [data-sources.json](data-sources.json).
The preprocessing follows the paper's recipes using released pycytominer 1.2.3;
this is not an exact reproduction of its historical software environment or figures.

| Input | Preparation and comparison |
| --- | --- |
| LINCS plate SQ00014812 | Supplied normalized/feature-selected profiles: 384 wells, 493 features, 24 DMSO controls, 58 compound groups with 6–12 wells. Different doses are pooled as in copairs' example, unlike the paper's same-dose/cross-plate analysis. |
| LINCS targets | Feature medians for all 51 annotated compounds, without activity filtering; 29 multi-member target groups. This differs from the earlier 28-target smoke check that selected active compounds first. |
| Cell Health | Nine plates, three cell lines. Whole-plate MAD robustization; variance/correlation/NA/blocklist/outlier feature selection; remove Costes features. Retain guides having exactly six wells per cell line and all Chr2/Luc/LacZ cutting controls: 100 guides plus 192 controls per cell line, 506 features. |
| Cell Health plate-matched variant | Same A549 profiles and positive graph; use same-plate controls and preserve two members per plate in permutation. This changes the AP comparison itself, so mAP need not match the pooled-control row. |
| Matched Cell Painting / nELISA | Whole-plate MAD robustization; feature selection for Cell Painting, none for nELISA. Harmonize compound IDs and counts as in the paper: 1,512 profiles, 300 compounds, 256 controls; 615 / 191 features. This is ID/count harmonization, not a newly asserted well-level pairing. |
| Matched target tests | Median profiles for all 300 annotated compounds in each modality, without activity filtering. 413 tested target groups; singleton labels are untestable. Multiple target labels are retained. |

Group selection uses metadata, never the resulting null p-values. Whole groups
are retained and AP contexts are rebuilt for every comparison. The two example
groups were chosen after comparison to illustrate equal/unequal pair counts;
they are not an unbiased sample of effect sizes. The six-well subset uses the
first compound alphabetically and first wells by ID, independent of scores.

The ES2 copied test was selected for a **fixed one-million-draw follow-up** after
its 100,000-draw results showed a large seed-dependent change in FDR decisions.
The follow-up is reported separately; it does not replace the common-budget
primary table or add adaptive stopping to copairs.

## Reproduce

Use Python 3.11. Install a wheel built from the candidate commit, then the pinned
analysis dependencies. No inputs or generated profile matrices need to be stored
in the source tree; use an external working directory with room for roughly
100 MB of downloads and prepared data, plus transient sampling memory.

```bash
python -m venv /tmp/copairs-release-analysis
source /tmp/copairs-release-analysis/bin/activate
pip install /path/to/copairs-0.6.0-py3-none-any.whl
pip install -r analysis/release_0_6_0/requirements.txt
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CODE=analysis/release_0_6_0
WORK=/tmp/copairs-release-data
OUT=/tmp/copairs-release-results
python "$CODE/prepare.py" --workdir "$WORK"
python "$CODE/compare.py" --workdir "$WORK" --output "$WORK/primary" --seeds 0
python "$CODE/compare.py" --workdir "$WORK" --output "$WORK/repeats" --seeds 19 41
python "$CODE/report.py" --workdir "$WORK" --inputs "$WORK/primary" "$WORK/repeats" --output "$OUT"
python "$CODE/explain.py" --workdir "$WORK" --output "$OUT"
python "$CODE/precision_check.py" --workdir "$WORK" --output "$OUT/precision"
```

`prepare.py` downloads public files with curl and verifies every SHA256 checksum.
It never executes downloaded code. All Python asserts must be enabled.

Resources: four workers, reference cap 512, pool-table budget 256 MiB, optional
key-table budget 32 MiB. These budgets are not total-process RAM caps. Each
copied call uses a fresh temporary cache. Timings are execution records, not a
controlled performance comparison. Dependency versions and prepared-input
summaries accompany the results.

## Interpretation

- **The retrieval score does not change; the reference distribution does.**
  Permutation moves labels over observed profiles, retaining real similarities;
  joint generates random pair similarities, sharing each within-group positive
  pair between its two members.
- **Copied can raise or lower p-values.** Equal pair-count configurations reuse
  one AP draw, so adding matching replicates does not narrow that null. Different
  configurations use separate draws, which can instead make the averaged tail
  much smaller; the real PSMB1 pair provides an independently checked example.
- **Experimental layout matters.** Controls, plate restrictions and overlapping
  target labels determine which comparisons and relabellings are permitted.
  Label exchangeability is an assumption, not established by connectivity alone.
- **Threshold decisions can be sampling-sensitive.** Small p-value differences
  near the FDR cutoff can flip several calls together. Report seed ranges and
  use a sufficiently large, fixed draw budget for final analysis.

Neither more discoveries nor fewer discoveries demonstrates better calibration.
The related workflows reuse data and labels; their groups are not independent
replicates of a calibration experiment. The GitHub release remains unpublished
unless separately authorized.
