# 0.6.0 release preparation: real-data comparisons

0.6.0 makes permutation the default mAP null, including for multilabel APs.
The implementations in #156 and #157 are merged; this page records the
prepublication comparison. See the [migration guide](migration-0.6.0.md) for
API and input-lifecycle changes. Publication remains a separate action.

## Minimal summary: what happens to p-values?

**mAP stays the same; the comparison used to judge its significance changes.**

- **Permutation versus joint:** permutation retains the measured similarities;
  joint replaces them with random pair similarities, sharing within-group
  positive pairs. Real clustering and the permitted label moves can make
  permutation p-values either larger or smaller.
- **Copied with equal pair counts:** members reuse the same random AP score,
  so adding replicates does not narrow its null. P-values can be larger than
  those from joint or permutation. **With unequal counts**, separate AP draws
  can instead make copied p-values much smaller.
- **Layout and precision matter:** plate restrictions and overlapping labels
  determine the permitted comparisons. Near the FDR cutoff, small sampling
  differences can flip several decisions together.

No method is uniformly more conservative, and a higher discovery count does
not establish better calibration. These comparisons use all three methods in
0.6.0; copied represents the earlier statistical model, not exact reproduction
of an older release's numbers.

## Ten real-data comparisons

All methods use the same AP table within a row. Counts below are groups flagged
at **BH-FDR 0.05**, with **100,000 draws per group**. Main counts use seed 0;
brackets show the range over seeds 0, 19 and 41 where it varies. The last column
counts different group decisions, not merely the difference between totals.

| Dataset / comparison | Groups | Permutation | Joint | Copied | Changed vs permutation (J / C) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LINCS: compound activity | 58 | 55 | 55 | 50 | 0 / 5 |
| LINCS: target consistency | 29 | 0 | 0 | 5 | 0 / 5 |
| Cell Health A549: guide activity | 100 | 50 | 50 | 31 | 0 / 19 |
| Cell Health A549: plate-matched controls | 100 | 51 | 50 | 33 | 1 / 18 |
| Cell Health ES2: guide activity | 100 | 77 | 76 [76–77] | 59 [45–59] | 1 / 18 |
| Cell Health HCC44: guide activity | 100 | 84 [84–85] | 81 | 71 [70–71] | 3 / 13 |
| Matched Cell Painting: compound activity | 300 | 263 | 266 | 266 [265–266] | 3 / 3 |
| Matched Cell Painting: target consistency | 413 | 80 | 79 | 88 [88–92] | 1 / 8 |
| nELISA secretome: compound activity | 300 | 143 | 133 [133–137] | 121 | 10 / 22 |
| nELISA secretome: target consistency | 413 | 6 | 4 | 13 | 2 / 13 |

The nELISA target comparison illustrates why totals alone are insufficient:
copied gains ten calls and loses three relative to permutation.

**Precision follow-up:** the ES2 copied count varied from 45 to 59 at 100,000
draws. A separate fixed **1,000,000-draw** run gave **45 in all three seeds**.
The primary table retains its common draw budget rather than substituting that
follow-up. Across the full primary comparison, 27 group/method decisions varied
with seed; the detailed evidence identifies them.

## Small real examples

Two groups illustrate opposite p-value changes under the copied model:

| Real group | Pair-count pattern | Observed mAP | Permutation p | Joint p | Copied p |
| --- | --- | ---: | ---: | ---: | ---: |
| LINCS compound BRD-K68132782-003-13-8 | Six members, each with 5 positives among 29 candidates | 0.4912 | 0.00726 | 0.00106 | 0.04725 |
| LINCS target PSMB1 | Two members, each with 1 positive, among 49 and 50 candidates | 1.0000 | 0.01068 | 0.00989 | 0.00039 |

These are raw p-values from seed 0, selected to illustrate the two conditions.
For the perfect PSMB1 pair, an independent calculation gives joint probability
`1/98 = 0.010204` and copied probability `1/(49×50) = 0.000408`, consistent with
sampling. The copied null's different configurations are sampled separately;
the joint null retains their shared positive pair.

A still smaller explanation uses **six real LINCS wells**: compound wells E13
and E14 at two doses, and controls A01–A04. The treated wells find each other
first, giving mAP 1. But controls A02 and A03 also form a perfect pair. Moving
two labels over the unchanged measurements gives 15 possible assignments;
exactly two achieve that perfect score. The exact permutation p-value is
**2/15 = 0.133**, versus **0.111 for joint** and **0.200 for copied**. All fifteen
assignments and independent checks are included with the analysis.

## Scope and reproducibility

These datasets appear in [Kalinin et al. (2025)](https://doi.org/10.1038/s41467-025-60306-2).
This is a workflow/model-sensitivity check, not reproduction of the published
figures or a calibration experiment:

- LINCS uses the example plate and pools doses; the paper used same-dose,
  cross-plate, sphered profiles. Target tests here include all 51 annotated
  compounds, unlike the earlier activity-selected smoke check.
- Cell Health retains six wells per guide and cutting controls in each cell
  line. The plate-matched variant changes reference rules, so its mAP can differ
  from the pooled-control row; scores remain identical across methods within
  each row.
- Matched Cell Painting/nELISA uses 300 shared compounds, harmonized by ID/count
  as in the paper. Target tests include all annotated compounds, with no
  activity-based selection. Singleton targets are untestable.
- Permutation assumes exchangeability within its pools; normalization or
  connected multilabel components do not establish that assumption.

All **90 case/method/seed runs** completed without fallback. Observed mAP and
normalized mAP matched across methods and seeds. Installed library files matched
candidate `742f2d8bdd24528cfbbd181ba46b940297324fd5` byte-for-byte.

[Reproduction instructions, pinned input hashes, scripts and full results](https://github.com/cytomining/copairs/tree/bc351a281182c553b11aa230fb7fa74e575a6ff6/analysis/release_0_6_0)
are preserved with the PR. No runtime code, default resource policy or
publication workflow is changed by this release-preparation report.
