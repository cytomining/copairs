# PR #156 synthetic null-layout results

Pinned production source: `eabb22bf87c65ec12219297d7d85fc8d2b58bf9c`. Data seeds: [7, 19, 41]; Philox sampler seed: 156; 2048 draws/dataset.

Exact statistics below enumerate equally weighted blockwise subsets. Complete positive graphs make within-block slot order immaterial. Ranges span dataset seeds, **not confidence intervals**. Exact comparisons have no Monte Carlo error conditional on the generated features; they do not remove dataset uncertainty.

A = production pools; B_local = filter A references by fixed plate rule; B_global = reapply the rule to all unlabelled profiles in the original union. Group/control roles move with labels in all models. Δp = p(B) − p(A).

| Scenario | Exact support | B_local W1 | B_global W1 | B_local Δp | B_global Δp |
|---|---:|---:|---:|---:|---:|
| shared_exchangeable | 56 | 0.0000–0.0000 | 0.0000–0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| same_plate_zero | 50 | 0.0000–0.0000 | 0.0000–0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| same_plate_strong | 50 | 0.0000–0.0000 | 0.0000–0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| cross_balanced_zero | 100 | 0.0954–0.1291 | 0.0153–0.0350 | 0.3300–0.5100 | -0.1500–-0.0400 |
| cross_balanced_weak | 100 | 0.1338–0.2113 | 0.0312–0.0940 | 0.0200–0.3900 | 0.0000–0.2400 |
| cross_balanced_strong | 100 | 0.2287–0.2578 | 0.1772–0.2036 | 0.1500–0.8900 | 0.0900–0.6400 |
| cross_imbalanced_zero | 168 | 0.0967–0.1327 | 0.0127–0.0241 | 0.3631–0.4345 | -0.0833–0.0417 |
| cross_imbalanced_weak | 168 | 0.1495–0.2029 | 0.0258–0.1139 | 0.0060–0.5893 | 0.0000–0.2381 |
| cross_imbalanced_strong | 168 | 0.2737–0.2998 | 0.2227–0.2351 | 0.1250–0.8869 | 0.0774–0.6429 |
| cross_role_swap | 6 | 0.2222–0.2222 | 0.2222–0.2222 | 0.1667–0.1667 | 0.1667–0.1667 |
| cross_reference_exhaustion | 10 | 0.2194–0.2194 | 0.2194–0.2194 | 0.2000–0.2000 | 0.2000–0.2000 |
| cross_clustered_zero | 100 | 0.1272–0.2103 | 0.0329–0.0758 | 0.0500–0.7100 | 0.0000–0.1400 |
| cross_clustered_strong | 100 | 0.2010–0.2714 | 0.1165–0.2102 | 0.6200–0.8900 | 0.1300–0.5400 |
| overlapping_three_plate | unsupported | — | — | — | — |

## Distribution detail (data seed 7)

All seeds are in JSON; these rows illustrate both a zero-effect and stress cases.

| Scenario | Model | Mean | SD | q95 | Exact p |
|---|---|---:|---:|---:|---:|
| cross_balanced_zero | A | 0.6255 | 0.1044 | 0.8282 | 0.1300 |
| cross_balanced_zero | B_local | 0.7546 | 0.0845 | 0.8849 | 0.5400 |
| cross_balanced_zero | B_global | 0.6220 | 0.1069 | 0.7887 | 0.0900 |
| cross_balanced_strong | A | 0.6233 | 0.1113 | 0.8619 | 0.0500 |
| cross_balanced_strong | B_local | 0.8811 | 0.0602 | 1.0000 | 0.5800 |
| cross_balanced_strong | B_global | 0.8268 | 0.0803 | 1.0000 | 0.3000 |
| cross_imbalanced_strong | A | 0.5699 | 0.1050 | 0.7581 | 0.0179 |
| cross_imbalanced_strong | B_local | 0.8572 | 0.0677 | 1.0000 | 0.2560 |
| cross_imbalanced_strong | B_global | 0.8050 | 0.0904 | 1.0000 | 0.1429 |
| cross_clustered_strong | A | 0.6412 | 0.0978 | 0.8293 | 0.0900 |
| cross_clustered_strong | B_local | 0.8423 | 0.0578 | 0.9307 | 0.7100 |
| cross_clustered_strong | B_global | 0.7590 | 0.0858 | 0.8697 | 0.3400 |

## Reference-count and conditioning diagnostics

Counts depend on layout, not feature seed/effect. Means are over slots and exact assignments. The zero-reference fraction is over slots, not whole draws.

| Layout example | Mean references A / local / global | Local / global zero-reference fraction | C support / A support |
|---|---:|---:|---:|
| cross_balanced_strong | 3.000 / 1.800 / 3.360 | 0.000 / 0.000 | 1 / 100 (identity only) |
| cross_imbalanced_strong | 4.250 / 2.342 / 3.832 | 0.000 / 0.000 | 1 / 168 (identity only) |
| cross_role_swap | 2.000 / 1.333 / 1.333 | 0.000 / 0.000 | 2 / 6 |
| cross_reference_exhaustion | 2.000 / 1.200 / 1.200 | 0.100 / 0.100 | 1 / 10 (identity only) |

## Sampler / independent-calculation checks

39 supported datasets; 3,528 enumerated A values compared against independent list sorting; maximum absolute error 0. Production sampler output also equals evaluation on the extracted Philox assignments, draw for draw.

Across all paired sampler comparisons, maximum approximate SE(Δp) = 0.0110; maximum |sampled Δp − exact Δp| = 0.0164. These are descriptive diagnostics, not simultaneous uncertainty bounds. The main tables use exact values.


## Positive/negative policy consistency and current guard asymmetry

The owner requires one pool-based relabelling policy for positive and negative pair restrictions; fixed metadata handling on both sides is future follow-up. The current positive-only `pos_diffby` rejection needs review for alignment and has not been separately approved as the final policy. This report documents current behavior; it does not claim that policy alignment has been implemented.

A matched six-profile example has one query member and two controls on each of two plates. Both specifications have the same observed positive edge (0,1) and negatives (0,4), (0,5), (1,2), (1,3). Public `average_precision` followed by `mean_average_precision(null='permutation')` gives:

| Positive plate rule | Negative plate rule | Pinned production outcome |
|---|---|---|
| none | `neg_diffby=['is_control','plate']` | accepted_by_production |
| `pos_diffby=['plate']` | same negative rule | rejected_by_production_guard |

The rejection says `pos_diffby values do not fix which relabelled profiles are positives`; its full text and explicit pair lists are saved in results.json. No rejected specification is assigned a production p-value, and **no guard is bypassed**.

Metadata-only enumeration of the 9 blockwise assignments finds 8 with forbidden A negative edges under a reapplied fixed plate rule, and 4 with zero positive edges if the fixed positive plate rule were reapplied. This is not a lower-level-kernel null run: no relabelled mAP or permutation p-value is computed for that rejected specification, and undefined assignments are not silently dropped. A full positive-graph/reference comparison, including a stated policy for zero-positive statistics, remains a coverage gap. The main bounded A/B_local/B_global comparisons above remain negative-reference comparisons with complete positive graphs and no `pos_diffby` rule.

## Interpretation and limitations

- Shared-reference and same-plate scenarios are invariance checks, not evidence that cross-plate restrictions are harmless. Differences can exist even at zero nuisance effect.
- B_local only removes negatives: its AP and upper-tail p cannot decrease on a common assignment. B_global can also add negatives from other original blocks, so neither its distribution nor its p-value has that ordering.
- Filtering changes reference counts and thus the statistic. Zero-negative slots retain all positives and have AP=1; they are reported, not omitted. Neither B is asserted to be a better-calibrated test or to test the same hypothesis as A.
- C_conditioned_A keeps A's statistic but accepts only assignments whose every A negative edge respects the fixed rule. Its separate support/p-values are in JSON; identity-only support is uninformative, not a substitute for A or B.
- Three-plate cross-reference sets overlap and production rejects them. No p-values or replacement nulls are reported for those cases.
- Clustered profiles share a Gaussian unit offset within each adjacent pair on a plate. Profilewise assignments need not be exchangeable; retaining geometry does not prove validity. No calibration/rejection-rate claim is made.
- Small complete-positive graphs, three synthetic seeds and one group per dataset are a bounded first slice, not an exhaustive design survey. Effects, dimensions, rules, identities, blocks and all per-seed summaries are in results.json.
- JSON includes means, population SD, q90/q95/q99, max ECDF distance, W1, exact upper-tail p-values, changed/zero reference counts, plus the production-sampler comparison on common assignments with approximate paired Monte Carlo SEs. Sampler p-values use (1+hits)/(1+draws); exact p-values use hits/support.

This analysis informs future consideration only; it does not change the decision to retain A or production code.
