# Cell-level gate: audit after the results

This page belongs to the cell-gate study in the [README](../README.md#cell-level-gate-confirmatory-study). Everything on it is exploratory: it was looked at after the results were known, and it changes no committed result, label or verdict. The counting rules are fixed in `writeup/inputs/audit_013_definition.txt`; the values come from `writeup/inputs/audit_013.json` (sha256 b6c3736998aa40301421f29f99ac113b8bd27e1e472852877316871804978cb1), which reads the run records and the committed result file. Notation: p10 is the share of pairs where the candidate succeeds and S0 fails (an improvement), p01 the share where the candidate fails and S0 succeeds (a regression), and Δ = p10 − p01.

## What the Oracle effect estimate compares (Exploratory)

On the Oracle split, a candidate runs once (repeat zero, with the same LLM seed as S0's repeat zero), while S0 is the case-wise majority of its 5 repeats. So the Oracle effect estimate of the cell-gate study is the mean, over the fixed Oracle incidents (50 per cell, equal cell weights), of a single candidate run minus the S0 majority. It is not the expected difference between two single-run policies. This was the registered design; the planning simulation of the null label rates used the same majority.

The table shows, from S0's own records, what this rule gives for a candidate that behaves exactly like S0: the cell mean of a single S0 run minus the majority, the same for S0's repeat zero, and the number of cases whose repeats disagree.

| Cell | Single run − majority | Repeat zero − majority | Cases with mixed repeats |
| --- | --- | --- | --- |
| bad_deployment/L1 | 0.000 | 0.000 | 0 of 50 |
| bad_deployment/L2 | -0.040 | -0.060 | 7 of 50 |
| dependency_failure/L1 | 0.000 | 0.000 | 0 of 50 |
| dependency_failure/L2 | -0.004 | 0.000 | 1 of 50 |
| traffic_spike/L1 | 0.044 | 0.080 | 15 of 50 |
| traffic_spike/L2 | 0.068 | 0.060 | 17 of 50 |

The table is on CD. On ES the offsets are zero except dependency_failure/L2 (-0.004), because S0's ES sits at the floor or the ceiling in every cell (see below).

**What this means for the labels.** The offsets reach the size of ε (0.050) and of τ_cell (0.050 in bad_deployment/L2).

- **Severe Local Harm.** The lowest cell of each of the 14 Severe candidates lies between -0.140 and -0.820, well beyond the largest offset. The Severe labels, and the RQ9-1 result built on them, stand as reported.
- **Local Harm without Severe.** Of the 11 such candidates, 11 have bad_deployment/L2 below −τ, and 9 have no other cell below −τ. In that cell a candidate that behaves like S0 already starts near -0.060. These labels should be read as limited evidence of harm. No label is changed.

## Thresholds h and δ (Exploratory)

- δ(N) resamples cases without cell stratification and picks one random pair of S0 repeats per draw (`exact.null_diff_quantile_exact`, as registered); C4 uses a cell-stratified paired bootstrap (`gates.gate_c4`). The mismatch noted for the mean-gate study is still present here.
- h(n_c) is a quantile of the minimum over the six cells of S0-vs-S0 cell differences (`rq9.cell_threshold_h`), so it already acts jointly on the cells; on CD, h(70) = 0.114 with an achieved false rejection of 0.035. The uncertainty of h itself is not reported: its draws reuse the same repeat pairs and the same 70 cases per cell.
- The calibration pairs use different LLM seeds, while every gate comparison pairs a candidate with S0 under the same seed. Different-seed S0 repeats disagreed on CD in 258 of 4200 pairs; by cell:

| Cell | CD disagreement between S0 calibration repeats |
| --- | --- |
| bad_deployment/L1 | 0 of 700 |
| bad_deployment/L2 | 44 of 700 |
| dependency_failure/L1 | 4 of 700 |
| dependency_failure/L2 | 4 of 700 |
| traffic_spike/L1 | 120 of 700 |
| traffic_spike/L2 | 86 of 700 |

The same-seed agreement of 575 of 600 pairs compares sequential with concurrent runs under two configurations, so it does not tell how much the same seed reduces disagreement in the gate comparisons.

## S0's cells: ceiling and floor (Exploratory)

| Cell | Dev-B CD | Dev-B ES | Oracle majority CD | Oracle majority ES |
| --- | --- | --- | --- | --- |
| bad_deployment/L1 | 70/70 | 0/70 | 50/50 | 0/50 |
| bad_deployment/L2 | 67/70 | 0/70 | 50/50 | 0/50 |
| dependency_failure/L1 | 70/70 | 70/70 | 50/50 | 50/50 |
| dependency_failure/L2 | 70/70 | 70/70 | 50/50 | 50/50 |
| traffic_spike/L1 | 8/70 | 0/70 | 2/50 | 0/50 |
| traffic_spike/L2 | 3/70 | 0/70 | 1/50 | 0/50 |

- **ES.** S0's ES is at the floor in 4 of 6 cells (every Oracle repeat and Dev-B) and at the ceiling in 2. In a floor cell p01 is zero, so a regression cannot appear on ES there; the ES cell condition could act only in the dependency cells. This is part of why C4 and C5 accepted the same candidates on ES; on Dev-B no candidate that C4 accepted on ES had a dependency cell below −h.
- **CD.** S0's CD is at the ceiling in 4 cells (Oracle majority) and at most a tenth in 2. A ceiling cell can show only regressions (p10 is zero), a near-floor cell mostly improvements, so no cell measures both directions well.
- This structure is kept as it is: the cell-gate result is a result under this stress test, not under a balanced baseline.

## Shared LLM seeds across levels (Exploratory)

The LLM seed rule takes the family, the incident seed and the repeat, but not the level (`seeds.derive_llm_seed`). The L1 and L2 incidents that share an incident seed number are different incidents (the L2 generator uses its own random stream), but they get the same LLM seed.

| Split | Cases | Family and seed groups | Groups holding both an L1 and an L2 case |
| --- | --- | --- | --- |
| Dev-B | 420 | 210 | 210 |
| Oracle | 300 | 150 | 150 |
| Calibration | 420 | 210 | 210 |

So the L1 and L2 cells of one family are not independent at the seed level. How much this couples their outcomes is not measured.

## Candidate pairs on Dev-B (Exploratory)

All 40 proposer candidates have Dev-B records paired with S0's repeat zero: 16800 pairs, 16800 of them with the same LLM seed. Each candidate and S0 ran once on Dev-B, so the candidate's own run-to-run variation cannot be estimated from these records.

## Idea groups (Exploratory)

The grouping method (same condition, same action or same added step; no gate or oracle values) is fixed in Section 6.6 of the cell-gate pre-registration, committed in 12a36fd before any proposer call. The group file (sha256 c3893a7737fe9e2cc76d3b4e1c2f5ffe2f1c8804ecc95fc68798e70a14d8bcc9) was committed with the pool in 4c3b532, before the run plan (7b17c8b) and before the main run started (2026-11-06T22:30:00+00:00). One rater (Claude) made the groups; there is no agreement check.

## What the Fisher and Clopper–Pearson results support (Exploratory)

- The Fisher p of RQ9-1 (1/9044) compares the set C5 rejected with a random rejection of the same size, inside this frozen pool and with the labels held fixed. It describes this pool.
- It is not evidence about a new population of candidates. The candidates share the S0 majority baseline, the LLM seeds and, within an idea group, the idea; the sensitivity check with one candidate per group (1/429) addresses only the last.
- The Clopper–Pearson interval of RQ9-3 treats the candidates as independent draws, which candidates of one idea group are not.

## Pairs in the Ticket 012 base-rate check (Exploratory)

The 120 same-seed pairs of the base-rate check pair the base run of each of 60 Dev-A cases with each of two reruns, so every pair shares the base run with one other pair. The cases carry 30 distinct LLM seeds (L1 and L2 share). Of the 16 differing pairs, 3 cases contributed both of their pairs and 10 cases one pair; the two reruns differed in 10 cases, a comparison the rule did not use. The Clopper–Pearson bound behind k treats the pairs as independent, so its effective sample is smaller than the count. k was an exploratory rule and is not re-judged here.

## S0′ as a baseline with mid-range success (Exploratory)

From the same three runs (Dev-A, 10 cases per cell and run): correct dispositions and Evidence-Safe outcomes per run, and the disagreements of the pairs above.

| Cell | CD per run | ES per run | Disagreeing pairs |
| --- | --- | --- | --- |
| bad_deployment/L1 | 10, 10, 10 | 6, 6, 6 | 0 of 20 |
| bad_deployment/L2 | 6, 7, 5 | 3, 4, 3 | 2 of 20 |
| dependency_failure/L1 | 8, 8, 9 | 8, 8, 9 | 1 of 20 |
| dependency_failure/L2 | 5, 6, 5 | 5, 6, 5 | 3 of 20 |
| traffic_spike/L1 | 8, 10, 8 | 1, 2, 1 | 4 of 20 |
| traffic_spike/L2 | 7, 8, 6 | 2, 2, 2 | 6 of 20 |

Both an improvement and a regression could be observed against S0′ in 5 of the six cells on CD and 6 on ES. With so few cases per cell this shows only that S0′ is not at the floor or the ceiling; it does not give rates precise enough to plan a study on.
