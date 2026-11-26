# Cell-level gate: a simulation of admission rules and a replay on the Ticket 009b pool

This page belongs to the cell-gate study in the [README](../README.md#cell-level-gate-confirmatory-study). Everything on it is exploratory: the simulation tests whether the existing admission rules can control a margin-level loss in one cell at a budget the project can run, and the replay applies the same rules to the Ticket 009b candidates. Neither changes a committed result, label or verdict of the cell-gate study.

## Setup (Exploratory)

- **Frozen plan.** `plans/ticket-013-sim-plan.json` (sha256 8c97f64145d8f9b79cb7cb7591fafbd4e1d4895f57c8f3c05e7d797db3db43a0) and its addendum `plans/ticket-013-sim-plan-addendum.json` (sha256 3ccd2b03beb17942dea5196f9f83504584a0584a5bf44b61b657fad99d7dcebe) were frozen and tagged before any simulation code ran. Results: `results/ticket-013/sim.json` (sha256 02b95b098df79c40ac4a84278551cefab0e15d726fef27334de01d2cb0feba46), `sim-gain-correction.json` (sha256 d15a49797d19b0a1a42ab5ed01a501125a1432ef64ce2d663bc1f2c2a31872b8), `replay-009b.json` (sha256 e134c4dab5a4b7c95bdfc972f57ded76eb806767cd009651915375f60e511640) and `replay-009b-c6.json` (sha256 8d61b1ec89ad1cd498e69f03e55197c96b458652350821a1b23b02ca8f5d8cd9).
- **No LLM calls.** Each simulated pool holds 40 candidates; every condition uses 2000 pools and 2000 bootstrap draws. Before the main run, the code reproduced the null label rates of the Ticket 009b planning simulation (Local Harm 0.167590 against 0.167205, Severe 0.000505 against 0.000475).
- **Baselines.** Baseline a uses S0's case rates from its calibration records (the ceiling and floor cells of Ticket 009b kept as a stress test). Baseline b uses S0′'s case rates from ten Dev-A cases per cell; it is a stylized mid-range structure, not an estimate of S0′.
- **Gates.** C4 and C5 as in the cell-gate study; C6-IUT = C4 and, in every cell, a one-sided lower bound of δ_c above −m_c (Tango score as the primary bound, the exact conditional bound as the secondary, α = 0.05); SAGE λ1 and λ2, a per-cell SAGE cap, and a credit-type gate (ε = 0.005). **SAGE, the per-cell cap and the credit-type gate are our own one-round implementations of rules from other papers, and τ = 0.05 is our value; the SAGE paper tuned its parameters on held-out data.**
- **Go rule (frozen).** In baseline a or b, a combination (gate, n_c, m_c) counts toward Go when the worst false accept over the inadmissible scenarios and both loss-cell placements is at most 0.05, the acceptance of a representative Useful candidate is at least 0.7, and Dev-B needs at most 75000 episodes; a combination within 2 MC SE of a threshold is borderline and does not count.

## Go decision (Exploratory)

The frozen rule gave Go: 4 combinations in baseline a and 2 in baseline b. Infeasible scenarios: baseline a, m_c 0.10, global_up_one_cell_minus_2m, worst.

| Baseline | Gate | n_c | m_c | Worst false accept ± MC SE | Worst scenario | Useful power ± MC SE | Dev-B episodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | C6-IUT | 150 | 0.05 | 0.0442 ± 0.0007 | global_up_one_cell_minus_m, realistic | 0.7419 ± 0.0048 | 36900 |
| a | C6-IUT | 150 | 0.10 | 0.0410 ± 0.0033 | global_up_one_cell_minus_m, worst | 0.9081 ± 0.0030 | 36900 |
| a | C5 | 300 | 0.10 | 0.0112 ± 0.0016 | global_up_one_cell_minus_m, worst | 0.9978 ± 0.0003 | 73800 |
| a | C6-IUT | 300 | 0.10 | 0.0474 ± 0.0007 | global_up_one_cell_minus_m, realistic | 0.9978 ± 0.0003 | 73800 |
| b | C6-IUT | 70 | 0.05 | 0.0448 ± 0.0019 | global_up_one_cell_minus_m, worst | 0.7525 ± 0.0046 | 17220 |
| b | C6-IUT | 150 | 0.10 | 0.0452 ± 0.0018 | global_up_one_cell_minus_m, worst | 1.0000 ± 0.0000 | 36900 |

Loss cells: baseline a, bad_deployment/L1 (realistic placement) and traffic_spike/L1 (worst placement); baseline b, bad_deployment/L1 and dependency_failure/L2.

## Gate comparison (Exploratory)

Main dependence setting. n_c 600 exceeds the episode budget and is shown for the shape only.

**Baseline a, m_c = 0.05.** Each cell: worst false accept / representative Useful power.

| Gate | n_c 70 | n_c 150 | n_c 300 | n_c 600 |
| --- | --- | --- | --- | --- |
| C4 | 1.0000 / 0.5878 | 1.0000 / 0.9089 | 1.0000 / 0.9978 | 1.0000 / 1.0000 |
| C5 | 0.9867 / 0.5878 | 0.9349 / 0.9088 | 0.6822 / 0.9978 | 0.2471 / 1.0000 |
| C6-IUT (Tango) | 0.0154 / 0.2390 | 0.0442 / 0.7419 | 0.0472 / 0.9711 | 0.0489 / 0.9998 |
| SAGE λ1 | 0.9816 / 0.7686 | 0.9987 / 0.9733 | 1.0000 / 0.9997 | 1.0000 / 1.0000 |
| SAGE λ2 | 0.9678 / 0.1057 | 0.9987 / 0.1973 | 1.0000 / 0.3475 | 1.0000 / 0.5836 |
| per-cell SAGE λ1 | 0.0094 / 0.0006 | 0.0002 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| per-cell SAGE λ2 | 0.0094 / 0.0006 | 0.0002 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| credit-type | 0.0011 / 0.0010 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

**Baseline a, m_c = 0.10.** Each cell: worst false accept / representative Useful power.

| Gate | n_c 70 | n_c 150 | n_c 300 | n_c 600 |
| --- | --- | --- | --- | --- |
| C4 | 1.0000 / 0.5878 | 1.0000 / 0.9089 | 1.0000 / 0.9978 | 1.0000 / 1.0000 |
| C5 | 0.7181 / 0.5878 | 0.2444 / 0.9088 | 0.0112 / 0.9978 | 0.0059 / 1.0000 |
| C6-IUT (Tango) | 0.0312 / 0.5556 | 0.0410 / 0.9081 | 0.0474 / 0.9978 | 0.0482 / 1.0000 |
| SAGE λ1 | 0.9411 / 0.7686 | 0.9880 / 0.9733 | 0.9994 / 0.9997 | 1.0000 / 1.0000 |
| SAGE λ2 | 0.9287 / 0.1057 | 0.9880 / 0.1973 | 0.9994 / 0.3475 | 1.0000 / 0.5836 |
| per-cell SAGE λ1 | 0.0020 / 0.0006 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| per-cell SAGE λ2 | 0.0020 / 0.0006 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| credit-type | 0.0002 / 0.0010 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

**Baseline b, m_c = 0.05.** Each cell: worst false accept / representative Useful power.

| Gate | n_c 70 | n_c 150 | n_c 300 | n_c 600 |
| --- | --- | --- | --- | --- |
| C4 | 0.9990 / 0.9892 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| C5 | 0.9971 / 0.9892 | 0.9886 / 1.0000 | 0.9046 / 1.0000 | 0.4568 / 1.0000 |
| C6-IUT (Tango) | 0.0448 / 0.7525 | 0.0544 / 0.9935 | 0.0515 / 1.0000 | 0.0497 / 1.0000 |
| SAGE λ1 | 0.9817 / 0.9790 | 0.9986 / 0.9985 | 1.0000 / 0.9999 | 1.0000 / 1.0000 |
| SAGE λ2 | 0.9333 / 0.8103 | 0.9982 / 0.9874 | 1.0000 / 0.9999 | 1.0000 / 1.0000 |
| per-cell SAGE λ1 | 0.1713 / 0.0527 | 0.2162 / 0.0227 | 0.2678 / 0.0062 | 0.2904 / 0.0006 |
| per-cell SAGE λ2 | 0.1710 / 0.0525 | 0.2162 / 0.0227 | 0.2678 / 0.0062 | 0.2904 / 0.0006 |
| credit-type | 0.0005 / 0.0007 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

**Baseline b, m_c = 0.10.** Each cell: worst false accept / representative Useful power.

| Gate | n_c 70 | n_c 150 | n_c 300 | n_c 600 |
| --- | --- | --- | --- | --- |
| C4 | 0.9987 / 0.9892 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| C5 | 0.8863 / 0.9892 | 0.4855 / 1.0000 | 0.1108 / 1.0000 | 0.0055 / 1.0000 |
| C6-IUT (Tango) | 0.0508 / 0.9651 | 0.0452 / 1.0000 | 0.0500 / 1.0000 | 0.0505 / 1.0000 |
| SAGE λ1 | 0.9132 / 0.9790 | 0.9751 / 0.9985 | 0.9973 / 0.9999 | 1.0000 / 1.0000 |
| SAGE λ2 | 0.8484 / 0.8103 | 0.9743 / 0.9874 | 0.9973 / 0.9999 | 1.0000 / 1.0000 |
| per-cell SAGE λ1 | 0.0312 / 0.0527 | 0.0083 / 0.0227 | 0.0009 / 0.0062 | 0.0000 / 0.0006 |
| per-cell SAGE λ2 | 0.0312 / 0.0525 | 0.0083 / 0.0227 | 0.0009 / 0.0062 | 0.0000 / 0.0006 |
| credit-type | 0.0001 / 0.0007 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

## What these results support (Exploratory)

- **Large local losses (the Severe candidates of Ticket 009b).** Of the 14 Severe candidates in the Ticket 009b pool, C4 accepted 11, C5 2, SAGE λ1 1 and SAGE λ2 1. C5 and SAGE (our implementation) both filtered most of the Severe candidates that C4 accepted. This gives no evidence that C5 is better than SAGE.
- **Margin-level local losses (simulation, one cell at −m_c).** At m_c = 0.05, C4, SAGE and C5 did not control false acceptance at any n_c (baseline a, n_c 150: C4 1.0000, SAGE λ1 0.9987, C5 0.9349), and only C6-IUT controlled it at the α level (0.0442). C5 is a filter for large losses; it does not control losses of the size of the margin. The cost of C6-IUT is the sample per cell: in baseline a it met the Go rule only from n_c 150. At m_c = 0.10 and n_c 300, C5 also met the Go rule in baseline a (0.0112).
- **Per-cell SAGE and the credit-type gate rejected practically everything,** in the simulation (largest representative Useful power: per-cell SAGE 0.0527, credit-type 0.0010) and in the replay (0 and 0 accepted).
- **C6-IUT on the Ticket 009b pool.** At 70 per cell, C6-IUT (Tango, m_c 0.05) was very conservative (1 accepted, 1 of 7 Clean Useful). This agrees with the simulated power at n_c 70 (0.2390 in baseline a). The exact conditional bound accepted more (4) and admitted 1 Severe candidate, which agrees with its coverage shortfall in the simulation.

## Limits of this simulation (Exploratory)

- **Our implementations.** SAGE, the per-cell SAGE cap and the credit-type gate are simplified one-round versions; τ = 0.05 is our value. Their papers' results are not reproduced here.
- **Exact conditional coverage below nominal.** The one-sided bounds should cover the true cell effect with probability 0.95. Over all baselines, n_c, m_c and scenarios, the exact conditional bound covered it in 0.8081 to 0.9929 of the cells, and the Tango bound in 0.9418 to 0.9795.
- **The percentile bootstrap demonstration gave no information.** In its one condition (baseline a, no effect in any cell, n_c 70, m_c 0.05), C6-IUT accepted 0.0034 of candidates with the percentile bound, 0.0026 with Tango and 0.0031 with the exact conditional bound, and C4 alone 0.0044; the condition does not separate the bounds.
- **The realized L1–L2 coupling was small.** The copula correlation was 0.5, but the binary (phi) correlation it produced in the simulated calibration runs was 0.051 to 0.061 in traffic_spike and 0.002 to 0.012 in dependency_failure (bad_deployment: undefined, L1 succeeds in every case). The coupled rows below hold under this small correlation. For description only, the Ticket 009b calibration records give 0.028 (traffic_spike) and -0.003 (dependency_failure).
- **Oracle label uncertainty at 50 incidents per cell.** The share of candidates whose simulated oracle interval crossed ±ε was, at n_c 70 and m_c 0.05: baseline a, 0.0394 (no effect), 0.8402 (representative Useful), 0.5863 (empirical); baseline b, 0.2959, 0.5540 and 0.5790.
- **Gain values come only from the correction file.** The gain per proposal in `results/ticket-013/sim.json` is invalid: the pool sums were truncated to integers before the MC SE. The values below are read from `results/ticket-013/sim-gain-correction.json`, which recomputes them from the same checkpoints; gain is not part of the Go rule.

| Go combination | Gain per proposal, empirical ± MC SE | Gain per proposal, representative Useful ± MC SE |
| --- | --- | --- |
| a, C6-IUT, n_c 150, m_c 0.05 | 0.0080 ± 0.0001 | 0.0247 ± 0.0002 |
| a, C6-IUT, n_c 150, m_c 0.10 | 0.0152 ± 0.0001 | 0.0303 ± 0.0001 |
| a, C5, n_c 300, m_c 0.10 | 0.0156 ± 0.0001 | 0.0333 ± 0.0000 |
| a, C6-IUT, n_c 300, m_c 0.10 | 0.0165 ± 0.0001 | 0.0333 ± 0.0000 |
| b, C6-IUT, n_c 70, m_c 0.05 | 0.0001 ± 0.0000 | 0.0627 ± 0.0004 |
| b, C6-IUT, n_c 150, m_c 0.10 | 0.0024 ± 0.0000 | 0.0833 ± 0.0000 |

## Dependence settings (Exploratory)

The Go rule uses the main setting only. Each cell: worst false accept / representative Useful power.

| Go combination | Main | S0 drawn per candidate | L1–L2 coupling |
| --- | --- | --- | --- |
| a, C6-IUT, n_c 150, m_c 0.05 | 0.0442 / 0.7419 | 0.0440 / 0.7402 | 0.0452 / 0.7241 |
| a, C6-IUT, n_c 150, m_c 0.10 | 0.0410 / 0.9081 | 0.0425 / 0.9081 | 0.0410 / 0.8908 |
| a, C5, n_c 300, m_c 0.10 | 0.0112 / 0.9978 | 0.0117 / 0.9978 | 0.0126 / 0.9952 |
| a, C6-IUT, n_c 300, m_c 0.10 | 0.0474 / 0.9978 | 0.0474 / 0.9978 | 0.0474 / 0.9953 |
| b, C6-IUT, n_c 70, m_c 0.05 | 0.0448 / 0.7525 | 0.0431 / 0.7542 | 0.0438 / 0.7472 |
| b, C6-IUT, n_c 150, m_c 0.10 | 0.0452 / 1.0000 | 0.0459 / 0.9999 | 0.0453 / 1.0000 |

## Replay on the Ticket 009b pool (Exploratory)

The frozen parameters applied to the Ticket 009b Dev-B pairs (40 candidates, CD); labels from the committed Ticket 009b analysis. C4 and C5 recomputed from the same records matched the committed decisions. The pool holds 14 Severe, 13 Useful and 7 Clean Useful candidates. The C6-IUT replay ran after m_c = 0.05 was fixed for the confirmatory study and changes neither m_c nor n_c.

| Gate | Accepted | Severe | Useful | Clean Useful |
| --- | --- | --- | --- | --- |
| C4 (committed decision) | 22 | 11 | 13 | 7 |
| C5 (committed decision) | 13 | 2 | 8 | 7 |
| SAGE λ1 | 9 | 1 | 6 | 5 |
| SAGE λ2 | 7 | 1 | 5 | 4 |
| per-cell SAGE λ1 | 0 | 0 | 0 | 0 |
| per-cell SAGE λ2 | 0 | 0 | 0 | 0 |
| credit-type | 0 | 0 | 0 | 0 |
| C6-IUT, Tango, m_c 0.05 | 1 | 0 | 1 | 1 |
| C6-IUT, exact conditional, m_c 0.05 | 4 | 1 | 3 | 2 |
| C6-IUT, Tango, m_c 0.10 | 6 | 1 | 3 | 2 |
| C6-IUT, exact conditional, m_c 0.10 | 7 | 1 | 4 | 3 |

**One representative per idea group** (29 candidates; 10 Severe, 6 Useful, 4 Clean Useful).

| Gate | Accepted | Severe | Useful | Clean Useful |
| --- | --- | --- | --- | --- |
| C4 (committed decision) | 14 | 7 | 6 | 4 |
| C5 (committed decision) | 8 | 1 | 4 | 4 |
| SAGE λ1 | 5 | 0 | 3 | 3 |
| SAGE λ2 | 4 | 0 | 2 | 2 |
| per-cell SAGE λ1 | 0 | 0 | 0 | 0 |
| per-cell SAGE λ2 | 0 | 0 | 0 | 0 |
| credit-type | 0 | 0 | 0 | 0 |
| C6-IUT, Tango, m_c 0.05 | 1 | 0 | 1 | 1 |
| C6-IUT, exact conditional, m_c 0.05 | 3 | 0 | 2 | 2 |
| C6-IUT, Tango, m_c 0.10 | 4 | 0 | 2 | 2 |
| C6-IUT, exact conditional, m_c 0.10 | 5 | 0 | 3 | 3 |
