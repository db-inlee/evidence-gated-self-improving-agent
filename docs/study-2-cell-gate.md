# Cell-level gate: detailed tables

These tables belong to the cell-gate study in the [README](../README.md#cell-level-gate-confirmatory-study). Every number is generated from `results/ticket-009b/analysis.json`, `results/ticket-009/thresholds.json` and the design constants; none was typed by hand. All rows are pre-registered; rows marked descriptive carry no verdict of the study.

## Gate metrics on ES

Counts over the candidates each gate accepted on ES, scored on the Oracle split (proposer candidates).

| Gate | Accepted | Severe | Local harm | Harmful | Unsupported | Useful | Clean Useful | Gain | False rejection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | 34 | 0/34 | 1/34 | 0/34 | 9/34 | 14/27 | 14/34 | 0.120 | 0/14 |
| C4 | 27 | 0/27 | 1/27 | 0/27 | 2/27 | 14/20 | 14/27 | 0.151 | 0/14 |
| C5 | 27 | 0/27 | 1/27 | 0/27 | 2/27 | 14/20 | 14/27 | 0.151 | 0/14 |
| C5′ | 34 | 0/34 | 1/34 | 0/34 | 9/34 | 14/27 | 14/34 | 0.120 | 0/14 |
| C4\* | 27 | 0/27 | 1/27 | 0/27 | 2/27 | 14/20 | 14/27 | 0.151 | 0/14 |

On ES, C4 and C5 accepted the same candidates, so the cell condition removed nothing; no gate accepted a candidate with Severe Local Harm on ES.

## Budget curve (descriptive, CD)

Each row recomputes the gates on stratified Dev-B subsamples of N = 24, 48, 96, 204 incidents, the same sizes as in the mean-gate study (1000 resamples each), and on the full Dev-B (420, one evaluation). Values are means over resamples: the share of proposer candidates accepted, the number of Severe candidates accepted, and the differences C4 minus C5.

| Dev-B size | Accepted C2 | Accepted C4 | Accepted C5 | Severe in C4 | Severe in C5 | Severe, C4 − C5 | Useful, C4 − C5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smallest subsample | 0.769 | 0.197 | 0.185 | 3.840 | 3.363 | 0.477 | 0.002 |
| second subsample | 0.737 | 0.291 | 0.224 | 5.611 | 2.968 | 2.643 | 0.038 |
| third subsample | 0.704 | 0.415 | 0.275 | 8.126 | 2.686 | 5.440 | 0.123 |
| fourth subsample | 0.663 | 0.518 | 0.314 | 10.181 | 2.212 | 7.969 | 0.194 |
| full Dev-B (420) | 0.650 | 0.550 | 0.325 | 11.000 | 2.000 | 9.000 | 0.000 |

## Thresholds h, δ and τ

h(n_c) and δ(N) come from S0-vs-S0 calibration runs (70 incidents per cell, repeats 0, 1, 2, 3, 4); τ_cell is the oracle threshold per cell.

| Metric | h(70) | False rejection of h | δ(420) |
| --- | --- | --- | --- |
| CD | 0.114 | 0.035 | 0.029 |
| ES | 0.029 | 0.043 | 0.007 |

τ_cell on CD by cell: bad_deployment/L1 0.050, bad_deployment/L2 0.050, dependency_failure/L1 0.050, dependency_failure/L2 0.050, traffic_spike/L1 0.050, traffic_spike/L2 0.050.

## Controls (Oracle only)

Hand-written controls with labels fixed before measurement. *Check* compares the oracle effect estimate with the label.

| Control | Label | CD Δ | CD check | ES Δ | ES check |
| --- | --- | --- | --- | --- | --- |
| mixed-scale-on-traffic | mixed | 0.283 | as expected | 0.623 | unexpected |
| neg-rollback-any | negative | -0.327 | as expected | -0.317 | as expected |
| neg-skip-metrics | negative | -0.010 | unexpected | -0.330 | as expected |
| neg-timing-shortcut | negative | -0.123 | as expected | -0.157 | as expected |
| neutral-rephrase | neutral | -0.017 | as expected | -0.003 | as expected |
| pos-timing-only | positive | 0.097 | as expected | 0.023 | unexpected |
| pos-traffic-check | positive | 0.280 | unexpected | 0.623 | as expected |

Fitness of the oracle (pre-registered F) asks whether some control moves the L1 and L2 cells of one family apart, by opposite signs beyond τ or by a large size gap, so that per-cell scoring can matter. CD: yes (sign rule met: no, size rule met: yes); ES: no (sign rule met: no, size rule met: no).

## Null label rates (planning estimate)

For a candidate with no effect, the planning simulation gave a Local Harm label rate of 0.167 and a Severe Local Harm rate of 0.000475. Because τ_cell equals ε, Local Harm is broad; Severe is the primary label.
