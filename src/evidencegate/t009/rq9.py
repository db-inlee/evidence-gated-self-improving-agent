"""Ticket 009 정의의 정확한 구현(§2.2 h, §2.3 Severe Local Harm, §2.4 RQ9-1·2·3).

- h(n_c) = -q. q는 m 분포의 5번째 백분위수(선형 보간, 유리수). 재표본 2,000회마다 `random.Random(0)`
  흐름 하나(호출마다 새로 시작)로 (1) 보정 S0 반복에서 순서 있는 쌍 (r, s)를 고르고(`rng.sample`,
  v1 δ(N)과 같음, 여섯 칸 공통) (2) 칸(`CELLS` 순서)마다 보정 케이스에서 n_c개를 복원 추출하고
  (`rng.choices`) (3) Δ_c = Σ(x_r - x_s) / n_c, m = min_c Δ_c. 달성 거짓 기각률 P(m < -h)도 낸다.
- 칸 조건: 모든 칸에서 Δ_cell >= -h면 통과(비엄격), 하나라도 Δ_cell < -h면 기각.
- Severe Local Harm: 어떤 칸에서 oracle Δ_cell의 95% 구간 상한 < -τ_cell이면 참, 그 밖은 거짓.
  구간은 칸 안 paired bootstrap이다. 색인은 `stratified_bootstrap_indices(칸 크기들, 2000, 0)`의 그
  칸 블록(공통 난수)이고, 상한은 39/40 백분위수(선형 보간)다. 불확정 범주는 없다.
- RQ9-1 단측 Fisher: 초기하 상단 꼬리, `math.comb`와 `Fraction`. 판정은 엄격 부등호.
"""

import random
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import comb
from typing import Any

import numpy as np

from evidencegate.eval_constants import N_BOOT, ORACLE_CI_QS, S0_ORACLE_REPEATS, SEED
from evidencegate.eval_data import CELLS, CaseId, Cell, PairedTable, cell_of
from evidencegate.exact import clopper_pearson, percentile_exact, stratified_bootstrap_indices
from evidencegate.gates import _percentile_of_totals

H_Q = Fraction(1, 20)  # §2.2: m 분포의 5번째 백분위수
FISHER_ALPHA = Fraction(1, 20)  # §2.4 RQ9-1: p < 0.05
USEFUL_LOSS_MAX = Fraction(1, 5)  # §2.4 RQ9-2: Useful 손실 비율 <= 1/5
RQ9_3_LOWER = 0.10  # §2.4 RQ9-3: Clopper-Pearson 95% 하한 >= 0.10
SEVERE_Q = ORACLE_CI_QS[1]  # 95% 양측 구간의 상한(39/40)


def cell_threshold_h(
    outcomes: Mapping[int, Mapping[CaseId, int]],
    n_c: int,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    cells: Sequence[Cell] = CELLS,
) -> tuple[Fraction, Fraction]:
    """보정 S0 반복(반복 -> 케이스 -> 0/1)에서 (h(n_c), 달성 거짓 기각률)."""
    if sorted(outcomes) != list(S0_ORACLE_REPEATS):
        raise ValueError(f"h needs S0 repeats {S0_ORACLE_REPEATS}")
    if n_c < 1:
        raise ValueError(f"n_c must be >= 1, got {n_c}")
    repeats = sorted(outcomes)
    by_cell = {
        cell: sorted(c for c in outcomes[repeats[0]] if cell_of(c) == cell) for cell in cells
    }
    if any(not cases for cases in by_cell.values()):
        raise ValueError("every cell needs calibration cases")
    rng = random.Random(seed)
    ms = []
    for _ in range(n_boot):
        r, s = rng.sample(repeats, 2)
        deltas = []
        for cell in cells:
            sample = rng.choices(by_cell[cell], k=n_c)
            deltas.append(Fraction(sum(outcomes[r][c] - outcomes[s][c] for c in sample), n_c))
        ms.append(min(deltas))
    q = percentile_exact(ms, H_Q)
    return -q, Fraction(sum(1 for m in ms if m < q), n_boot)


def cell_condition(cell_deltas: Sequence[Fraction], h: Fraction) -> bool:
    """모든 칸에서 Δ_cell >= -h(비엄격 통과)."""
    return all(d >= -h for d in cell_deltas)


def severe_local_harm(
    table: PairedTable, tau: Mapping[Cell, Fraction], n_boot: int = N_BOOT, seed: int = SEED
) -> bool:
    groups = table.cell_indices()
    deltas = np.asarray(table.deltas, dtype=np.int64)
    blocks = stratified_bootstrap_indices([len(g) for g in groups], n_boot=n_boot, seed=seed)
    for cell, group, block in zip(CELLS, groups, blocks, strict=True):
        totals = sorted(
            int(t) for t in deltas[np.asarray(group, dtype=np.int64)][block].sum(axis=1)
        )
        upper = _percentile_of_totals(totals, SEVERE_Q) / len(group)
        if upper < -tau[cell]:
            return True
    return False


def fisher_one_sided(*, a: int, f: int, k: int, n: int) -> Fraction:
    """P(X >= a), X ~ 초기하(전체 n, 성공 k, 추출 f)."""
    top = min(f, k)
    return Fraction(sum(comb(k, x) * comb(n - k, f - x) for x in range(a, top + 1)), comb(n, f))


def rq9_1(rows: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """rows: C4 채택 proposer 후보마다 (C5 채택, Severe)."""
    n = len(rows)
    f = sum(1 for c5, _ in rows if not c5)
    k = sum(1 for _, severe in rows if severe)
    a = sum(1 for c5, severe in rows if not c5 and severe)
    out: dict[str, Any] = {"a": a, "f": f, "k": k, "n": n, "p": None}
    if n < 2 or f in (0, n) or k in (0, n):
        return {**out, "verdict": "undecidable"}
    p = fisher_one_sided(a=a, f=f, k=k, n=n)
    higher = Fraction(a, f) > Fraction(k - a, n - f)
    verdict = (
        "not_supported" if not higher else "supported" if p < FISHER_ALPHA else "direction_only"
    )
    return {**out, "p": p, "verdict": verdict}


def rq9_2(useful_c4: int, useful_rejected: int, *, any_filtered: bool) -> dict[str, Any]:
    """useful_c4: C4 채택 Useful 수, useful_rejected: 그중 C5가 기각한 수(u)."""
    out: dict[str, Any] = {"useful_c4": useful_c4, "u": useful_rejected}
    if useful_c4 == 0 or not any_filtered:
        return {**out, "verdict": "undecidable"}
    ok = useful_rejected == 0 or Fraction(useful_rejected, useful_c4) <= USEFUL_LOSS_MAX
    return {**out, "verdict": "supported" if ok else "not_supported"}


def rq9_3(*, severe_accepted: int, accepted: int) -> dict[str, Any]:
    interval = clopper_pearson(severe_accepted, accepted)
    out: dict[str, Any] = {"k": severe_accepted, "n": accepted, "interval": interval}
    if interval is None:
        return {**out, "verdict": "undecidable"}
    return {**out, "verdict": "supported" if interval[0] >= RQ9_3_LOWER else "not_supported"}
