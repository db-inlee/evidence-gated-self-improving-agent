"""게이트 C1~C4 (설계 문서 7.8 B, Ticket 005 §8-1·4·7·8·29). CD와 ES를 각각 판정한다.

| 게이트 | 데이터 | 채택 규칙 |
| C1 | Dev-A | 후보 평균 > S0 평균(같으면 기각) |
| C2 | Dev-B | 평균 Δ >= 0 |
| C3 | Dev-B | 평균 Δ > δ(N) (CD 보조, ES도 계산해 보고) |
| C4 | Dev-B | 칸 층화 bootstrap(2,000회, seed 0) 평균 Δ의 5번째 백분위수 > 0 그리고 평균 Δ > δ(N) |

- 모든 비교는 유리수로 한다(§8-4). δ(N)은 `exact.null_diff_quantile_exact`의 값을 받는다(§8-29).
- C4의 bootstrap은 칸마다 그 표의 칸 크기만큼 복원 추출한다(§8-7). 재표본 평균은 모두 분모가
  N이라, 정수 합계의 백분위수를 구해 N으로 나눈다(유리수 백분위수와 같은 값).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from evidencegate.eval_constants import C4_Q, N_BOOT, SEED
from evidencegate.eval_data import PairedTable
from evidencegate.exact import mean_fraction, stratified_bootstrap_indices


@dataclass(frozen=True)
class GateResult:
    gate: str  # "C1" ~ "C4"
    metric: str
    accepted: bool
    n: int  # 케이스 수
    mean_delta: Fraction
    threshold: Fraction | None = None  # C3·C4의 δ(N)
    lower: Fraction | None = None  # C4의 bootstrap 하한


def gate_c1(dev_a: PairedTable) -> GateResult:
    _need_split(dev_a, "dev_a", "C1")
    accepted = mean_fraction(dev_a.candidate) > mean_fraction(dev_a.baseline)
    return GateResult("C1", dev_a.metric, accepted, len(dev_a.cases), mean_fraction(dev_a.deltas))


def gate_c2(dev_b: PairedTable) -> GateResult:
    _need_split(dev_b, "dev_b", "C2")
    mean = mean_fraction(dev_b.deltas)
    return GateResult("C2", dev_b.metric, mean >= 0, len(dev_b.cases), mean)


def gate_c3(dev_b: PairedTable, delta: Fraction) -> GateResult:
    _need_split(dev_b, "dev_b", "C3")
    _need_exact(delta)
    mean = mean_fraction(dev_b.deltas)
    return GateResult("C3", dev_b.metric, mean > delta, len(dev_b.cases), mean, threshold=delta)


def gate_c4(
    dev_b: PairedTable, delta: Fraction, n_boot: int = N_BOOT, seed: int = SEED
) -> GateResult:
    _need_split(dev_b, "dev_b", "C4")
    _need_exact(delta)
    mean = mean_fraction(dev_b.deltas)
    lower = bootstrap_lower(dev_b, n_boot=n_boot, seed=seed)
    accepted = lower > 0 and mean > delta
    return GateResult(
        "C4", dev_b.metric, accepted, len(dev_b.cases), mean, threshold=delta, lower=lower
    )


def decide_gates(dev_a: PairedTable, dev_b: PairedTable, delta: Fraction) -> tuple[GateResult, ...]:
    """한 후보·한 metric의 C1~C4. δ는 그 metric의 δ(N = Dev-B 크기)."""
    if dev_a.metric != dev_b.metric:
        raise ValueError(f"metric differs: dev_a {dev_a.metric}, dev_b {dev_b.metric}")
    return gate_c1(dev_a), gate_c2(dev_b), gate_c3(dev_b, delta), gate_c4(dev_b, delta)


def bootstrap_lower(
    table: PairedTable, n_boot: int = N_BOOT, seed: int = SEED, q: Fraction = C4_Q
) -> Fraction:
    """칸 층화 paired bootstrap 평균 Δ의 q 백분위수(선형 보간, 유리수)."""
    groups = table.cell_indices()
    if any(not g for g in groups):
        raise ValueError("every cell needs at least one case for the stratified bootstrap")
    deltas = np.asarray(table.deltas, dtype=np.int64)
    blocks = stratified_bootstrap_indices([len(g) for g in groups], n_boot=n_boot, seed=seed)
    totals = np.zeros(n_boot, dtype=np.int64)
    for group, block in zip(groups, blocks, strict=True):
        totals += deltas[np.asarray(group, dtype=np.int64)][block].sum(axis=1)
    return _percentile_of_totals(sorted(int(t) for t in totals), q) / len(table.deltas)


def _percentile_of_totals(ordered: Sequence[int], q: Fraction) -> Fraction:
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _need_split(table: PairedTable, split: str, gate: str) -> None:
    if table.split != split:
        raise ValueError(f"{gate} uses {split}, got {table.split}")


def _need_exact(delta: object) -> None:
    if isinstance(delta, bool) or not isinstance(delta, Fraction | int):
        raise TypeError(f"delta must be a Fraction (exact, §8-29), got {type(delta).__name__}")
