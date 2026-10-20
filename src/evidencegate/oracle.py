"""Oracle 채점, τ_cell, 불확정, 하위집합(E), 적합성(F) (7.8 D·E·F, Ticket 005 §8-13~18·22·23).

- Oracle Δ = 후보(반복 0) - S0 반복 0~4 다수결, 전체(300)와 칸별(50). 구간은 칸 층화 paired
  bootstrap 95% 양측(2.5·97.5 백분위수, 2,000회, seed 0).
- τ_cell = max(ε, 칸 잡음 q95). 칸 잡음: S0 Oracle 반복 r마다 나머지 4회 평균을 기준값으로 한 칸의
  |평균(x_r - 기준값)|, 다섯 값의 95번째 백분위수(선형 보간). metric별·칸별(§8-22).
- 불확정(§8-13): 판정 조건을 구간 하한·상한에 각각 적용해 결과가 다르면 None. 아니면 공통 결과.
  지표는 경계 하나짜리 조건에 적용한다(§8-15: Useful +ε, Harmful -ε). 하위집합과 보조 지표는
  down / none / up 세 갈래 분류에 적용한다. 그러면 "구간이 ±ε 중 하나라도 가로지르면 불확정"과
  같다(§8-17·18). neutral 조건(|Δ| ≤ ε)에 따로 적용하면 neutral 띠를 통째로 덮는 구간을 확정으로
  잘못 판정하므로 그렇게 하지 않는다.
- 칸 단위 조건(Local Harm, E의 칸 조건, neutral의 큰 칸, F)은 점추정으로 판정한다(§8-14).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from evidencegate.eval_constants import (
    EPSILON,
    FITNESS_SIZE,
    N_BOOT,
    ORACLE_CI_QS,
    S0_ORACLE_REPEATS,
    SEED,
    TAU_Q,
)
from evidencegate.eval_data import CELLS, ORACLE_SPLIT, CaseId, Cell, PairedTable, cell_of
from evidencegate.exact import mean_fraction, percentile_exact
from evidencegate.gates import bootstrap_quantiles

PRIOR_LABELS = ("positive", "negative", "mixed", "neutral")


@dataclass(frozen=True)
class OracleEstimate:
    metric: str
    delta: Fraction
    interval: tuple[Fraction, Fraction]
    cell_deltas: Mapping[Cell, Fraction]
    tau: Mapping[Cell, Fraction]


@dataclass(frozen=True)
class SubsetResult:
    label: str  # positive / negative / mixed / neutral / uncertain
    big_cells: tuple[Cell, ...] = ()  # neutral 안에서 |Δ_cell| > τ_cell인 칸


@dataclass(frozen=True)
class FitnessResult:
    fit: bool
    sign_met: bool
    size_met: bool
    sign_hits: tuple[tuple[str, str], ...]  # (통제 후보, family)
    size_hits: tuple[tuple[str, str], ...]


def estimate(
    table: PairedTable,
    tau: Mapping[Cell, Fraction],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> OracleEstimate:
    if table.split != ORACLE_SPLIT:
        raise ValueError(f"oracle scoring uses {ORACLE_SPLIT}, got {table.split}")
    low, high = bootstrap_quantiles(table, ORACLE_CI_QS, n_boot=n_boot, seed=seed)
    groups = table.cell_indices()
    cells = {
        cell: mean_fraction([table.deltas[i] for i in g])
        for cell, g in zip(CELLS, groups, strict=True)
    }
    return OracleEstimate(
        metric=table.metric,
        delta=mean_fraction(table.deltas),
        interval=(low, high),
        cell_deltas=cells,
        tau=dict(tau),
    )


def tau_cells(s0_repeats: Mapping[int, Mapping[CaseId, int]]) -> dict[Cell, Fraction]:
    """S0 Oracle 반복 5회(반복 → 케이스 → 0/1)에서 칸별 τ_cell."""
    if sorted(s0_repeats) != list(S0_ORACLE_REPEATS):
        raise ValueError(f"tau needs {len(S0_ORACLE_REPEATS)} repeats {S0_ORACLE_REPEATS}")
    repeats = sorted(s0_repeats)
    cases = list(s0_repeats[repeats[0]])
    tau = {}
    for cell in CELLS:
        in_cell = [c for c in cases if cell_of(c) == cell]
        noise = []
        for r in repeats:
            others = [o for o in repeats if o != r]
            diffs = [
                s0_repeats[r][c] - Fraction(sum(s0_repeats[o][c] for o in others), len(others))
                for c in in_cell
            ]
            noise.append(abs(mean_fraction(diffs)))
        tau[cell] = max(EPSILON, percentile_exact(noise, TAU_Q))
    return tau


def useful(e: OracleEstimate) -> bool | None:
    """Δ > ε. 경계 +ε(§8-15)."""
    return _judge(e.interval, lambda x: x > EPSILON)


def harmful(e: OracleEstimate) -> bool | None:
    """Δ < -ε. 경계 -ε(§8-15)."""
    return _judge(e.interval, lambda x: x < -EPSILON)


def unsupported(e: OracleEstimate) -> bool:
    """구간 하한 <= 0. 불확정 제외 없음(7.8 D의 예외)."""
    return e.interval[0] <= 0


def classify_effect(e: OracleEstimate) -> str | None:
    """down(Δ < -ε) / none(|Δ| <= ε) / up(Δ > ε). 구간이 ±ε를 가로지르면 None(§8-17·18)."""
    return _judge(e.interval, _effect_class)


def local_harm(e: OracleEstimate) -> bool:
    """어떤 칸에서 Δ_cell < -τ_cell(점추정, §8-14)."""
    return any(e.cell_deltas[c] < -e.tau[c] for c in CELLS)


def subset_label(e: OracleEstimate) -> SubsetResult:
    effect = classify_effect(e)
    if effect is None:
        return SubsetResult("uncertain")
    if effect == "down":
        return SubsetResult("negative")
    if effect == "up":
        return SubsetResult("mixed" if local_harm(e) else "positive")
    big = tuple(c for c in CELLS if abs(e.cell_deltas[c]) > e.tau[c])
    return SubsetResult("neutral", big_cells=big)


def check_control_label(prior: str, e: OracleEstimate) -> str:
    """as_expected / unexpected("예상과 다름") / uncertain(§8-17)."""
    if prior not in PRIOR_LABELS:
        raise ValueError(f"unknown prior label {prior!r}; expected one of {PRIOR_LABELS}")
    label = subset_label(e).label
    if label == "uncertain":
        return "uncertain"
    return "as_expected" if label == prior else "unexpected"


def fitness(controls: Mapping[str, OracleEstimate]) -> FitnessResult:
    """한 metric에서 통제 후보들로 F를 본다. 부호 기준 또는 크기 기준 하나 이상이면 적합(§8-23)."""
    families = sorted({cell[0] for cell in CELLS})
    sign_hits = []
    size_hits = []
    for name, e in sorted(controls.items()):
        for family in families:
            l1, l2 = (family, "L1"), (family, "L2")
            d1, d2 = e.cell_deltas[l1], e.cell_deltas[l2]
            up = d1 > e.tau[l1] or d2 > e.tau[l2]
            down = d1 < -e.tau[l1] or d2 < -e.tau[l2]
            if up and down:
                sign_hits.append((name, family))
            if abs(d1 - d2) >= FITNESS_SIZE:
                size_hits.append((name, family))
    return FitnessResult(
        fit=bool(sign_hits or size_hits),
        sign_met=bool(sign_hits),
        size_met=bool(size_hits),
        sign_hits=tuple(sign_hits),
        size_hits=tuple(size_hits),
    )


def _effect_class(x: Fraction) -> str:
    if x > EPSILON:
        return "up"
    if x < -EPSILON:
        return "down"
    return "none"


def _judge[T](interval: tuple[Fraction, Fraction], decide: Callable[[Fraction], T]) -> T | None:
    low, high = interval
    at_low, at_high = decide(low), decide(high)
    return at_low if at_low == at_high else None
