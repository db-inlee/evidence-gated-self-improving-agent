"""지표 6개와 보조 지표 (7.8 D, Ticket 005 §8-13~16·18·19·21).

게이트마다, 한 후보 집합(proposer 12개 또는 통제 7개, §8-19)에서 계산한다. 게이트 metric과 oracle
metric은 보통 같고, RQ1에서는 다르다(CD로 채택, ES로 해로움).

| 지표 | 정의 | 불확정 제외 |
| Harmful Acceptance Rate | P(Δ < -ε | 채택) | 경계 -ε |
| Unsupported Acceptance Rate | P(구간 하한 <= 0 | 채택) | 없음(예외) |
| Useful Acceptance Precision | P(Δ > ε | 채택) | 경계 +ε |
| Expected Oracle Gain | 채택 후보의 평균 Δ(구간 없음, §8-16) | 없음 |
| False Rejection Rate | P(기각 | Δ > ε) | 분모 조건 +ε |
| Local Harm Acceptance Rate | P(어떤 칸 Δ_cell < -τ_cell | 채택) | 없음(§8-14) |

비율은 분자/분모, 제외 수, Clopper-Pearson 95%(분모 0이면 판정 불가)로 낸다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from evidencegate.exact import clopper_pearson
from evidencegate.metrics import METRICS
from evidencegate.oracle import (
    OracleEstimate,
    classify_effect,
    harmful,
    local_harm,
    unsupported,
    useful,
)

GATE_NAMES = ("C1", "C2", "C3", "C4")
EFFECT_KEYS = {"correct_disposition": "cd", "evidence_safe": "es"}  # 증거 카드 expected_effect의 키


@dataclass(frozen=True)
class CandidateScore:
    name: str
    accepted: Mapping[tuple[str, str], bool]  # (게이트, metric) → 채택
    oracle: Mapping[str, OracleEstimate]  # metric → Oracle 추정
    expected_effect: Mapping[str, str] | None = None  # 증거 카드("cd"/"es" → up/down/none)


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int  # 분모(불확정 제외 뒤)
    excluded: int  # 불확정으로 뺀 수

    @property
    def value(self) -> Fraction | None:
        return Fraction(self.successes, self.n) if self.n else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return clopper_pearson(self.successes, self.n)


@dataclass(frozen=True)
class Gain:
    n: int
    mean: Fraction | None


@dataclass(frozen=True)
class GateMetrics:
    gate: str
    gate_metric: str
    oracle_metric: str
    accepted: tuple[str, ...]
    harmful: Rate
    unsupported: Rate
    useful: Rate
    gain: Gain
    false_rejection: Rate
    local_harm: Rate


def gate_metrics(
    candidates: Sequence[CandidateScore],
    gate: str,
    gate_metric: str,
    oracle_metric: str | None = None,
) -> GateMetrics:
    if gate not in GATE_NAMES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {GATE_NAMES}")
    oracle_metric = oracle_metric or gate_metric
    for m in (gate_metric, oracle_metric):
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}; expected one of {METRICS}")
    for c in candidates:
        if (gate, gate_metric) not in c.accepted:
            raise ValueError(f"{c.name} has no decision for {gate} on {gate_metric}")

    accepted = [c for c in candidates if c.accepted[(gate, gate_metric)]]
    estimates = [c.oracle[oracle_metric] for c in accepted]
    deltas = [e.delta for e in estimates]
    useful_all = [(c, useful(c.oracle[oracle_metric])) for c in candidates]
    return GateMetrics(
        gate=gate,
        gate_metric=gate_metric,
        oracle_metric=oracle_metric,
        accepted=tuple(c.name for c in accepted),
        harmful=_rate([harmful(e) for e in estimates]),
        unsupported=_rate([unsupported(e) for e in estimates]),
        useful=_rate([useful(e) for e in estimates]),
        gain=Gain(len(deltas), sum(deltas, Fraction(0)) / len(deltas) if deltas else None),
        # 분모: Δ > ε인 후보(불확정은 제외 수로), 분자: 그중 기각된 것
        false_rejection=_rate(
            [
                None if is_useful is None else not c.accepted[(gate, gate_metric)]
                for c, is_useful in useful_all
                if is_useful is not False
            ]
        ),
        local_harm=_rate([local_harm(e) for e in estimates]),
    )


def effect_agreement(candidates: Sequence[CandidateScore], metric: str) -> Rate:
    """증거 카드의 예상 방향과 oracle 방향(down / none / up)의 일치율. 카드 없는 후보는 대상이
    아니고, 방향이 불확정인 후보는 빼고 센다(§8-18)."""
    key = EFFECT_KEYS[metric]
    judged = []
    for c in candidates:
        if c.expected_effect is None:
            continue
        actual = classify_effect(c.oracle[metric])
        judged.append(None if actual is None else actual == c.expected_effect[key])
    return _rate(judged)


def _rate(judgments: Sequence[bool | None]) -> Rate:
    return Rate(
        successes=sum(1 for j in judgments if j is True),
        n=sum(1 for j in judgments if j is not None),
        excluded=sum(1 for j in judgments if j is None),
    )
