"""지표 6개와 보조 지표 (7.8 D, Ticket 005 §8-13~16·18·19·21). 손으로 만든 추정값으로 확인한다."""

from fractions import Fraction

import pytest

from evidencegate.eval_constants import EPSILON
from evidencegate.eval_data import CELLS, Cell
from evidencegate.eval_metrics import (
    CandidateScore,
    Gain,
    Rate,
    effect_agreement,
    gate_metrics,
)
from evidencegate.exact import clopper_pearson
from evidencegate.oracle import OracleEstimate

CD = "correct_disposition"
ES = "evidence_safe"
EPS = EPSILON
TAU = dict.fromkeys(CELLS, EPS)


def oracle(
    delta: Fraction,
    interval: tuple[Fraction, Fraction] | None = None,
    cells: dict[Cell, Fraction] | None = None,
    metric: str = CD,
) -> OracleEstimate:
    width = Fraction(1, 100)
    return OracleEstimate(
        metric=metric,
        delta=delta,
        interval=interval or (delta - width, delta + width),
        cell_deltas=cells or dict.fromkeys(CELLS, delta),
        tau=TAU,
    )


USEFUL = Fraction(3, 20)  # 구간 [0.14, 0.16]: 유용, 해롭지 않음, 하한 > 0
HARMFUL = Fraction(-3, 20)  # 구간 [-0.16, -0.14]: 해로움, 하한 <= 0
NEUTRAL = Fraction(0)  # 구간 [-0.01, 0.01]: 중립, 하한 <= 0
UNCERTAIN_UP = (EPS, Fraction(1, 5))  # 하한 = ε → 유용 불확정


def cand(
    name: str,
    *,
    accepted: dict[str, bool],
    cd: OracleEstimate,
    es: OracleEstimate | None = None,
    effect: dict[str, str] | None = None,
    es_accepted: dict[str, bool] | None = None,
) -> CandidateScore:
    acc = {(g, CD): v for g, v in accepted.items()}
    acc |= {(g, ES): v for g, v in (es_accepted or accepted).items()}
    return CandidateScore(
        name=name,
        accepted=acc,
        oracle={CD: cd, ES: es or oracle(cd.delta, cd.interval, dict(cd.cell_deltas), ES)},
        expected_effect=effect,
    )


GATES = ("C1", "C2", "C3", "C4")


def acc(*gates: str) -> dict[str, bool]:
    return {g: g in gates for g in GATES}


# ---------------------------------------------------------------------------
# Rate와 Clopper-Pearson
# ---------------------------------------------------------------------------


def test_rate_value_and_interval() -> None:
    r = Rate(successes=1, n=4, excluded=2)
    assert r.value == Fraction(1, 4)
    assert r.interval == clopper_pearson(1, 4)


def test_rate_with_zero_denominator_is_undecidable() -> None:
    r = Rate(successes=0, n=0, excluded=1)
    assert r.value is None
    assert r.interval is None


# ---------------------------------------------------------------------------
# 지표 6개 (게이트마다, 같은 metric으로)
# ---------------------------------------------------------------------------


def pool() -> list[CandidateScore]:
    return [
        cand("useful", accepted=acc("C1", "C2"), cd=oracle(USEFUL)),
        cand("harmful", accepted=acc("C1", "C2"), cd=oracle(HARMFUL)),
        cand("neutral", accepted=acc("C2"), cd=oracle(NEUTRAL)),
        cand("uncertain", accepted=acc("C2"), cd=oracle(Fraction(1, 10), UNCERTAIN_UP)),
        cand("rejected-useful", accepted=acc(), cd=oracle(USEFUL)),
        cand(
            "local-harm",
            accepted=acc("C1", "C2"),
            cd=oracle(USEFUL, cells=dict.fromkeys(CELLS, USEFUL) | {CELLS[0]: Fraction(-1, 10)}),
        ),
    ]


def test_accepted_set_by_gate() -> None:
    m = gate_metrics(pool(), "C2", CD)
    assert m.accepted == ("useful", "harmful", "neutral", "uncertain", "local-harm")
    assert gate_metrics(pool(), "C1", CD).accepted == ("useful", "harmful", "local-harm")


def test_harmful_acceptance_rate_excludes_uncertain() -> None:
    """C2 채택 5개. Harmful 경계(-ε)에서 불확정은 없고 해로운 것은 1개."""
    m = gate_metrics(pool(), "C2", CD)
    assert (m.harmful.successes, m.harmful.n, m.harmful.excluded) == (1, 5, 0)


def test_useful_precision_excludes_the_uncertain_candidate() -> None:
    """'uncertain'은 하한 = ε라 Useful이 불확정 → 분모·분자에서 빼고 개수를 따로(§8-15)."""
    m = gate_metrics(pool(), "C2", CD)
    assert (m.useful.successes, m.useful.n, m.useful.excluded) == (2, 4, 1)


def test_unsupported_counts_every_accepted_candidate() -> None:
    """예외: 불확정 제외 없음. 하한 <= 0인 것은 harmful, neutral."""
    m = gate_metrics(pool(), "C2", CD)
    assert (m.unsupported.successes, m.unsupported.n, m.unsupported.excluded) == (2, 5, 0)


def test_expected_oracle_gain_is_the_mean_delta_of_accepted() -> None:
    """§8-16: 평균과 채택 수만, 구간 없음. 불확정도 포함."""
    m = gate_metrics(pool(), "C2", CD)
    expected = (USEFUL + HARMFUL + NEUTRAL + Fraction(1, 10) + USEFUL) / 5
    assert m.gain == Gain(n=5, mean=expected)


def test_false_rejection_rate_uses_certain_useful_candidates() -> None:
    """분모: Δ > ε가 확정인 후보(useful, rejected-useful, local-harm). 불확정 'uncertain'은 뺀다."""
    m = gate_metrics(pool(), "C2", CD)
    assert (m.false_rejection.successes, m.false_rejection.n, m.false_rejection.excluded) == (
        1,
        3,
        1,
    )
    m = gate_metrics(pool(), "C1", CD)
    assert (m.false_rejection.successes, m.false_rejection.n) == (1, 3)


def test_local_harm_acceptance_uses_cells_without_exclusion() -> None:
    m = gate_metrics(pool(), "C2", CD)
    # local-harm(칸 0만 -1/10)과 harmful(모든 칸 -3/20)이 해당한다
    assert (m.local_harm.successes, m.local_harm.n, m.local_harm.excluded) == (2, 5, 0)


def test_gate_that_accepts_nothing_has_zero_denominators() -> None:
    m = gate_metrics(pool(), "C4", CD)
    assert m.accepted == ()
    for rate in (m.harmful, m.unsupported, m.useful, m.local_harm):
        assert rate.n == 0 and rate.value is None
    assert m.gain == Gain(n=0, mean=None)
    assert m.false_rejection.n == 3  # 유용 확정 후보 3개 모두 기각


def test_metrics_use_the_named_metric_for_gate_and_oracle() -> None:
    """RQ1용: CD로 채택한 후보의 ES 해로움(게이트 metric과 oracle metric이 다를 수 있다)."""
    candidates = [
        cand(
            "a",
            accepted=acc("C2"),
            es_accepted=acc(),
            cd=oracle(USEFUL),
            es=oracle(HARMFUL, metric=ES),
        ),
        cand(
            "b",
            accepted=acc("C2"),
            es_accepted=acc("C2"),
            cd=oracle(USEFUL),
            es=oracle(NEUTRAL, metric=ES),
        ),
    ]
    cross = gate_metrics(candidates, "C2", CD, oracle_metric=ES)
    assert (cross.harmful.successes, cross.harmful.n) == (1, 2)
    same = gate_metrics(candidates, "C2", ES)
    assert (same.harmful.successes, same.harmful.n) == (0, 1)
    assert (cross.gate_metric, cross.oracle_metric, same.oracle_metric) == (CD, ES, ES)


def test_unknown_gate_or_metric_is_an_error() -> None:
    with pytest.raises(ValueError, match="gate"):
        gate_metrics(pool(), "C5", CD)
    with pytest.raises(ValueError, match="metric"):
        gate_metrics(pool(), "C2", "cbu")


def test_candidate_missing_a_gate_decision_is_an_error() -> None:
    broken = CandidateScore(name="x", accepted={("C1", CD): True}, oracle={CD: oracle(USEFUL)})
    with pytest.raises(ValueError, match="decision"):
        gate_metrics([broken], "C2", CD)


# ---------------------------------------------------------------------------
# 보조 지표: expected_effect 방향과 oracle 방향의 일치율 (§8-18)
# ---------------------------------------------------------------------------


def test_effect_agreement_excludes_uncertain_and_cards_without_effect() -> None:
    candidates = [
        cand("up-up", accepted=acc(), cd=oracle(USEFUL), effect={"cd": "up", "es": "up"}),
        cand("down-up", accepted=acc(), cd=oracle(USEFUL), effect={"cd": "down", "es": "up"}),
        cand("none-none", accepted=acc(), cd=oracle(NEUTRAL), effect={"cd": "none", "es": "none"}),
        cand("down-down", accepted=acc(), cd=oracle(HARMFUL), effect={"cd": "down", "es": "none"}),
        cand(
            "uncertain",
            accepted=acc(),
            cd=oracle(Fraction(1, 10), UNCERTAIN_UP),
            effect={"cd": "up", "es": "up"},
        ),
        cand("up-down", accepted=acc(), cd=oracle(USEFUL), effect={"cd": "up", "es": "down"}),
        cand("control", accepted=acc(), cd=oracle(USEFUL)),  # 증거 카드 없음
    ]
    cd = effect_agreement(candidates, CD)
    # cd: up-up, none-none, down-down, up-down 일치. down-up 불일치
    assert (cd.successes, cd.n, cd.excluded) == (4, 5, 1)
    es = effect_agreement(candidates, ES)
    # es: up-up, down-up, none-none 일치. down-down(예측 none, 실제 down), up-down 불일치
    assert (es.successes, es.n, es.excluded) == (3, 5, 1)


def test_harmful_rate_excludes_candidates_uncertain_at_minus_epsilon() -> None:
    """Harmful 경계 -ε에서 불확정(상한 = -ε)이면 분모·분자에서 빼고 개수를 따로."""
    candidates = [
        cand("harmful", accepted=acc("C2"), cd=oracle(HARMFUL)),
        cand(
            "borderline", accepted=acc("C2"), cd=oracle(Fraction(-1, 10), (Fraction(-1, 5), -EPS))
        ),
        cand("neutral", accepted=acc("C2"), cd=oracle(NEUTRAL)),
    ]
    m = gate_metrics(candidates, "C2", CD)
    assert (m.harmful.successes, m.harmful.n, m.harmful.excluded) == (1, 2, 1)
    assert m.unsupported.n == 3  # 예외: 제외 없음
