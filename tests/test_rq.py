"""RQ1~RQ3 판정 (7.8 G, Ticket 005 §8-24~28·30~33)."""

from fractions import Fraction

from evidencegate.eval_constants import EPSILON
from evidencegate.eval_data import CELLS
from evidencegate.eval_metrics import CandidateScore, Rate, gate_metrics
from evidencegate.oracle import OracleEstimate
from evidencegate.rq import (
    GROWS,
    NOT_GROWS,
    NOT_SUPPORTED,
    PARTIAL,
    PARTLY_GROWS,
    SUPPORTED,
    TREND_NOT_SUPPORTED,
    TREND_PARTIAL,
    TREND_SUPPORTED,
    UNDECIDABLE,
    rq1,
    rq2_first,
    rq2_second,
    rq3_first,
    rq3_trend,
)

CD = "correct_disposition"
ES = "evidence_safe"


def rate(k: int, n: int) -> Rate:
    return Rate(successes=k, n=n, excluded=0)


EMPTY = rate(0, 0)


# ---------------------------------------------------------------------------
# RQ1 (§8-25): 게이트마다 A = P(ES 해로움 | CD 채택) > B = P(ES 해로움 | ES 채택)
# ---------------------------------------------------------------------------


def test_rq1_supported_when_both_gates_hold() -> None:
    v = rq1({"C2": (rate(2, 4), rate(1, 4)), "C4": (rate(1, 2), rate(0, 2))})
    assert v.verdict == SUPPORTED
    assert [c.holds for c in v.comparisons] == [True, True]


def test_rq1_partial_when_one_gate_holds() -> None:
    v = rq1({"C2": (rate(2, 4), rate(1, 4)), "C4": (rate(0, 2), rate(1, 2))})
    assert v.verdict == PARTIAL


def test_rq1_not_supported_when_no_gate_holds() -> None:
    v = rq1({"C2": (rate(0, 4), rate(1, 4)), "C4": (rate(0, 2), rate(1, 2))})
    assert v.verdict == NOT_SUPPORTED


def test_rq1_tie_is_not_supported() -> None:
    """§8-28: 같으면 지지 안 됨."""
    v = rq1({"C2": (rate(1, 4), rate(2, 8)), "C4": (rate(1, 2), rate(1, 2))})
    assert [c.holds for c in v.comparisons] == [False, False]
    assert v.verdict == NOT_SUPPORTED


def test_rq1_zero_denominator_rises_to_undecidable() -> None:
    """C4가 CD로 아무것도 채택하지 않음 → C4 비교 판정 불가 → RQ1 종합 판정 불가.
    판정 가능한 C2 비교는 따로 보고한다(§8-33)."""
    v = rq1({"C2": (rate(2, 4), rate(1, 4)), "C4": (EMPTY, rate(0, 2))})
    assert v.verdict == UNDECIDABLE
    c2, c4 = v.comparisons
    assert (c2.name, c2.holds) == ("C2", True)
    assert (c4.name, c4.holds, c4.left) == ("C4", None, None)


def oracle(delta: Fraction, metric: str) -> OracleEstimate:
    w = Fraction(1, 100)
    return OracleEstimate(
        metric,
        delta,
        (delta - w, delta + w),
        dict.fromkeys(CELLS, delta),
        dict.fromkeys(CELLS, EPSILON),
    )


def test_rq1_from_gate_metrics_when_c4_accepts_nothing() -> None:
    """지표 계산부터 이어서: C4가 아무것도 채택하지 않으면 RQ1은 판정 불가."""
    harmful_es, neutral_es = oracle(Fraction(-3, 20), ES), oracle(Fraction(0), ES)
    useful_cd = oracle(Fraction(3, 20), CD)
    gates = ("C1", "C2", "C3", "C4")

    def score(name: str, cd_acc: set[str], es_acc: set[str], es: OracleEstimate) -> CandidateScore:
        accepted = {(g, CD): g in cd_acc for g in gates} | {(g, ES): g in es_acc for g in gates}
        return CandidateScore(name, accepted, {CD: useful_cd, ES: es})

    candidates = [
        score("a", {"C2"}, set(), harmful_es),
        score("b", {"C2"}, {"C2", "C4"}, neutral_es),
    ]
    pairs = {
        g: (
            gate_metrics(candidates, g, CD, oracle_metric=ES).harmful,
            gate_metrics(candidates, g, ES).harmful,
        )
        for g in ("C2", "C4")
    }
    assert pairs["C4"][0].n == 0
    v = rq1(pairs)
    assert v.verdict == UNDECIDABLE
    assert v.comparisons[0].holds is True  # C2: 1/2 > 0/1


# ---------------------------------------------------------------------------
# RQ2 첫째 문장 (§8-26): Harmful·Local Harm 각각 C1 > C2, C1 > C4
# ---------------------------------------------------------------------------


def rq2_rates(
    c1: tuple[Rate, Rate], c2: tuple[Rate, Rate], c4: tuple[Rate, Rate]
) -> dict[tuple[str, str], Rate]:
    """(harmful, local_harm) 쌍을 게이트별로."""
    out: dict[tuple[str, str], Rate] = {}
    for gate, (h, lh) in (("C1", c1), ("C2", c2), ("C4", c4)):
        out[("harmful", gate)] = h
        out[("local_harm", gate)] = lh
    return out


def test_rq2_first_supported() -> None:
    v = rq2_first(
        rq2_rates((rate(2, 4), rate(2, 4)), (rate(0, 4), rate(1, 4)), (rate(0, 2), rate(0, 2)))
    )
    assert v.verdict == SUPPORTED
    assert len(v.comparisons) == 4


def test_rq2_first_partial() -> None:
    v = rq2_first(
        rq2_rates((rate(2, 4), rate(1, 4)), (rate(0, 4), rate(1, 4)), (rate(0, 2), rate(0, 2)))
    )
    # Local Harm C1 = C2(동률) → 그 비교는 성립 안 함
    assert [c.holds for c in v.comparisons] == [True, True, False, True]
    assert v.verdict == PARTIAL


def test_rq2_first_tie_everywhere_is_not_supported() -> None:
    v = rq2_first(
        rq2_rates((rate(1, 4), rate(1, 4)), (rate(1, 4), rate(1, 4)), (rate(1, 4), rate(1, 4)))
    )
    assert v.verdict == NOT_SUPPORTED


def test_rq2_first_some_comparisons_undecidable() -> None:
    """§8-33: C4가 아무것도 채택하지 않아 C1 대 C4 비교 둘이 판정 불가 → 종합 판정 불가.
    C1 대 C2 비교 둘은 성립하고 각각 보고된다."""
    v = rq2_first(rq2_rates((rate(2, 4), rate(2, 4)), (rate(0, 4), rate(0, 4)), (EMPTY, EMPTY)))
    assert v.verdict == UNDECIDABLE
    assert [(c.name, c.holds) for c in v.comparisons] == [
        ("harmful: C1 > C2", True),
        ("harmful: C1 > C4", None),
        ("local_harm: C1 > C2", True),
        ("local_harm: C1 > C4", None),
    ]


# ---------------------------------------------------------------------------
# RQ2 둘째 문장 (§8-30): 민감도 재표본 평균 차이 > 원래 차이
# ---------------------------------------------------------------------------


def test_rq2_second_partly_grows_harmful_but_not_local_harm() -> None:
    """Harmful: C1이 1/4 → 민감도 평균 1/2로 커져 차이가 커짐. Local Harm: C1이 1/4 → 1/4(같음)."""
    original = {
        ("harmful", "C1"): Fraction(1, 4),
        ("harmful", "C2"): Fraction(0),
        ("harmful", "C4"): Fraction(0),
        ("local_harm", "C1"): Fraction(1, 4),
        ("local_harm", "C2"): Fraction(0),
        ("local_harm", "C4"): Fraction(0),
    }
    sensitivity = {"harmful": Fraction(1, 2), "local_harm": Fraction(1, 4)}
    v = rq2_second(original, sensitivity)
    assert v.verdict == PARTLY_GROWS
    assert [c.holds for c in v.comparisons] == [True, True, False, False]


def test_rq2_second_grows_and_does_not_grow() -> None:
    original = dict.fromkeys(
        [(r, g) for r in ("harmful", "local_harm") for g in ("C1", "C2", "C4")], Fraction(1, 10)
    )
    assert (
        rq2_second(original, {"harmful": Fraction(1, 5), "local_harm": Fraction(1, 5)}).verdict
        == GROWS
    )
    assert (
        rq2_second(original, {"harmful": Fraction(1, 20), "local_harm": Fraction(1, 10)}).verdict
        == NOT_GROWS
    )


def test_rq2_second_undecidable_when_a_rate_is_undefined() -> None:
    original: dict[tuple[str, str], Fraction | None] = dict.fromkeys(
        [(r, g) for r in ("harmful", "local_harm") for g in ("C1", "C2", "C4")], Fraction(1, 10)
    )
    original[("harmful", "C4")] = None
    v = rq2_second(original, {"harmful": Fraction(1, 5), "local_harm": Fraction(1, 5)})
    assert v.verdict == UNDECIDABLE
    assert [c.holds for c in v.comparisons] == [True, None, True, True]


# ---------------------------------------------------------------------------
# RQ3 (§8-27·31·32)
# ---------------------------------------------------------------------------


def test_rq3_first_supported_partial_not() -> None:
    assert (
        rq3_first(
            useful={"C2": rate(1, 4), "C4": rate(1, 2)},
            harmful={"C2": rate(1, 4), "C4": rate(0, 2)},
        ).verdict
        == SUPPORTED
    )
    v = rq3_first(
        useful={"C2": rate(1, 4), "C4": rate(1, 2)}, harmful={"C2": rate(0, 4), "C4": rate(0, 2)}
    )
    assert v.verdict == PARTIAL  # Harmful 동률 → 그 조건 성립 안 함
    assert (
        rq3_first(
            useful={"C2": rate(1, 2), "C4": rate(1, 2)},
            harmful={"C2": rate(0, 4), "C4": rate(1, 2)},
        ).verdict
        == NOT_SUPPORTED
    )


def test_rq3_first_undecidable_when_c4_accepts_nothing() -> None:
    v = rq3_first(useful={"C2": rate(1, 4), "C4": EMPTY}, harmful={"C2": rate(1, 4), "C4": EMPTY})
    assert v.verdict == UNDECIDABLE


def test_rq3_trend_categories() -> None:
    falling = {24: Fraction(1, 2), 48: Fraction(1, 3), 96: Fraction(1, 4), 204: Fraction(1, 5)}
    flat = dict.fromkeys((24, 48, 96, 204), Fraction(1, 5))
    rising = {24: Fraction(0), 48: Fraction(1, 10), 96: Fraction(1, 5), 204: Fraction(1, 4)}
    assert rq3_trend(falling, falling).verdict == TREND_SUPPORTED
    assert (
        rq3_trend(falling, flat).verdict == TREND_PARTIAL
    )  # d_H(24) = d_H(204): 동률은 성립 안 함
    assert rq3_trend(flat, rising).verdict == TREND_NOT_SUPPORTED


def test_rq3_trend_reports_monotonicity_separately() -> None:
    bumpy = {24: Fraction(1, 2), 48: Fraction(3, 5), 96: Fraction(1, 4), 204: Fraction(1, 5)}
    v = rq3_trend(bumpy, bumpy)
    assert v.verdict == TREND_SUPPORTED  # d(24) > d(204)만 본다
    assert v.monotone == {"useful": False, "harmful": False}
    falling = {24: Fraction(1, 2), 48: Fraction(1, 3), 96: Fraction(1, 3), 204: Fraction(1, 5)}
    assert rq3_trend(falling, falling).monotone == {"useful": True, "harmful": True}


def test_rq3_trend_undecidable_when_a_point_is_missing() -> None:
    d = {24: Fraction(1, 2), 48: Fraction(1, 3), 96: Fraction(1, 4), 204: None}
    full = {24: Fraction(1, 2), 48: Fraction(1, 3), 96: Fraction(1, 4), 204: Fraction(1, 5)}
    v = rq3_trend(d, full)
    assert v.verdict == UNDECIDABLE
    assert [c.holds for c in v.comparisons] == [None, True]
    assert v.monotone == {"useful": None, "harmful": True}
