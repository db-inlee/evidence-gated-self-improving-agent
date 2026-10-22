"""RQ1~RQ3 판정 (7.8 G, Ticket 005 §8-24~28·30~33).

- 판정은 proposer 후보로 하고(7.8 G), RQ2·RQ3은 CD로 판정한다(§8-24). 이 모듈은 이미 계산한
  비율·값을 받아 비교만 한다.
- 비교는 엄격 부등호다. 같으면 성립하지 않는다(§8-28). 값이 없으면(분모 0) 그 비교는 판정 불가다.
- 한 문장 안의 비교 중 하나라도 판정 불가면 그 문장의 종합은 판정 불가다. 판정 가능한 비교의
  결과는 각각 남긴다(§8-33).
- RQ2 둘째 문장에서는 C2·C4가 고정이라 "민감도 차이 > 원래 차이"가 "C1 재표본 평균 > C1 원래
  값"과 같다. 그래서 C2·C4 쌍은 항상 같은 결과를 낸다(결과 보고서 해석 주의, §8-33).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise

from evidencegate.eval_constants import BUDGET_NS
from evidencegate.eval_metrics import Rate

SUPPORTED = "supported"  # 지지
PARTIAL = "partially_supported"  # 부분 지지
NOT_SUPPORTED = "not_supported"  # 지지 안 됨
UNDECIDABLE = "undecidable"  # 판정 불가
GROWS = "grows"  # 커진다
PARTLY_GROWS = "partly_grows"  # 일부 커진다
NOT_GROWS = "does_not_grow"  # 커지지 않는다
TREND_SUPPORTED = "trend_supported"  # 추세 지지
TREND_PARTIAL = "partial_trend_support"  # 부분 추세 지지
TREND_NOT_SUPPORTED = "trend_not_supported"  # 추세 지지 안 됨

RATE_NAMES = ("harmful", "local_harm")


@dataclass(frozen=True)
class Comparison:
    name: str
    left: Fraction | None
    right: Fraction | None
    holds: bool | None  # None = 판정 불가


@dataclass(frozen=True)
class Verdict:
    verdict: str
    comparisons: tuple[Comparison, ...]
    monotone: Mapping[str, bool | None] = field(default_factory=dict)  # RQ3 추세(기술용)


def rq1(pairs: Mapping[str, tuple[Rate, Rate]]) -> Verdict:
    """게이트 → (A = P(ES 해로움 | CD 채택), B = P(ES 해로움 | ES 채택)). A > B면 그 게이트 지지."""
    comparisons = [_compare(g, a.value, b.value, _gt) for g, (a, b) in pairs.items()]
    return _verdict(comparisons, (SUPPORTED, PARTIAL, NOT_SUPPORTED))


def rq2_first(rates: Mapping[tuple[str, str], Rate]) -> Verdict:
    """(지표, 게이트) → 비율. Harmful·Local Harm 각각 C1 > C2, C1 > C4."""
    comparisons = [
        _compare(f"{r}: C1 > {g}", rates[(r, "C1")].value, rates[(r, g)].value, _gt)
        for r in RATE_NAMES
        for g in ("C2", "C4")
    ]
    return _verdict(comparisons, (SUPPORTED, PARTIAL, NOT_SUPPORTED))


def rq2_second(
    original: Mapping[tuple[str, str], Fraction | None],
    sensitivity_c1: Mapping[str, Fraction | None],
) -> Verdict:
    """원래 비율 (지표, 게이트)와 민감도 재표본 평균의 C1 비율로 차이 rate(C1) - rate(Cx)를 비교."""
    comparisons = []
    for r in RATE_NAMES:
        for g in ("C2", "C4"):
            cx, c1, s1 = original[(r, g)], original[(r, "C1")], sensitivity_c1[r]
            sens = None if s1 is None or cx is None else s1 - cx
            orig = None if c1 is None or cx is None else c1 - cx
            comparisons.append(_compare(f"{r}: C1 - {g}", sens, orig, _gt))
    return _verdict(comparisons, (GROWS, PARTLY_GROWS, NOT_GROWS))


def rq3_first(useful: Mapping[str, Rate], harmful: Mapping[str, Rate]) -> Verdict:
    """N = 210에서 Useful(C4) > Useful(C2), Harmful(C4) < Harmful(C2)."""
    comparisons = [
        _compare("useful: C4 > C2", useful["C4"].value, useful["C2"].value, _gt),
        _compare("harmful: C4 < C2", harmful["C4"].value, harmful["C2"].value, _lt),
    ]
    return _verdict(comparisons, (SUPPORTED, PARTIAL, NOT_SUPPORTED))


def rq3_trend(
    d_useful: Mapping[int, Fraction | None], d_harmful: Mapping[int, Fraction | None]
) -> Verdict:
    """d_U(N) = Useful(C4) - Useful(C2), d_H(N) = Harmful(C2) - Harmful(C4). d(24) > d(204)."""
    first, last = BUDGET_NS[0], BUDGET_NS[-1]
    comparisons = [
        _compare(f"{name}: d({first}) > d({last})", d[first], d[last], _gt)
        for name, d in (("useful", d_useful), ("harmful", d_harmful))
    ]
    monotone = {"useful": _non_increasing(d_useful), "harmful": _non_increasing(d_harmful)}
    return Verdict(
        _verdict(comparisons, (TREND_SUPPORTED, TREND_PARTIAL, TREND_NOT_SUPPORTED)).verdict,
        tuple(comparisons),
        monotone,
    )


def _compare(
    name: str,
    left: Fraction | None,
    right: Fraction | None,
    op: Callable[[Fraction, Fraction], bool],
) -> Comparison:
    holds = None if left is None or right is None else op(left, right)
    return Comparison(name, left, right, holds)


def _verdict(comparisons: Sequence[Comparison], labels: tuple[str, str, str]) -> Verdict:
    full, partial, none = labels
    results = [c.holds for c in comparisons]
    if any(h is None for h in results):
        verdict = UNDECIDABLE
    elif all(results):
        verdict = full
    elif any(results):
        verdict = partial
    else:
        verdict = none
    return Verdict(verdict, tuple(comparisons))


def _non_increasing(d: Mapping[int, Fraction | None]) -> bool | None:
    known = [v for n in BUDGET_NS if (v := d[n]) is not None]
    if len(known) != len(BUDGET_NS):
        return None
    return all(a >= b for a, b in pairwise(known))


def _gt(a: Fraction, b: Fraction) -> bool:
    return a > b


def _lt(a: Fraction, b: Fraction) -> bool:
    return a < b
