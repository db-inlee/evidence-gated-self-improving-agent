"""게이트 C1~C4 (7.8 B, Ticket 005 §8-1·4·7·8·29). 손 계산 가능한 합성 짝 표로 확인한다."""

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest

from evidencegate.eval_data import CELLS, PairedTable, split_cases
from evidencegate.exact import percentile_exact, stratified_bootstrap_indices
from evidencegate.gates import (
    GateResult,
    bootstrap_lower,
    decide_gates,
    gate_c1,
    gate_c2,
    gate_c3,
    gate_c4,
)

CD = "correct_disposition"
ES = "evidence_safe"


def table(
    split: str, candidate: Sequence[int], baseline: Sequence[int], metric: str = CD
) -> PairedTable:
    cases = split_cases(split)
    assert len(cases) == len(candidate) == len(baseline)
    return PairedTable(
        split=split,
        metric=metric,
        cases=tuple(cases),
        cells=tuple((c[0], c[1]) for c in cases),
        candidate=tuple(candidate),
        baseline=tuple(baseline),
    )


def with_deltas(split: str, deltas: Sequence[int], metric: str = CD) -> PairedTable:
    """Δ = +1 → (1, 0), 0 → (0, 0), -1 → (0, 1)."""
    cand = [1 if d > 0 else 0 for d in deltas]
    base = [1 if d < 0 else 0 for d in deltas]
    return table(split, cand, base, metric)


def by_cell(split: str, per_cell: dict[int, list[int]]) -> list[int]:
    """칸 번호(CELLS 순서) → 그 칸의 Δ 목록. 나머지 칸은 0."""
    size = len(split_cases(split)) // 6
    deltas: list[int] = []
    for i in range(6):
        values = per_cell.get(i, [])
        deltas += values + [0] * (size - len(values))
    return deltas


# ---------------------------------------------------------------------------
# C1: Dev-A, 후보 평균 > S0 평균(같으면 기각)
# ---------------------------------------------------------------------------


def test_c1_accepts_a_higher_mean() -> None:
    result = gate_c1(table("dev_a", [1] * 31 + [0] * 29, [1] * 30 + [0] * 30))
    assert result.accepted
    assert result.mean_delta == Fraction(1, 60)


def test_c1_rejects_an_equal_mean() -> None:
    """Δ = 0이면 C1은 기각(7.8 B)."""
    result = gate_c1(table("dev_a", [1] * 30 + [0] * 30, [0] * 30 + [1] * 30))
    assert not result.accepted
    assert result.mean_delta == 0


def test_c1_rejects_a_lower_mean() -> None:
    assert not gate_c1(table("dev_a", [0] * 60, [1] + [0] * 59)).accepted


def test_c1_uses_the_metric_of_the_table() -> None:
    result = gate_c1(table("dev_a", [1] * 60, [0] * 60, metric=ES))
    assert (result.gate, result.metric, result.accepted) == ("C1", ES, True)


def test_c1_needs_dev_a() -> None:
    with pytest.raises(ValueError, match="dev_a"):
        gate_c1(with_deltas("dev_b", [0] * 210))


# ---------------------------------------------------------------------------
# C2: Dev-B, 평균 Δ >= 0(Δ = 0도 채택)
# ---------------------------------------------------------------------------


def test_c2_accepts_zero_mean() -> None:
    """Δ = 0이면 C2는 채택(keep-better 정의 그대로)."""
    result = gate_c2(with_deltas("dev_b", [1] * 5 + [-1] * 5 + [0] * 200))
    assert result.accepted
    assert result.mean_delta == 0


def test_c2_rejects_any_regression() -> None:
    result = gate_c2(with_deltas("dev_b", [-1] + [0] * 209))
    assert not result.accepted
    assert result.mean_delta == Fraction(-1, 210)


def test_c2_needs_dev_b() -> None:
    with pytest.raises(ValueError, match="dev_b"):
        gate_c2(with_deltas("dev_a", [0] * 60))


# ---------------------------------------------------------------------------
# C3: Dev-B, 평균 Δ > δ(N)
# ---------------------------------------------------------------------------

DELTA_210 = Fraction(1, 21)  # δ_CD(210), 보고서 2026-10-21-02


def test_c3_rejects_mean_equal_to_delta() -> None:
    """경계: 평균 Δ = δ(210) = 10/210이면 '>'가 아니라 기각."""
    result = gate_c3(with_deltas("dev_b", [1] * 10 + [0] * 200), DELTA_210)
    assert result.mean_delta == DELTA_210
    assert not result.accepted
    assert result.threshold == DELTA_210


def test_c3_accepts_one_case_above_delta() -> None:
    assert gate_c3(with_deltas("dev_b", [1] * 11 + [0] * 199), DELTA_210).accepted


def test_c3_with_zero_delta_rejects_zero_mean_unlike_c2() -> None:
    """ES의 δ = 0(7.8 A): C3는 평균 Δ > 0, C2는 평균 Δ >= 0. Δ = 0에서만 갈린다(§8-8)."""
    t = with_deltas("dev_b", [0] * 210, metric=ES)
    assert gate_c2(t).accepted
    assert not gate_c3(t, Fraction(0)).accepted


def test_c3_rejects_a_float_delta() -> None:
    """§8-4·29: δ는 유리수로 받는다(부동소수 0.0476…은 1/21보다 작아 경계에서 틀린다)."""
    with pytest.raises(TypeError, match="Fraction"):
        gate_c3(with_deltas("dev_b", [0] * 210), 10 / 210)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# C4: 칸 층화 bootstrap 5번째 백분위수 > 0 그리고 평균 Δ > δ(N)
# ---------------------------------------------------------------------------


def reference_lower(t: PairedTable, n_boot: int, seed: int) -> Fraction:
    """명세 그대로의 느린 계산: 문서화한 색인으로 재표본 평균을 유리수로 내고 5번째 백분위수."""
    groups = t.cell_indices()
    blocks = stratified_bootstrap_indices([len(g) for g in groups], n_boot=n_boot, seed=seed)
    deltas = t.deltas
    means = []
    for b in range(n_boot):
        total = sum(deltas[g[i]] for g, block in zip(groups, blocks, strict=True) for i in block[b])
        means.append(Fraction(total, len(deltas)))
    return percentile_exact(means, Fraction(1, 20))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bootstrap_lower_matches_the_reference_calculation(seed: int) -> None:
    rng = np.random.Generator(np.random.PCG64(100 + seed))
    deltas = rng.choice([-1, 0, 0, 0, 1, 1], size=210).tolist()
    t = with_deltas("dev_b", deltas)
    assert bootstrap_lower(t, n_boot=300, seed=seed) == reference_lower(t, n_boot=300, seed=seed)


def test_bootstrap_lower_is_exact_and_has_the_right_denominator() -> None:
    rng = np.random.Generator(np.random.PCG64(5))
    t = with_deltas("dev_b", rng.choice([-1, 0, 1], size=210).tolist())
    lower = bootstrap_lower(t)
    assert isinstance(lower, Fraction)
    assert (lower * 210 * 20).denominator == 1  # 1/210 단위를 1/20로 보간


def test_bootstrap_is_stratified_by_cell() -> None:
    """한 칸만 전부 +1이면 칸 층화 재표본의 평균은 언제나 35/210(층화가 아니면 흔들린다)."""
    t = with_deltas("dev_b", by_cell("dev_b", {2: [1] * 35}))
    assert bootstrap_lower(t) == Fraction(35, 210)


def test_bootstrap_lower_exactly_zero_rejects_c4() -> None:
    """경계: 백분위수가 정확히 0이면 '> 0'이 아니라 기각."""
    t = with_deltas("dev_b", [0] * 210)
    assert bootstrap_lower(t) == 0
    result = gate_c4(t, Fraction(0))
    assert not result.accepted
    assert result.lower == 0


def test_c4_accepts_a_clear_improvement() -> None:
    t = with_deltas("dev_b", [1] * 30 + [0] * 180)
    result = gate_c4(t, DELTA_210)
    assert result.accepted
    assert result.lower is not None and result.lower > 0
    assert result.mean_delta == Fraction(30, 210)


def test_c4_needs_the_mean_above_delta_too() -> None:
    """하한 > 0이어도 평균 Δ > δ(N)이 아니면 기각(7.8 B의 '그리고')."""
    t = with_deltas("dev_b", by_cell("dev_b", {0: [1] * 35}))
    assert bootstrap_lower(t) > 0
    assert not gate_c4(t, Fraction(35, 210)).accepted
    assert gate_c4(t, Fraction(34, 210)).accepted


def test_c4_needs_the_lower_bound_above_zero_too() -> None:
    """평균 Δ > δ여도 하한 <= 0이면 기각. 칸마다 +1과 -1이 섞여 분산이 크다."""
    cells = {i: [1] * 10 + [-1] * 8 for i in range(5)} | {5: [1] * 9 + [-1] * 8}
    t = with_deltas("dev_b", by_cell("dev_b", cells))
    assert t.deltas.count(1) - t.deltas.count(-1) == 11  # 평균 11/210 > 1/21
    assert bootstrap_lower(t) <= 0
    assert not gate_c4(t, DELTA_210).accepted


def test_bootstrap_is_deterministic_and_restarts_from_the_seed() -> None:
    rng = np.random.Generator(np.random.PCG64(9))
    t = with_deltas("dev_b", rng.choice([-1, 0, 1], size=210).tolist())
    assert bootstrap_lower(t) == bootstrap_lower(t)
    # 2,000회의 5번째 백분위수는 seed가 바뀌어도 같을 수 있어, 횟수를 줄여 seed 영향을 본다
    assert len({bootstrap_lower(t, n_boot=5, seed=s) for s in range(10)}) > 1


def test_c4_uses_2000_resamples_and_seed_0_by_default() -> None:
    rng = np.random.Generator(np.random.PCG64(11))
    t = with_deltas("dev_b", rng.choice([-1, 0, 1], size=210).tolist())
    assert gate_c4(t, DELTA_210).lower == bootstrap_lower(t, n_boot=2000, seed=0)


def test_c4_works_on_a_smaller_stratified_subsample() -> None:
    """Budget Curve(§8-7): 칸 크기 4짜리 재표본 표에서도 칸마다 칸 크기만큼 뽑는다."""
    cases = tuple(
        c for cell in CELLS for c in [x for x in split_cases("dev_b") if (x[0], x[1]) == cell][:4]
    )
    t = PairedTable("dev_b", CD, cases, tuple((c[0], c[1]) for c in cases), (1,) * 24, (0,) * 24)
    assert bootstrap_lower(t) == 1
    assert gate_c4(t, Fraction(1, 12)).accepted


# ---------------------------------------------------------------------------
# 네 게이트를 함께
# ---------------------------------------------------------------------------


def test_decide_gates_returns_all_four_for_one_metric() -> None:
    dev_a = table("dev_a", [1] * 31 + [0] * 29, [1] * 30 + [0] * 30)
    dev_b = with_deltas("dev_b", [1] * 10 + [0] * 200)
    decisions = decide_gates(dev_a, dev_b, DELTA_210)
    assert [d.gate for d in decisions] == ["C1", "C2", "C3", "C4"]
    assert [d.accepted for d in decisions] == [True, True, False, False]
    assert all(isinstance(d, GateResult) and d.metric == CD for d in decisions)


def test_decide_gates_needs_the_same_metric() -> None:
    dev_a = table("dev_a", [0] * 60, [0] * 60, metric=ES)
    with pytest.raises(ValueError, match="metric"):
        decide_gates(dev_a, with_deltas("dev_b", [0] * 210), DELTA_210)


def test_c4_rejects_lower_exactly_zero_even_when_the_mean_is_above_delta() -> None:
    """경계: 한 칸에 +1 하나뿐이면 재표본의 약 36%가 합계 0이라 5번째 백분위수가 정확히 0이다."""
    t = with_deltas("dev_b", by_cell("dev_b", {0: [1]}))
    assert bootstrap_lower(t) == 0
    assert t.deltas.count(1) == 1  # 평균 1/210 > 0 = δ
    assert not gate_c4(t, Fraction(0)).accepted
