"""Oracle 채점, τ_cell, 불확정, 하위집합(E), 적합성(F) (7.8 D·E·F, Ticket 005 §8-13~17·22·23)."""

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest

from evidencegate.eval_constants import EPSILON
from evidencegate.eval_data import CELLS, Cell, PairedTable, cell_of, split_cases
from evidencegate.exact import percentile_exact, stratified_bootstrap_indices
from evidencegate.gates import bootstrap_quantiles
from evidencegate.oracle import (
    OracleEstimate,
    check_control_label,
    classify_effect,
    estimate,
    fitness,
    harmful,
    local_harm,
    subset_label,
    tau_cells,
    unsupported,
    useful,
)

CD = "correct_disposition"
ORACLE = split_cases("oracle")
EPS = Fraction(1, 20)


def oracle_table(deltas: Sequence[int], metric: str = CD) -> PairedTable:
    cand = [1 if d > 0 else 0 for d in deltas]
    base = [1 if d < 0 else 0 for d in deltas]
    return PairedTable(
        "oracle", metric, tuple(ORACLE), tuple(cell_of(c) for c in ORACLE), tuple(cand), tuple(base)
    )


def per_cell(values: dict[int, list[int]]) -> list[int]:
    """칸 번호(CELLS 순서) → 그 칸 앞쪽 케이스의 Δ. 나머지는 0. 칸마다 50개."""
    out: list[int] = []
    for i in range(6):
        v = values.get(i, [])
        out += v + [0] * (50 - len(v))
    return out


def flat_tau(value: Fraction = EPS) -> dict[Cell, Fraction]:
    return dict.fromkeys(CELLS, value)


def midpoint(interval: tuple[Fraction, Fraction]) -> Fraction:
    return (interval[0] + interval[1]) / 2


def est(
    delta: Fraction,
    interval: tuple[Fraction, Fraction],
    cells: dict[Cell, Fraction] | None = None,
    tau: dict[Cell, Fraction] | None = None,
) -> OracleEstimate:
    return OracleEstimate(
        metric=CD,
        delta=delta,
        interval=interval,
        cell_deltas=cells or dict.fromkeys(CELLS, delta),
        tau=tau or flat_tau(),
    )


# ---------------------------------------------------------------------------
# Oracle Δ, 칸별 Δ, 구간
# ---------------------------------------------------------------------------


def test_epsilon_is_five_points() -> None:
    assert EPSILON == EPS


def test_estimate_overall_and_cell_deltas() -> None:
    t = oracle_table(per_cell({0: [1] * 10, 3: [-1] * 5}))
    e = estimate(t, flat_tau())
    assert e.delta == Fraction(5, 300)
    assert e.cell_deltas[CELLS[0]] == Fraction(10, 50)
    assert e.cell_deltas[CELLS[3]] == Fraction(-5, 50)
    assert e.cell_deltas[CELLS[1]] == 0


def test_estimate_interval_is_the_2_5_and_97_5_percentiles() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    t = oracle_table(rng.choice([-1, 0, 0, 1], size=300).tolist())
    e = estimate(t, flat_tau())
    assert e.interval == bootstrap_quantiles(t, (Fraction(1, 40), Fraction(39, 40)))
    assert e.interval[0] <= e.delta <= e.interval[1]


def reference_quantiles(
    t: PairedTable, qs: Sequence[Fraction], n_boot: int, seed: int
) -> tuple[Fraction, ...]:
    groups = t.cell_indices()
    blocks = stratified_bootstrap_indices([len(g) for g in groups], n_boot=n_boot, seed=seed)
    means = [
        Fraction(
            sum(t.deltas[g[i]] for g, block in zip(groups, blocks, strict=True) for i in block[b]),
            len(t.deltas),
        )
        for b in range(n_boot)
    ]
    return tuple(percentile_exact(means, q) for q in qs)


def test_bootstrap_quantiles_match_the_reference() -> None:
    rng = np.random.Generator(np.random.PCG64(2))
    t = oracle_table(rng.choice([-1, 0, 1], size=300).tolist())
    qs = (Fraction(1, 40), Fraction(39, 40))
    assert bootstrap_quantiles(t, qs, n_boot=200, seed=0) == reference_quantiles(t, qs, 200, 0)


def test_estimate_is_deterministic() -> None:
    rng = np.random.Generator(np.random.PCG64(3))
    t = oracle_table(rng.choice([-1, 0, 1], size=300).tolist())
    assert estimate(t, flat_tau()) == estimate(t, flat_tau())


def test_estimate_needs_the_oracle_split() -> None:
    cases = split_cases("dev_b")
    t = PairedTable(
        "dev_b", CD, tuple(cases), tuple(cell_of(c) for c in cases), (0,) * 210, (0,) * 210
    )
    with pytest.raises(ValueError, match="oracle"):
        estimate(t, flat_tau())


# ---------------------------------------------------------------------------
# τ_cell (7.8 D, §8-22): 반복 하나 대 나머지 4회 평균, 다섯 값의 q95, max(ε, q95)
# ---------------------------------------------------------------------------


def s0_repeats(cell_flips: dict[int, int]) -> dict[int, dict[tuple[str, str, int], int]]:
    """칸 번호 → 그 칸에서 반복 0만 1인 케이스 수. 나머지는 모든 반복이 0."""
    table: dict[int, dict[tuple[str, str, int], int]] = {r: {} for r in range(5)}
    for i, cell in enumerate(CELLS):
        cases = [c for c in ORACLE if cell_of(c) == cell]
        for j, case in enumerate(cases):
            for r in range(5):
                table[r][case] = int(r == 0 and j < cell_flips.get(i, 0))
    return table


def test_tau_is_epsilon_without_noise() -> None:
    assert tau_cells(s0_repeats({})) == flat_tau()


def test_tau_by_hand() -> None:
    """칸 1에서 반복 0만 10개가 1: r=0 → |10/50 - 0| = 1/5, r=1..4 → |0 - (10·1/4)/50| = 1/20.
    정렬 [1/20, 1/20, 1/20, 1/20, 1/5]의 q95(위치 3.8) = 1/20 + (3/20)(4/5) = 17/100 > ε."""
    tau = tau_cells(s0_repeats({1: 10}))
    assert tau[CELLS[1]] == Fraction(17, 100)
    assert all(tau[c] == EPS for c in CELLS if c != CELLS[1])


def test_tau_is_per_metric_and_never_below_epsilon() -> None:
    """작은 잡음: 칸 2에서 1개 → 값 [1/200 x4, 1/50].
    q95 = 1/200 + (3/200)(4/5) = 17/1000 < ε → τ = ε."""
    tau = tau_cells(s0_repeats({2: 1}))
    assert tau[CELLS[2]] == EPS


def test_tau_needs_five_repeats() -> None:
    table = s0_repeats({})
    del table[4]
    with pytest.raises(ValueError, match="5 repeats"):
        tau_cells(table)


# ---------------------------------------------------------------------------
# 불확정 판정 (§8-13·15): 판정 조건을 하한·상한에 각각 적용해 다르면 불확정(None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((Fraction(1, 10), Fraction(1, 5)), True),
        ((EPS, Fraction(1, 5)), None),  # 경계: 하한 = ε → 하한에서 거짓, 상한에서 참
        ((EPS, EPS), False),  # Δ > ε는 엄격 부등호
        ((Fraction(-1, 10), Fraction(1, 25)), False),
    ],
)
def test_useful_uses_the_plus_epsilon_boundary(
    interval: tuple[Fraction, Fraction], expected: bool | None
) -> None:
    assert useful(est(midpoint(interval), interval)) is expected


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((Fraction(-1, 5), Fraction(-1, 10)), True),
        ((Fraction(-1, 5), -EPS), None),  # 상한 = -ε → 상한에서 거짓
        ((-EPS, Fraction(1, 10)), False),
    ],
)
def test_harmful_uses_the_minus_epsilon_boundary(
    interval: tuple[Fraction, Fraction], expected: bool | None
) -> None:
    assert harmful(est(midpoint(interval), interval)) is expected


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((Fraction(-1, 100), Fraction(1, 10)), True),
        ((Fraction(0), Fraction(1, 10)), True),
        ((Fraction(1, 100), Fraction(1, 10)), False),
    ],
)
def test_unsupported_is_lower_bound_at_most_zero_without_uncertainty(
    interval: tuple[Fraction, Fraction], expected: bool
) -> None:
    """Unsupported는 예외(7.8 D): 구간 하한 <= 0이면 참, 불확정 없음."""
    assert unsupported(est(Fraction(1, 20), interval)) is expected


def test_oracle_delta_exactly_epsilon_is_not_useful_but_uncertain() -> None:
    """경계: Oracle Δ = ε = 15/300. 점추정은 '유용' 아님(엄격). 구간이 ε를 가로지르면 불확정."""
    t = oracle_table(per_cell({0: [1] * 8, 1: [1] * 7}))
    e = estimate(t, flat_tau())
    assert e.delta == Fraction(15, 300) == EPSILON
    assert e.interval[0] < EPSILON < e.interval[1]
    assert useful(e) is None
    assert classify_effect(e) is None
    assert harmful(e) is False


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((Fraction(1, 10), Fraction(1, 5)), "up"),
        ((Fraction(-1, 5), Fraction(-1, 10)), "down"),
        ((-EPS, EPS), "none"),  # 양 끝 모두 |x| <= ε
        (
            (Fraction(-1, 10), Fraction(1, 10)),
            None,
        ),  # neutral 띠 전체를 덮어도 양 끝이 down·up이라 불확정
        ((Fraction(0), Fraction(1, 10)), None),
        ((EPS, Fraction(1, 10)), None),
    ],
)
def test_classify_effect_three_way(
    interval: tuple[Fraction, Fraction], expected: str | None
) -> None:
    """§8-17·18: down / none / up 분류에 양 끝 규칙. 구간이 ±ε를 가로지르면 불확정."""
    assert classify_effect(est(midpoint(interval), interval)) == expected


# ---------------------------------------------------------------------------
# 칸 단위: Local Harm, 하위집합(E) — 점추정, τ_cell
# ---------------------------------------------------------------------------

CERTAIN_UP = (Fraction(1, 10), Fraction(1, 5))
MIXED_TAU = {c: (Fraction(17, 100) if i == 1 else EPS) for i, c in enumerate(CELLS)}


def cells_with(values: dict[int, Fraction]) -> dict[Cell, Fraction]:
    return {c: values.get(i, Fraction(0)) for i, c in enumerate(CELLS)}


def test_local_harm_with_mixed_tau_cells() -> None:
    """τ_cell = ε인 칸과 τ_cell > ε인 칸이 섞인 경우. Δ_cell = -1/10은 칸 0(τ = ε)에서만 해로움."""
    harm_in_noisy_cell = est(
        Fraction(3, 20), CERTAIN_UP, cells_with({1: Fraction(-1, 10)}), MIXED_TAU
    )
    harm_in_quiet_cell = est(
        Fraction(3, 20), CERTAIN_UP, cells_with({0: Fraction(-1, 10)}), MIXED_TAU
    )
    assert local_harm(harm_in_noisy_cell) is False
    assert local_harm(harm_in_quiet_cell) is True


def test_local_harm_boundary_is_strict() -> None:
    """Δ_cell = -τ_cell 정확히는 해로움 아님(엄격 부등호)."""
    e = est(Fraction(3, 20), CERTAIN_UP, cells_with({1: Fraction(-17, 100)}), MIXED_TAU)
    assert local_harm(e) is False
    e = est(Fraction(3, 20), CERTAIN_UP, cells_with({1: Fraction(-18, 100)}), MIXED_TAU)
    assert local_harm(e) is True


def test_subset_positive_and_mixed_with_mixed_tau_cells() -> None:
    noisy_dip = est(Fraction(3, 20), CERTAIN_UP, cells_with({1: Fraction(-1, 10)}), MIXED_TAU)
    quiet_dip = est(Fraction(3, 20), CERTAIN_UP, cells_with({0: Fraction(-1, 10)}), MIXED_TAU)
    assert subset_label(noisy_dip).label == "positive"
    assert subset_label(quiet_dip).label == "mixed"


def test_subset_negative() -> None:
    e = est(Fraction(-3, 20), (Fraction(-1, 5), Fraction(-1, 10)))
    assert subset_label(e).label == "negative"


def test_subset_neutral_reports_big_cells_by_tau() -> None:
    """neutral 안의 큰 칸: |Δ_cell| > τ_cell.
    칸 1(τ = 17/100)의 1/10은 크지 않고, 칸 0(τ = ε)의 1/10은 크다."""
    e = est(
        Fraction(0),
        (-EPS / 2, EPS / 2),
        cells_with({0: Fraction(1, 10), 1: Fraction(1, 10), 4: Fraction(-1, 10)}),
        MIXED_TAU,
    )
    result = subset_label(e)
    assert result.label == "neutral"
    assert result.big_cells == (CELLS[0], CELLS[4])


def test_subset_uncertain_when_the_interval_crosses_epsilon() -> None:
    e = est(Fraction(1, 10), (EPS, Fraction(3, 20)))
    assert subset_label(e).label == "uncertain"


# ---------------------------------------------------------------------------
# 통제 후보 라벨 검증 (7.8 E, §8-17)
# ---------------------------------------------------------------------------


def test_control_matching_its_label() -> None:
    e = est(Fraction(3, 20), CERTAIN_UP)
    assert check_control_label("positive", e) == "as_expected"


def test_control_with_a_different_label_is_unexpected() -> None:
    e = est(Fraction(-3, 20), (Fraction(-1, 5), Fraction(-1, 10)))
    assert check_control_label("positive", e) == "unexpected"


def test_uncertain_control_is_reported_as_uncertain_not_unexpected() -> None:
    """§8-17: 통제 후보가 불확정이면 '예상과 다름'이 아니라 '불확정'."""
    e = est(Fraction(1, 10), (EPS, Fraction(3, 20)))
    assert check_control_label("positive", e) == "uncertain"
    assert check_control_label("negative", e) == "uncertain"


def test_unknown_prior_label_is_an_error() -> None:
    with pytest.raises(ValueError, match="label"):
        check_control_label("pos", est(Fraction(0), (-EPS, EPS)))


# ---------------------------------------------------------------------------
# Oracle 적합성 F (7.8 F, §8-23): 통제 후보만, metric마다 부호 또는 크기 기준
# ---------------------------------------------------------------------------


def test_fitness_sign_criterion_within_a_family() -> None:
    """같은 family의 L1은 Δ_cell > τ, L2는 Δ_cell < -τ."""
    e = est(Fraction(0), (-EPS, EPS), cells_with({0: Fraction(1, 10), 1: Fraction(-1, 10)}))
    result = fitness({"ctrl": e})
    assert result.sign_met and result.fit
    assert ("ctrl", "bad_deployment") in result.sign_hits


def test_fitness_sign_needs_the_same_family() -> None:
    e = est(Fraction(0), (-EPS, EPS), cells_with({0: Fraction(1, 10), 3: Fraction(-1, 10)}))
    assert not fitness({"ctrl": e}).sign_met


def test_fitness_size_criterion_is_at_least_30_points() -> None:
    exact = est(Fraction(0), (-EPS, EPS), cells_with({4: Fraction(3, 10), 5: Fraction(0)}))
    below = est(Fraction(0), (-EPS, EPS), cells_with({4: Fraction(29, 100), 5: Fraction(0)}))
    assert fitness({"c": exact}).size_met
    assert fitness({"c": exact}).size_hits == (("c", "traffic_spike"),)
    assert not fitness({"c": below}).size_met


def test_fitness_fails_when_neither_criterion_is_met() -> None:
    e = est(Fraction(0), (-EPS, EPS), cells_with({0: Fraction(1, 10)}))
    result = fitness({"a": e, "b": est(Fraction(0), (-EPS, EPS))})
    assert not result.fit
    assert (result.sign_met, result.size_met) == (False, False)


def test_fitness_sign_uses_each_cells_own_tau() -> None:
    """L1(칸 0, τ = ε)은 +1/10 > τ, L2(칸 1, τ = 17/100)는 -1/10이 -τ보다 크다 → 부호 기준 아님."""
    e = est(
        Fraction(0), (-EPS, EPS), cells_with({0: Fraction(1, 10), 1: Fraction(-1, 10)}), MIXED_TAU
    )
    assert not fitness({"ctrl": e}).sign_met


def test_neutral_big_cell_boundary_is_strict() -> None:
    """|Δ_cell| = τ_cell 정확히는 큰 칸이 아니다(엄격 부등호). 칸 1의 τ = 17/100."""
    e = est(Fraction(0), (-EPS / 2, EPS / 2), cells_with({1: Fraction(17, 100), 0: EPS}), MIXED_TAU)
    assert subset_label(e).big_cells == ()


def test_fitness_sign_boundary_is_strict() -> None:
    """Δ_cell = τ_cell 정확히는 '> τ'가 아니다."""
    e = est(Fraction(0), (-EPS, EPS), cells_with({0: EPS, 1: Fraction(-1, 10)}))
    assert not fitness({"ctrl": e}).sign_met
