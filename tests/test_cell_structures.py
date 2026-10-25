"""칸 구조 일반화 (Ticket 006 고정 결정 4: "칸 구조만 L3에 맞게 바꾼다").

Ticket 005 함수가 기본값(6칸 CELLS)으로는 예전과 똑같이 돌고, 칸 목록을 넘기면 그 칸으로 돈다.
"""

from fractions import Fraction

import pytest

from evidencegate.eval_constants import EPSILON
from evidencegate.eval_data import CELLS, PairedTable, outcomes, split_cases
from evidencegate.exact import mean_fraction, stratified_bootstrap_indices
from evidencegate.gates import bootstrap_quantiles
from evidencegate.oracle import OracleEstimate, local_harm, subset_label, tau_cells
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord

CD = "correct_disposition"
THREE = (("bad_deployment", "L3"), ("dependency_failure", "L3"), ("traffic_spike", "L3"))


def l3_table(deltas_by_cell: list[list[int]]) -> PairedTable:
    cases, cells, cand, base = [], [], [], []
    for cell, deltas in zip(THREE, deltas_by_cell, strict=True):
        for i, d in enumerate(deltas):
            cases.append((cell[0], "L3", 400_000 + i))
            cells.append(cell)
            cand.append(1 if d > 0 else 0)
            base.append(1 if d < 0 else 0)
    return PairedTable("l3_oracle", CD, tuple(cases), tuple(cells), tuple(cand), tuple(base))


def test_cell_indices_default_to_the_six_cells_and_accept_other_cells() -> None:
    table = l3_table([[1, 0], [0, -1, 0], [1]])
    assert table.cell_indices() == [[] for _ in CELLS]
    assert table.cell_indices(THREE) == [[0, 1], [2, 3, 4], [5]]


def test_bootstrap_quantiles_with_three_cells_match_a_manual_draw() -> None:
    table = l3_table([[1, 0, 0, 1], [0, -1, 0, 0], [1, 1, 0, -1]])
    qs = (Fraction(1, 40), Fraction(39, 40))
    got = bootstrap_quantiles(table, qs, n_boot=200, seed=0, cells=THREE)
    blocks = stratified_bootstrap_indices([4, 4, 4], n_boot=200, seed=0)
    deltas = table.deltas
    totals = sorted(
        sum(deltas[4 * c + int(i)] for c, block in enumerate(blocks) for i in block[b])
        for b in range(200)
    )

    def pct(q: Fraction) -> Fraction:
        pos = q * (len(totals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(totals) - 1)
        return (totals[lo] + (totals[hi] - totals[lo]) * (pos - lo)) / Fraction(12)

    assert got == (pct(qs[0]), pct(qs[1]))


def test_bootstrap_quantiles_still_need_every_cell() -> None:
    with pytest.raises(ValueError, match="every cell"):
        bootstrap_quantiles(l3_table([[1], [0], [1]]), (Fraction(1, 2),), n_boot=10)


def estimate_with(cells: dict[tuple[str, str], Fraction], tau: Fraction) -> OracleEstimate:
    return OracleEstimate(
        metric=CD,
        delta=Fraction(1, 10),
        interval=(Fraction(8, 100), Fraction(12, 100)),
        cell_deltas=cells,
        tau=dict.fromkeys(cells, tau),
    )


def test_local_harm_and_subsets_use_the_estimate_cells() -> None:
    harm = estimate_with(
        dict(zip(THREE, [Fraction(3, 10), Fraction(-1, 10), Fraction(0)], strict=True)), EPSILON
    )
    assert local_harm(harm)
    assert subset_label(harm).label == "mixed"
    fine = estimate_with(
        dict(zip(THREE, [Fraction(3, 10), Fraction(0), Fraction(0)], strict=True)), EPSILON
    )
    assert not local_harm(fine)
    assert subset_label(fine).label == "positive"


def test_neutral_big_cells_come_from_the_estimate_cells() -> None:
    e = OracleEstimate(
        metric=CD,
        delta=Fraction(0),
        interval=(Fraction(-1, 100), Fraction(1, 100)),
        cell_deltas=dict(zip(THREE, [Fraction(2, 10), Fraction(0), Fraction(-2, 10)], strict=True)),
        tau=dict.fromkeys(THREE, EPSILON),
    )
    result = subset_label(e)
    assert result.label == "neutral" and result.big_cells == (THREE[0], THREE[2])


def test_tau_cells_accept_cells_and_a_cell_key() -> None:
    """family 단위 묶음(L1+L2 → family 한 칸)으로 τ를 낸다."""
    cases = [
        (f, lv, s)
        for f in ("bad_deployment", "traffic_spike")
        for lv in ("L1", "L2")
        for s in (0, 1)
    ]
    flips = {("bad_deployment", "L1", 0): {0}}  # bad 칸에 반복 0에서만 1
    repeats = {r: {c: int(r in flips.get(c, set())) for c in cases} for r in range(5)}
    family_cells = (("bad_deployment", "L1+L2"), ("traffic_spike", "L1+L2"))
    tau = tau_cells(repeats, cells=family_cells, cell_key=lambda c: (c[0], "L1+L2"))
    assert set(tau) == set(family_cells)
    # bad 칸 4개: 반복 0의 |평균(x - 나머지 평균)| = 1/4, 나머지 반복은 |0 - 1/4| / 4 = 1/16
    noise = sorted([Fraction(1, 4), *[Fraction(1, 16)] * 4])
    q95 = noise[3] + (noise[4] - noise[3]) * Fraction(4, 5)  # 위치 0.95 x 4 = 3.8
    assert tau[family_cells[0]] == max(EPSILON, q95)
    assert tau[family_cells[1]] == EPSILON


def record(case: tuple[str, str, int], repeat: int, cd: bool) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey("p", "c", "t", "s0", case[0], case[2], repeat, case[1]),
        policy_config={},
        llm_seed=0,
        correct_disposition=cd,
        evidence_safe=cd,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-10-25T00:00:00+00:00",
        duration_s=1.0,
    )


def test_outcomes_accept_an_explicit_case_list() -> None:
    cases = [("bad_deployment", "L3", 400_000), ("traffic_spike", "L3", 400_000)]
    records = [record(cases[0], 0, True), record(cases[1], 0, False)]
    got = outcomes(records, skill_id="s0", split="l3_oracle", metric=CD, repeat=0, cases=cases)
    assert got == {cases[0]: 1, cases[1]: 0}
    with pytest.raises(ValueError, match="missing"):
        outcomes(records[:1], skill_id="s0", split="l3_oracle", metric=CD, repeat=0, cases=cases)


def test_outcomes_default_to_the_ticket_005_split_cases() -> None:
    cases = split_cases("dev_a")
    records = [record(c, 0, True) for c in cases]
    got = outcomes(records, skill_id="s0", split="dev_a", metric=CD, repeat=0)
    assert list(got) == cases and mean_fraction(list(got.values())) == 1
