"""Budget Curve와 RQ2 민감도 분석 (7.8 C, Ticket 005 §8-1·7·9~12)."""

from collections import Counter
from fractions import Fraction

import numpy as np
import pytest

from evidencegate.budget_curve import (
    BudgetCandidate,
    budget_decisions,
    budget_points,
    resample_positions,
    sensitivity_decisions,
    sensitivity_positions,
    sensitivity_rates,
    subtable,
    summarize_values,
)
from evidencegate.eval_constants import EPSILON
from evidencegate.eval_data import CELLS, PairedTable, cell_of, split_cases
from evidencegate.exact import stratified_bootstrap_indices
from evidencegate.gates import bootstrap_quantiles, gate_c1, gate_c2, gate_c3, gate_c4
from evidencegate.oracle import OracleEstimate

CD = "correct_disposition"
ES = "evidence_safe"
METRICS = (CD, ES)


def paired(split: str, deltas: list[int], metric: str) -> PairedTable:
    cases = split_cases(split)
    return PairedTable(
        split,
        metric,
        tuple(cases),
        tuple(cell_of(c) for c in cases),
        tuple(1 if d > 0 else 0 for d in deltas),
        tuple(1 if d < 0 else 0 for d in deltas),
    )


def oracle(delta: Fraction, metric: str) -> OracleEstimate:
    w = Fraction(1, 100)
    return OracleEstimate(
        metric,
        delta,
        (delta - w, delta + w),
        dict.fromkeys(CELLS, delta),
        dict.fromkeys(CELLS, EPSILON),
    )


def candidate(
    name: str, seed: int, *, bias: float = 0.0, delta: Fraction = Fraction(0)
) -> BudgetCandidate:
    rng = np.random.Generator(np.random.PCG64(seed))

    def deltas(n: int) -> list[int]:
        p = [0.2 - bias / 2, 0.6, 0.2 + bias / 2]
        return [int(x) for x in rng.choice([-1, 0, 1], size=n, p=p)]

    return BudgetCandidate(
        name=name,
        dev_a={m: paired("dev_a", deltas(60), m) for m in METRICS},
        dev_b={m: paired("dev_b", deltas(210), m) for m in METRICS},
        oracle={m: oracle(delta, m) for m in METRICS},
    )


# 보고서 2026-10-21-02의 δ_CD(N). N마다 달라서, N을 섞어 쓰면 판정이 달라진다. δ_ES = 0.
DELTA_CD = {24: Fraction(1, 12), 48: Fraction(1, 16), 96: Fraction(1, 16), 204: Fraction(5, 102)}
DELTAS = {(m, n): Fraction(0) if m == ES else DELTA_CD[n] for m in METRICS for n in DELTA_CD}


# ---------------------------------------------------------------------------
# 재표본 선택 (§8-9): N마다 하나의 seed 0 흐름, 칸 층화 비복원, 모든 후보가 공유
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "per_cell"), [(24, 4), (48, 8), (96, 16), (204, 34)])
def test_resample_positions_are_stratified_without_replacement(n: int, per_cell: int) -> None:
    table = paired("dev_b", [0] * 210, CD)
    groups = table.cell_indices()
    resamples = resample_positions(n, n_resample=30)
    assert len(resamples) == 30
    for positions in resamples:
        assert len(positions) == n == len(set(positions))
        for cell, group in zip(CELLS, groups, strict=True):
            assert sum(1 for p in positions if table.cells[p] == cell) == per_cell
            assert all(p in group for p in positions if table.cells[p] == cell)


def test_resample_positions_must_split_evenly_into_cells() -> None:
    with pytest.raises(ValueError, match="6"):
        resample_positions(25, n_resample=1)


def test_resample_positions_restart_from_seed_0_and_differ_between_resamples() -> None:
    a = resample_positions(24, n_resample=20)
    assert a == resample_positions(24, n_resample=20)
    assert len({tuple(p) for p in a}) == 20


def test_all_candidates_share_the_same_resamples() -> None:
    """공통 난수(§8-9): 같은 표를 가진 두 후보는 재표본마다 판정이 같다."""
    a = candidate("a", 1, bias=0.1)
    twin = BudgetCandidate("twin", a.dev_a, a.dev_b, a.oracle)
    decisions = budget_decisions([a, twin], DELTAS, ns=(24,), n_resample=50)
    for per_resample in decisions[24]:
        assert per_resample["a"] == per_resample["twin"]


def test_budget_decisions_match_the_gate_functions_on_explicit_subtables() -> None:
    """재표본 표에 C2·C3·C4를 그대로 적용한 것과 같다. C4는 재표본마다 seed 0으로 새로 시작한다."""
    c = candidate("c", 2, bias=0.15)
    decisions = budget_decisions([c], DELTAS, ns=(24, 96), n_resample=15)
    for n in (24, 96):
        for positions, per_resample in zip(
            resample_positions(n, n_resample=15), decisions[n], strict=True
        ):
            for m in METRICS:
                t = subtable(c.dev_b[m], positions)
                d = DELTAS[(m, n)]
                assert per_resample["c"][("C2", m)] == gate_c2(t).accepted
                assert per_resample["c"][("C3", m)] == gate_c3(t, d).accepted
                assert per_resample["c"][("C4", m)] == gate_c4(t, d).accepted


def test_subtable_keeps_cell_grouping_and_values() -> None:
    t = paired("dev_b", [1, -1, 0] * 70, CD)
    positions = resample_positions(24, n_resample=1)[0]
    s = subtable(t, positions)
    assert s.split == "dev_b" and len(s.cases) == 24
    assert [len(g) for g in s.cell_indices()] == [4] * 6
    assert s.deltas == tuple(t.deltas[p] for p in positions)


def test_c4_with_precomputed_blocks_equals_c4_without() -> None:
    """같은 칸 크기·seed 0이면 색인이 같으므로, 미리 만든 색인을 넘겨도 값이 같다."""
    c = candidate("c", 3, bias=0.2)
    positions = resample_positions(48, n_resample=1)[0]
    t = subtable(c.dev_b[CD], positions)
    blocks = stratified_bootstrap_indices([8] * 6)
    assert gate_c4(t, Fraction(1, 16), blocks=blocks) == gate_c4(t, Fraction(1, 16))
    qs = (Fraction(1, 40), Fraction(39, 40))
    assert bootstrap_quantiles(t, qs, blocks=blocks) == bootstrap_quantiles(t, qs)


def test_precomputed_blocks_must_match_the_cell_sizes() -> None:
    c = candidate("c", 3)
    t = subtable(c.dev_b[CD], resample_positions(48, n_resample=1)[0])
    with pytest.raises(ValueError, match="blocks"):
        gate_c4(t, Fraction(1, 16), blocks=stratified_bootstrap_indices([4] * 6))


# ---------------------------------------------------------------------------
# 재표본 평균과 분모 0 (§8-10)
# ---------------------------------------------------------------------------


def test_summarize_values_drops_undefined_resamples_and_counts_them() -> None:
    s = summarize_values([Fraction(1, 2), None, Fraction(0), None, Fraction(1)])
    assert s.mean == Fraction(1, 2)
    assert (s.undefined, s.n_resample) == (2, 5)
    # 정의된 [0, 1/2, 1]: 위치 1/20 → 1/40, 위치 39/20 → 1/2 + (1/2)(19/20) = 39/40
    assert (s.low, s.high) == (Fraction(1, 40), Fraction(39, 40))


def test_summarize_values_all_undefined() -> None:
    s = summarize_values([None, None])
    assert (s.mean, s.low, s.high, s.undefined) == (None, None, None, 2)


def test_budget_points_count_resamples_where_a_gate_accepts_nothing() -> None:
    """한 후보만 있고 C4가 일부 재표본에서만 채택 → Harmful 등은 그 재표본에서만 정의된다."""
    c = candidate("c", 4, bias=0.12, delta=Fraction(3, 20))
    decisions = budget_decisions([c], DELTAS, ns=(24,), n_resample=200)
    accepted = [d["c"][("C4", CD)] for d in decisions[24]]
    assert 0 < sum(accepted) < 200
    point = budget_points(decisions[24], [c])[("C4", CD)]
    assert point.metrics["useful"].undefined == 200 - sum(accepted)
    assert point.metrics["useful"].mean == 1  # 정의된 재표본에서는 언제나 유용(Δ = 3/20)
    assert point.metrics["gain"].undefined == 200 - sum(accepted)
    assert point.acceptance["c"] == Fraction(sum(accepted), 200)
    assert point.metrics["false_rejection"].undefined == 0  # 분모는 유용 후보 1개로 늘 정의됨


def test_budget_points_cover_c2_c3_c4_for_both_metrics() -> None:
    c = candidate("c", 5)
    decisions = budget_decisions([c], DELTAS, ns=(24,), n_resample=5)
    points = budget_points(decisions[24], [c])
    assert sorted(points) == sorted((g, m) for g in ("C2", "C3", "C4") for m in METRICS)
    assert set(points[("C2", CD)].metrics) == {
        "harmful",
        "unsupported",
        "useful",
        "gain",
        "false_rejection",
        "local_harm",
    }


def test_budget_decisions_are_deterministic() -> None:
    cands = [candidate("a", 6, bias=0.1), candidate("b", 7)]
    assert budget_decisions(cands, DELTAS, ns=(24, 48), n_resample=20) == budget_decisions(
        cands, DELTAS, ns=(24, 48), n_resample=20
    )


# ---------------------------------------------------------------------------
# RQ2 민감도 (§8-12): Dev-A 칸마다 복원 추출, bad 18·18, dep 3·3, traffic 9·9
# ---------------------------------------------------------------------------


def test_sensitivity_resample_cell_composition() -> None:
    table = paired("dev_a", [0] * 60, CD)
    resamples = sensitivity_positions(n_resample=100)
    assert len(resamples) == 100
    family_totals: Counter[str] = Counter()
    for positions in resamples:
        cells = Counter(table.cells[p] for p in positions)
        assert [cells[c] for c in CELLS] == [18, 18, 3, 3, 9, 9]
        family_totals.update(c[0] for c in (table.cells[p] for p in positions))
    assert family_totals == {
        "bad_deployment": 3600,
        "traffic_spike": 1800,
        "dependency_failure": 600,
    }
    assert any(len(set(p)) < 60 for p in resamples)  # 복원 추출


def test_sensitivity_resamples_are_shared_and_seeded() -> None:
    assert sensitivity_positions(n_resample=10) == sensitivity_positions(n_resample=10)
    assert len({tuple(p) for p in sensitivity_positions(n_resample=10)}) == 10


def test_sensitivity_decisions_recompute_c1_on_the_resample() -> None:
    c = candidate("c", 8, bias=0.1)
    decisions = sensitivity_decisions([c], n_resample=20)
    for positions, per_resample in zip(
        sensitivity_positions(n_resample=20), decisions, strict=True
    ):
        for m in METRICS:
            assert per_resample["c"][("C1", m)] == gate_c1(subtable(c.dev_a[m], positions)).accepted


def test_sensitivity_rates_average_c1_harmful_and_local_harm() -> None:
    harmful_c = candidate("h", 9, bias=0.3, delta=Fraction(-3, 20))
    useful_c = candidate("u", 10, bias=0.3, delta=Fraction(3, 20))
    decisions = sensitivity_decisions([harmful_c, useful_c], n_resample=50)
    rates = sensitivity_rates(decisions, [harmful_c, useful_c], CD)
    assert set(rates) == {"harmful", "local_harm"}
    per = []
    for d in decisions:
        accepted = [n for n in ("h", "u") if d[n][("C1", CD)]]
        per.append(Fraction(sum(n == "h" for n in accepted), len(accepted)) if accepted else None)
    assert rates["harmful"] == summarize_values(per)


def test_each_n_uses_its_own_delta() -> None:
    """δ(N)은 그 N의 값을 쓴다. N=24는 δ = -1(언제나 넘음), N=96은 δ = 1(넘을 수 없음)."""
    c = candidate("c", 11)
    deltas = {(m, 24): Fraction(-1) for m in METRICS} | {(m, 96): Fraction(1) for m in METRICS}
    decisions = budget_decisions([c], deltas, ns=(24, 96), n_resample=10)
    assert all(d["c"][("C3", m)] for d in decisions[24] for m in METRICS)
    assert not any(d["c"][("C3", m)] or d["c"][("C4", m)] for d in decisions[96] for m in METRICS)
