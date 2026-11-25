"""Ticket 013 본 모의의 계산부(`evidencegate.t013.sim_core`). 작은 설정만 쓴다."""

from fractions import Fraction as F

import numpy as np
import pytest

from evidencegate.exact import stratified_bootstrap_indices
from evidencegate.t009 import power_sim as ps
from evidencegate.t013 import sim_core as sc

M05 = F(1, 20)


# 이동량 풀기(정확) -------------------------------------------------------------------------


def test_realized_effect_clips_at_zero_and_one() -> None:
    rates = (F(0), F(1, 2), F(1))
    assert sc.realized(rates, F(1, 4)) == F(1, 6)
    assert sc.realized(rates, F(-1, 4)) == F(-1, 6)
    assert sc.realized(rates, F(0)) == 0


def test_solve_cell_shift_meets_the_target_exactly_or_reports_none() -> None:
    rates = (F(0), F(1, 2), F(1))
    s = sc.solve_shift([rates], F(1, 6))
    assert s is not None and s == F(1, 4) and sc.realized(rates, s) == F(1, 6)
    down = sc.solve_shift([rates], F(-1, 2))  # 모든 성공 제거
    assert down is not None and sc.realized(rates, down) == F(-1, 2)
    assert sc.solve_shift([rates], F(3, 5)) is None  # 1 - p의 평균(1/2)을 넘음
    assert sc.solve_shift([rates], F(0)) == 0


def test_solve_common_shift_sums_over_cells() -> None:
    a, b = (F(9, 10), F(1)), (F(0), F(0))
    s = sc.solve_shift([a, b], F(1, 4))  # (realized_a + realized_b) = 1/4
    assert s is not None
    assert sc.realized(a, s) + sc.realized(b, s) == F(1, 4)


# 시나리오 ----------------------------------------------------------------------------------------


def toy_baseline() -> sc.Baseline:
    cells = (
        (F(1), F(1)),          # bad/L1: 1.0 (가장 높음, 현실 배치)
        (F(1), F(4, 5)),       # bad/L2: 0.9 (room 경계 포함)
        (F(1, 2), F(1, 2)),    # dep/L1: 0.5 (분산 최대, 최악 배치)
        (F(1), F(1)),          # dep/L2: 1.0
        (F(0), F(1, 5)),       # traffic/L1: 0.1
        (F(1, 5), F(2, 5)),    # traffic/L2: 0.3
    )  # fmt: skip
    return sc.Baseline("toy", cells)


def test_room_and_loss_cells_follow_the_plan_and_the_addendum() -> None:
    b = toy_baseline()
    assert sc.room_cells(b) == [1, 2, 4, 5]  # 평균 <= 0.9
    assert sc.loss_cell(b, "realistic") == 0  # 평균 최고, 동점이면 앞 칸
    assert sc.loss_cell(b, "worst") == 2  # p(1 - p) 최대


def test_scenarios_reach_their_targets_exactly() -> None:
    b = toy_baseline()
    by = {(s.name, s.placement): s for s in sc.build_scenarios(b, M05)}
    null = by[("null_all_cells", None)]
    assert null.feasible and null.deltas == (F(0),) * 6
    for placement in ("realistic", "worst"):
        s = by[("global_up_one_cell_minus_m", placement)]
        loss = sc.loss_cell(b, placement)
        assert s.feasible and s.deltas[loss] == -M05 and s.delta == F(1, 10)
        s2 = by[("global_up_one_cell_minus_2m", placement)]
        assert s2.deltas[loss] == -2 * M05 and s2.delta == F(1, 10)
        s0 = by[("no_global_one_cell_minus_m", placement)]
        assert s0.deltas[loss] == -M05 and s0.delta == 0
        others = [c for c in sc.room_cells(b) if c != loss]
        assert len({s.shifts[c] for c in others}) == 1  # 같은 이동량
    useful = by[("useful_representative", None)]
    assert useful.deltas == (F(0), F(1, 10), F(1, 10), F(0), F(1, 10), F(1, 10))


def test_an_unreachable_target_is_infeasible() -> None:
    b = sc.Baseline("tiny", ((F(1),),) * 4 + ((F(19, 20),),) * 2)  # room 없음
    by = {(s.name, s.placement): s for s in sc.build_scenarios(b, M05)}
    assert not by[("global_up_one_cell_minus_m", "realistic")].feasible
    assert not by[("useful_representative", None)].feasible


def test_inadmissible_and_useful_use_population_effects_with_the_boundary_in_the_null() -> None:
    assert sc.inadmissible((0,) * 6, F(0), M05)
    assert sc.inadmissible((F(1, 5), -M05, 0, 0, 0, 0), F(1, 40), M05)  # δ_c = -m는 귀무
    assert not sc.inadmissible((F(1, 5), F(-1, 25), 0, 0, 0, 0), F(1, 75), M05)
    assert sc.useful((F(1, 5), F(-1, 25), 0, 0, 0, 0), F(1, 75), M05)


# copula와 상관 -----------------------------------------------------------------------------------


def test_copula_only_transforms_the_l2_cells_and_couples_them() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 20000))
    none = sc.uniforms(x, coupled=False)
    both = sc.uniforms(x, coupled=True)
    assert np.array_equal(none[[0, 2, 4]], both[[0, 2, 4]])
    corr = np.corrcoef(both[0], both[1])[0, 1]
    assert 0.4 < corr < 0.55  # Gaussian 0.5의 균등 상관은 약 0.48
    assert abs(np.corrcoef(none[0], none[1])[0, 1]) < 0.03


def test_phi_from_pair_counts() -> None:
    assert sc.phi({"n11": 10, "n10": 0, "n01": 0, "n00": 10}) == pytest.approx(1.0)
    assert sc.phi({"n11": 5, "n10": 5, "n01": 5, "n00": 5}) == pytest.approx(0.0)
    assert sc.phi({"n11": 0, "n10": 0, "n01": 0, "n00": 9}) is None  # 한쪽이 상수


# 빠른 C4와 칸 하한 --------------------------------------------------------------------------------


def test_fast_c4_matches_power_sim() -> None:
    rng = np.random.default_rng(1)
    n, k = 12, 5
    d = rng.integers(-1, 2, size=(6 * n, k)).astype(np.int64)
    blocks = stratified_bootstrap_indices([n] * 6, 200, 0)
    counts = sc.block_counts(blocks)
    delta = F(1, 30)
    assert np.array_equal(sc.c4_fast(d, counts, delta), ps.c4_parts(d, blocks, delta)[0])


def test_percentile_cell_lower_fast_matches_the_scalar_version() -> None:
    from evidencegate.t013 import sim

    rng = np.random.default_rng(2)
    n, k = 10, 3
    d = rng.integers(-1, 2, size=(6 * n, k)).astype(np.int64)
    blocks = stratified_bootstrap_indices([n] * 6, 100, 0)
    counts = sc.block_counts(blocks)
    got = sc.percentile_lowers(d, counts, n, M05)
    for c in range(6):
        for j in range(k):
            want = sim.percentile_cell_lower(d[c * n : (c + 1) * n, j], blocks[c], M05)
            assert got[c, j] == pytest.approx(want)


# 풀 하나 ---------------------------------------------------------------------------------------


def small_setup() -> sc.Setup:
    b = toy_baseline()
    return sc.Setup(
        baseline=b,
        baseline_index=0,
        n_c=6,
        coupled=False,
        n_boot=40,
        candidates=8,
        margins=(M05, F(1, 10)),
        v1_effects=((F(0),) * 4 + (F(1, 10), F(1, 10)), (F(-1, 20),) + (F(0),) * 5),
        group_sizes=(3, 2, 1, 1, 1),
        demo=True,
    )


def test_evaluate_pool_is_deterministic_and_fills_every_key() -> None:
    setup = small_setup()
    one = sc.evaluate_pool(setup, pool=3)
    two = sc.evaluate_pool(setup, pool=3)
    assert one == two
    keys = set(one["gates"])
    for m in ("1/20", "1/10"):
        assert ("1/20", "null_all_cells", "-", "main") in {k[:4] for k in keys}
        assert any(k[0] == m and k[1] == "empirical" and k[3] == "idea_groups" for k in keys)
    row = next(iter(one["gates"].values()))
    assert set(row) == {"n", "inadmissible", "accepted", "accepted_inadmissible", "useful",
                        "accepted_useful", "gain"}  # fmt: skip
    assert set(one["calibration"]) == {"h", "delta", "pairs"}
    assert any(k[4] == "C6_IUT_percentile" for k in keys)  # 시연 조건


def test_evaluate_pool_counts_are_consistent() -> None:
    out = sc.evaluate_pool(small_setup(), pool=0)
    for row in out["gates"].values():
        assert row["accepted_inadmissible"] <= min(row["accepted"], row["inadmissible"])
        assert row["accepted_useful"] <= min(row["accepted"], row["useful"])
        assert row["n"] == 8
    for cov in out["coverage"].values():
        assert 0 <= cov["tango_covered"] <= cov["cells"] and cov["exact_d_zero"] <= cov["cells"]


# 집계와 판정 ------------------------------------------------------------------------------------


def test_aggregate_uses_the_pool_ratio_se() -> None:
    from evidencegate.t013 import sim

    pools = [{"n": 10, "inadmissible": 10, "accepted_inadmissible": a} for a in (0, 1, 0, 2)]
    got = sc.rate(pools, "accepted_inadmissible", "inadmissible")
    assert (got["estimate"], got["se"]) == sim.mc_se([0, 1, 0, 2], [10] * 4)


def decision_rows(fa: float, fa_se: float, power: float, power_se: float) -> dict[str, object]:
    return {
        "false_accept": {"estimate": fa, "se": fa_se},
        "useful_power": {"estimate": power, "se": power_se},
    }


def test_decide_marks_borderline_and_needs_all_three_thresholds() -> None:
    ok = sc.judge(decision_rows(0.01, 0.002, 0.9, 0.01), n_c=300)
    assert ok["meets"] and not ok["borderline"]
    near = sc.judge(decision_rows(0.047, 0.002, 0.9, 0.01), n_c=300)
    assert near["meets"] and near["borderline"]  # |0.047 - 0.05| < 2 x 0.002
    weak = sc.judge(decision_rows(0.01, 0.002, 0.6, 0.01), n_c=150)
    assert not weak["meets"]
    costly = sc.judge(decision_rows(0.01, 0.002, 0.9, 0.01), n_c=600)
    assert costly["episodes"] == 147600 and not costly["meets"]
