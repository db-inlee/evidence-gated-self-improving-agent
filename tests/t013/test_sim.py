"""Ticket 013 3단계 모의 부품(`evidencegate.t013.sim`). LLM·runs/ 없이 돈다."""

import json
import math
import shutil
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from evidencegate.exact import clopper_pearson
from evidencegate.t013 import sim

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHA = Fraction(1, 20)


# 동결 계획 ------------------------------------------------------------------------------------


def test_load_plan_checks_the_pinned_sha256() -> None:
    plan = sim.load_plan(REPO_ROOT)
    assert plan["status"] == "frozen" and plan["tag"] == "t013-sim-plan"
    assert sim.PLAN_SHA256.startswith("8c97f641")


def test_load_plan_refuses_a_changed_plan(tmp_path: Path) -> None:
    (tmp_path / "plans").mkdir()
    for name in (sim.PLAN_PATH.name, sim.PLAN_PATH.name + ".sha256"):
        shutil.copy(REPO_ROOT / "plans" / name, tmp_path / "plans" / name)
    path = tmp_path / sim.PLAN_PATH
    path.write_bytes(path.read_bytes().replace(b'"tau": 0.05', b'"tau": 0.10'))
    with pytest.raises(ValueError, match="sha256"):
        sim.load_plan(tmp_path)


# Tango score 하한 --------------------------------------------------------------------------------


def test_tango_lower_without_discordant_pairs_matches_the_closed_form() -> None:
    # w = l = 0이면 Z(Δ0) = sqrt(n |Δ0| / (1 - |Δ0|)), 한쪽 z에서 |Δ0| = z^2 / (n + z^2)
    z = NormalDist().inv_cdf(0.95)
    assert sim.tango_lower(0, 0, 70, ALPHA) == pytest.approx(-(z * z) / (70 + z * z), abs=1e-9)


def test_tango_lower_sits_below_the_estimate_and_falls_with_losses() -> None:
    est = (10 - 3) / 70
    low = sim.tango_lower(10, 3, 70, ALPHA)
    assert -1 < low < est
    assert sim.tango_lower(10, 6, 70, ALPHA) < low
    assert sim.tango_lower(40, 12, 280, ALPHA) > low  # 같은 비율(4배), 더 큰 n


def test_tango_score_is_zero_at_the_point_estimate() -> None:
    assert sim.tango_z(10, 3, 70, Fraction(7, 70)) == pytest.approx(0.0, abs=1e-12)


# 정확 조건부 하한 -------------------------------------------------------------------------------


def test_exact_conditional_lower_uses_the_one_sided_clopper_pearson_bound() -> None:
    q_low = clopper_pearson(9, 12, 2 * ALPHA)
    assert q_low is not None
    expected = (12 / 70) * (2 * q_low[0] - 1)
    assert sim.exact_conditional_lower(9, 3, 70, ALPHA) == pytest.approx(expected)
    assert sim.exact_conditional_lower(0, 0, 70, ALPHA) == 0.0  # 계획 식 그대로(d = 0)


# SAGE와 신용형 ----------------------------------------------------------------------------------


def test_sage_commits_only_with_gain_a_small_p_and_a_capped_regression_rate() -> None:
    # λ = 1: p = P[Bin(9, 1/2) >= 8] = 10/512
    assert sim.sage_p(8, 1, 1) == Fraction(10, 512)
    assert sim.sage_accept(8, 1, 100, lam=1, tau=0.05, alpha=ALPHA)
    assert not sim.sage_accept(8, 1, 10, lam=1, tau=0.05, alpha=ALPHA)  # rho = 1/10 > tau
    assert not sim.sage_accept(6, 3, 100, lam=2, tau=0.05, alpha=ALPHA)  # Δ_λ = 0
    assert not sim.sage_accept(0, 0, 100, lam=1, tau=0.05, alpha=ALPHA)  # m = 0이면 p = 1


def test_sage_lambda_two_uses_the_two_thirds_threshold() -> None:
    assert sim.sage_p(10, 1, 2) == sum(
        Fraction(math.comb(11, k)) * Fraction(2, 3) ** k * Fraction(1, 3) ** (11 - k)
        for k in range(10, 12)
    )


def test_sage_cell_cap_applies_in_every_cell() -> None:
    assert sim.sage_cells_ok([0, 1, 0], [20, 20, 0], tau=0.05)
    assert not sim.sage_cells_ok([0, 2, 0], [20, 20, 0], tau=0.05)


def test_credit_type_needs_a_positive_wald_bound_and_few_new_errors() -> None:
    z = NormalDist().inv_cdf(0.975)
    w, loss, n, n_s = 30, 1, 420, 300
    g = (w - loss) / n
    var = (w + loss) / n**2 - (w - loss) ** 2 / n**3
    assert (g - z * math.sqrt(var) > 0) and loss / n_s <= 0.005
    assert sim.credit_accept(w, loss, n, n_s, eps=0.005)
    assert not sim.credit_accept(30, 2, 420, 300, eps=0.005)  # 2/300 > 0.005
    assert not sim.credit_accept(0, 0, 420, 300, eps=0.005)


# MC SE -----------------------------------------------------------------------------------------


def test_mc_se_is_the_pool_ratio_formula() -> None:
    a, n = [1, 3, 0, 2], [10, 10, 10, 10]
    p, se = sim.mc_se(a, n)
    assert p == pytest.approx(6 / 40)
    r = 4
    expected = (
        math.sqrt(r / (r - 1) * sum((ai - p * ni) ** 2 for ai, ni in zip(a, n, strict=True))) / 40
    )
    assert se == pytest.approx(expected)


# 귀무 라벨률 관문 --------------------------------------------------------------------------------


def flat_p_hat() -> dict[tuple[str, str], list[float]]:
    from evidencegate.eval_data import CELLS

    return {cell: [0.0, 0.5, 1.0, 1.0] for cell in CELLS}


def test_null_gate_is_deterministic_and_counts_per_pool() -> None:
    one = sim.null_label_gate(flat_p_hat(), pools=3, n_null=20, n_boot=50, seed=0)
    two = sim.null_label_gate(flat_p_hat(), pools=3, n_null=20, n_boot=50, seed=0)
    assert one == two
    assert len(one["per_pool"]["local_harm"]) == 3 and one["n"] == 60
    for label in ("local_harm", "severe"):
        assert 0 <= one[label]["rate"] <= 1 and one[label]["se"] >= 0


def test_null_gate_compares_with_the_registered_rates_at_two_root_two_se() -> None:
    verdict = sim.gate_verdict(
        {"local_harm": {"rate": 0.17, "se": 0.002}, "severe": {"rate": 0.0005, "se": 0.0001}}
    )
    assert verdict["local_harm"]["bound"] == pytest.approx(2 * math.sqrt(2) * 0.002)
    assert verdict["local_harm"]["reproduced"] is (abs(0.17 - 0.167205) <= 2 * math.sqrt(2) * 0.002)
    assert verdict["severe"]["reproduced"] is True
    assert verdict["reproduced"] is (verdict["local_harm"]["reproduced"] and True)


def test_registered_rates_are_read_from_the_committed_power_file() -> None:
    data = json.loads((REPO_ROOT / sim.POWER_009_FINAL).read_text())
    assert {
        "local_harm": data["h"]["null_label_rate"]["local_harm"],
        "severe": data["h"]["null_label_rate"]["severe"],
    } == sim.REGISTERED_NULL


def test_percentile_cell_lower_is_the_alpha_quantile_of_cell_bootstrap_means() -> None:
    d = np.asarray([1, 0, 0, -1, 1, 0, 0, 0], dtype=np.int64)
    rng_blocks = np.random.Generator(np.random.PCG64(0)).integers(0, 8, size=(200, 8))
    got = sim.percentile_cell_lower(d, rng_blocks, ALPHA)
    means = np.sort(d[rng_blocks].mean(axis=1))
    pos = 0.05 * (200 - 1)
    lo = math.floor(pos)
    assert got == pytest.approx(means[lo] + (means[lo + 1] - means[lo]) * (pos - lo))


def test_the_v1_oracle_pin_comes_from_the_committed_run_summary() -> None:
    summary = json.loads((REPO_ROOT / sim.V1_RUN_SUMMARY).read_text())
    assert summary["inputs"][str(sim.V1_ORACLE_S0)] == sim.V1_ORACLE_S0_SHA256
