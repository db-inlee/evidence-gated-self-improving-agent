"""Ticket 013 본 모의 실행기(`evidencegate.t013.sim` run).

부록 확인, 블록 체크포인트, 한 번 쓰기를 본다.
"""

import json
import shutil
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import pytest

from evidencegate.t013 import sim
from evidencegate.t013 import sim_core as sc

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return sim.load_plan(REPO_ROOT)


@pytest.fixture(scope="module")
def addendum() -> dict[str, Any]:
    return sim.load_addendum(REPO_ROOT)


# 계획·부록과 코드 상수 ----------------------------------------------------------------------------


def test_core_constants_are_the_frozen_values(
    plan: dict[str, Any], addendum: dict[str, Any]
) -> None:
    gates, effects, decision = plan["gates"], plan["effects"], plan["decision"]
    assert float(sc.ALPHA) == plan["grid"]["alpha"]
    assert list(sc.SAGE_LAMBDAS) == gates["SAGE"]["lambda"] and gates["SAGE"]["tau"] == sc.SAGE_TAU
    assert gates["credit_type"]["epsilon"] == sc.CREDIT_EPS
    useful = effects["useful_representative"]
    assert (
        float(sc.USEFUL_SHIFT) == useful["shift"]
        and float(sc.ROOM_AT_MOST) == useful["baseline_rate_at_most"]
    )
    t = decision["thresholds"]
    assert (
        t["false_accept_at_most"], t["useful_joint_power_at_least"], t["dev_b_episodes_at_most"],
    ) == (sc.FALSE_ACCEPT_AT_MOST, sc.USEFUL_POWER_AT_LEAST, sc.EPISODES_AT_MOST)  # fmt: skip
    assert decision["borderline"]["mc_se_multiple"] == sc.BORDERLINE_SE
    assert list(sc.INADMISSIBLE_SCENARIOS) == decision["false_accept_definition"]["scenarios"]
    assert list(sc.PLACEMENTS) == decision["false_accept_definition"]["placements"]
    assert "0.5" in addendum["decisions"]["seed_coupling"]["rule"] and sc.COUPLING_RHO == 0.5
    s = sim.main_settings(plan)
    assert (s.n_cs, s.pools, s.n_boot, s.candidates) == ((70, 150, 300, 600), 2000, 2000, 40)
    assert s.margins == (F(1, 20), F(1, 10))


def test_load_addendum_checks_both_hashes(tmp_path: Path) -> None:
    (tmp_path / "plans").mkdir()
    for name in ("ticket-013-sim-plan.json", "ticket-013-sim-plan-addendum.json"):
        for suffix in ("", ".sha256"):
            shutil.copy(REPO_ROOT / "plans" / (name + suffix), tmp_path / "plans" / (name + suffix))
    sim.load_addendum(tmp_path)
    path = tmp_path / sim.ADDENDUM_PATH
    path.write_bytes(path.read_bytes().replace(b"0.5", b"0.6", 1))
    with pytest.raises(ValueError, match="sha256"):
        sim.load_addendum(tmp_path)


# 작은 실행 -------------------------------------------------------------------------------------


def tiny_inputs() -> sim.Inputs:
    cells = (
        (F(1), F(1)),
        (F(1), F(4, 5)),
        (F(1, 2), F(1, 2)),
        (F(1), F(1)),
        (F(0), F(1, 5)),
        (F(1, 5), F(2, 5)),
    )
    return sim.Inputs(
        baselines=(sc.Baseline("a", cells), sc.Baseline("b", cells)),
        v1_effects=((F(0),) * 4 + (F(1, 10), F(1, 10)), (F(-1, 20),) + (F(0),) * 5),
        group_sizes=(3, 2, 1, 1, 1),
        calibration_pairs_009b={"bad_deployment": {"n11": 3, "n10": 1, "n01": 1, "n00": 3}},
        gate={"verdict": {"reproduced": True}},
    )


def tiny_settings() -> sim.RunSettings:
    return sim.RunSettings(
        n_cs=(4,), margins=(F(1, 20),), pools=3, n_boot=20, candidates=8, workers=1
    )


def test_run_writes_blocks_then_the_final_file_once(tmp_path: Path) -> None:
    out = sim.run_simulation(tiny_inputs(), tiny_settings(), blocks_dir=tmp_path / "blocks",
                             final=tmp_path / "sim.json")  # fmt: skip
    assert sorted(p.name for p in (tmp_path / "blocks").iterdir()) == [
        "a_4_coupled.json", "a_4_none.json", "b_4_coupled.json", "b_4_none.json",
    ]  # fmt: skip
    data = json.loads((tmp_path / "sim.json").read_text())
    assert data == json.loads(json.dumps(out))
    sidecar = (tmp_path / "sim.json.sha256").read_text().split()
    assert sidecar[1] == "sim.json"
    assert data["settings"]["plan_sha256"] == sim.PLAN_SHA256
    assert data["settings"]["addendum_sha256"] == sim.ADDENDUM_SHA256
    with pytest.raises(FileExistsError):
        sim.run_simulation(tiny_inputs(), tiny_settings(), blocks_dir=tmp_path / "blocks",
                           final=tmp_path / "sim.json")  # fmt: skip


def test_a_restart_reuses_finished_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = tmp_path / "blocks"
    sim.run_blocks(tiny_inputs(), tiny_settings(), blocks_dir=blocks)
    calls: list[int] = []
    real = sc.evaluate_pool

    def counted(setup: sc.Setup, pool: int) -> dict[str, Any]:
        calls.append(pool)
        return real(setup, pool)

    monkeypatch.setattr(sc, "evaluate_pool", counted)
    sim.run_blocks(tiny_inputs(), tiny_settings(), blocks_dir=blocks)
    assert calls == []


def test_a_checkpoint_from_another_plan_is_refused(tmp_path: Path) -> None:
    blocks = tmp_path / "blocks"
    sim.run_blocks(tiny_inputs(), tiny_settings(), blocks_dir=blocks)
    path = blocks / "a_4_none.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data | {"addendum_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checkpoint"):
        sim.run_blocks(tiny_inputs(), tiny_settings(), blocks_dir=blocks)


def test_final_file_needs_every_block(tmp_path: Path) -> None:
    blocks = tmp_path / "blocks"
    sim.run_blocks(tiny_inputs(), tiny_settings(), blocks_dir=blocks)
    (blocks / "b_4_coupled.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        sim.assemble(tiny_inputs(), tiny_settings(), blocks_dir=blocks)


def test_results_report_shifts_correlations_and_a_decision(tmp_path: Path) -> None:
    out = sim.run_simulation(tiny_inputs(), tiny_settings(), blocks_dir=tmp_path / "blocks",
                             final=tmp_path / "sim.json")  # fmt: skip
    scen = out["baselines"]["a"]["scenarios"]["1/20"]["global_up_one_cell_minus_m|worst"]
    assert scen["feasible"] and set(scen["shifts"]) == set(sc.CELL_NAMES)
    assert out["correlations"]["009b_calibration_descriptive"]["bad_deployment"] == pytest.approx(
        0.5
    )
    assert set(out["correlations"]["simulated"]) == {
        "a_4_none",
        "a_4_coupled",
        "b_4_none",
        "b_4_coupled",
    }
    combos = out["decision"]["a"]
    assert {c["gate"] for c in combos} == set(sc.GO_GATES)
    assert out["decision"]["go"] in (True, False)
    assert out["gate"] == {"verdict": {"reproduced": True}}


def test_false_accept_is_the_worst_scenario_and_placement() -> None:
    rows: dict[str, dict[str, Any]] = {
        "null_all_cells|-": {"estimate": 0.01, "se": 0.001, "eligible": 10},
        "no_global_one_cell_minus_m|realistic": {"estimate": 0.02, "se": 0.001, "eligible": 10},
        "no_global_one_cell_minus_m|worst": {"estimate": 0.08, "se": 0.002, "eligible": 10},
        "global_up_one_cell_minus_m|worst": {"estimate": None, "se": None, "eligible": 0},
    }
    got = sim.worst_false_accept(rows)
    assert got is not None
    assert got["estimate"] == 0.08 and got["scenario"] == "no_global_one_cell_minus_m|worst"
