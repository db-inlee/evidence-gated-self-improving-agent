"""Ticket 013 2-prime 단계: 모의 계획을 모의 코드 전에 고정한다.

고정 파일은 `plans/ticket-013-sim-plan.json`(+ `.sha256`)이다.

고정 뒤 파일이 바뀌지 않았는지, David 결정(2026-11-25)이 계획에 그대로 들어 있는지 본다.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans" / "ticket-013-sim-plan.json"


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(PLAN.read_text(encoding="utf-8"))
    return data


def test_plan_matches_its_sha256() -> None:
    recorded = PLAN.with_name(PLAN.name + ".sha256").read_text().split()
    assert recorded == [hashlib.sha256(PLAN.read_bytes()).hexdigest(), PLAN.name]


def test_plan_is_frozen_with_nothing_left_open(plan: dict[str, Any]) -> None:
    assert plan["status"] == "frozen"
    assert plan["tag"] == "t013-sim-plan"
    assert plan["open_before_freeze"] == []
    assert "open" not in json.dumps(plan).lower().replace("open_before_freeze", "")


def test_cell_bounds_are_tango_then_exact_conditional(plan: dict[str, Any]) -> None:
    bounds = plan["cell_lower_bounds"]
    assert bounds["primary"] == "tango_score"
    assert bounds["secondary"] == "exact_conditional"
    assert bounds["exact_conditional"]["report_coverage"] is True
    assert any("Chan" in item for item in plan["limitations"])


def test_frozen_values_are_the_decided_ones(plan: dict[str, Any]) -> None:
    useful = plan["effects"]["useful_representative"]
    got = (useful["shift"], useful["baseline_rate_at_most"], useful["other_cells"])
    assert got == (0.10, 0.9, 0)
    assert plan["monte_carlo"]["pools_per_condition"] == 2000
    sage = plan["gates"]["SAGE"]
    assert (sage["lambda"], sage["tau"], sage["alpha"]) == ([1, 2], 0.05, 0.05)
    assert "held-out" in sage["tau_note"]
    assert plan["grid"] == {
        "n_c": [70, 150, 300, 600], "m_c": [0.05, 0.10], "m_c_note": plan["grid"]["m_c_note"],
        "alpha": 0.05, "cells": 6, "candidates": 40,
    }  # fmt: skip


def test_go_rule_uses_point_estimates_and_marks_borderline(plan: dict[str, Any]) -> None:
    decision = plan["decision"]
    assert decision["thresholds"] == {
        "false_accept_at_most": 0.05,
        "useful_joint_power_at_least": 0.7,
        "dev_b_episodes_at_most": 75000,
    }
    assert decision["borderline"]["mc_se_multiple"] == 2
    assert "not counted" in decision["borderline"]["rule"]
    assert "sqrt" in plan["monte_carlo"]["mc_se"]["formula"]


def test_loss_cell_has_a_realistic_and_a_worst_placement(plan: dict[str, Any]) -> None:
    placements = plan["effects"]["loss_cell_placements"]
    assert {"realistic", "worst"} <= set(placements)
    assert "both placements" in placements["use"]
    assert "highest baseline mean rate" in placements["realistic"]
    assert "p (1 - p)" in placements["worst"]
    assert "CELLS" in placements["realistic"] and "CELLS" in placements["worst"]
    assert "loss_cell" not in plan["effects"]


def test_false_accept_is_the_maximum_over_scenarios_and_placements(plan: dict[str, Any]) -> None:
    rule = plan["decision"]["false_accept_definition"]
    assert rule["take"] == "maximum"
    assert rule["scenarios"] == [
        "null_all_cells",
        "no_global_one_cell_minus_m",
        "global_up_one_cell_minus_m",
        "global_up_one_cell_minus_2m",
    ]
    assert rule["placements"] == ["realistic", "worst"]
    assert "infeasible" in rule["infeasible"]


def test_limitations_name_es_and_the_stylized_baseline_b(plan: dict[str, Any]) -> None:
    text = " ".join(plan["limitations"])
    assert "ES is not simulated" in text
    assert "stylized" in text and "heterogeneity" in text
