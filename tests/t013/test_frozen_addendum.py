"""Ticket 013 계획 부록을 본 모의 전에 고정한다.

고정 파일은 `plans/ticket-013-sim-plan-addendum.json`(+ `.sha256`)이다.

David 결정(2026-11-25, 보고서 2026-11-25-02 §5)이 부록에 그대로 들어 있는지 본다.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADDENDUM = ROOT / "plans" / "ticket-013-sim-plan-addendum.json"
PLAN = ROOT / "plans" / "ticket-013-sim-plan.json"


@pytest.fixture(scope="module")
def addendum() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    return data


def test_addendum_matches_its_sha256() -> None:
    recorded = ADDENDUM.with_name(ADDENDUM.name + ".sha256").read_text().split()
    assert recorded == [hashlib.sha256(ADDENDUM.read_bytes()).hexdigest(), ADDENDUM.name]


def test_addendum_points_at_the_frozen_plan(addendum: dict[str, Any]) -> None:
    assert addendum["status"] == "frozen" and addendum["tag"] == "t013-sim-addendum"
    target = addendum["addendum_to"]
    assert target["sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    assert target["tag"] == "t013-sim-plan"


def test_the_four_decisions_and_the_conventions_are_recorded(addendum: dict[str, Any]) -> None:
    d = addendum["decisions"]
    assert "whole case population" in d["true_effect"]["rule"]
    assert (
        "infeasible" in d["shift_solving"]["infeasible"]
        and "applied shift" in d["shift_solving"]["record"]
    )
    assert "only S0's Dev-B run" in d["s0_independent_variant"]
    coupling = d["seed_coupling"]
    assert (
        "correlation 0.5" in coupling["rule"]
        and "i-th L1 case and the i-th L2 case" in coupling["pairing"]
    )
    assert any("not used to change 0.5" in line for line in coupling["report"])
    assert set(addendum["conventions"]) == {
        "common_random_numbers", "oracle_label_uncertainty", "idea_group_sizes",
        "exact_conditional_d_zero",
    }  # fmt: skip
    assert "6, 4, 2, 2, 2 and 24 groups of 1" in addendum["conventions"]["idea_group_sizes"]


def test_execution_writes_results_once_and_checkpoints_outside_git(
    addendum: dict[str, Any],
) -> None:
    e = addendum["execution"]
    assert "runs/ticket013_sim/blocks/" in e["checkpoints"]
    assert "written once" in e["final_output"] and "never overwritten" in e["final_output"]
    assert "never to results/" in e["trial_mode"]
