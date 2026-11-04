"""Ticket 009 단계별 실행 계획(사전 등록 1절, 티켓 §4). 블록 섞기와 sha256 동결은 v1 방식."""

import json
import random
from pathlib import Path

import pytest

from evidencegate.eval_constants import PLAN_ORDER_SEED, S0_SKILL_ID
from evidencegate.run_plan import load_plan, plan_bytes
from evidencegate.t009.run_plan import (
    PHASES,
    PoolEntry,
    build_calibration_plan,
    build_dev_a_plan,
    build_main_plan,
    canonical_main_blocks,
    main,
    plan_path,
)
from evidencegate.t009.splits import split_cases

ROOT = Path(__file__).resolve().parents[2]


def test_calibration_plan_is_s0_times_five_repeats_on_the_addition() -> None:
    plan = build_calibration_plan(ROOT)
    assert sorted(b.repeat for b in plan) == [0, 1, 2, 3, 4]
    assert {(b.name, b.group, b.skill_id, b.split) for b in plan} == {
        ("S0", "s0", S0_SKILL_ID, "t009_calibration")
    }
    assert all(list(b.cases) == split_cases("t009_calibration") for b in plan)
    assert sum(len(b.cases) for b in plan) == 1500
    assert [b.index for b in plan] == list(range(5))
    expected = list(range(5))
    random.Random(PLAN_ORDER_SEED).shuffle(expected)
    assert [b.repeat for b in plan] == expected  # v1과 같은 섞기 규칙


def test_dev_a_plan_is_one_s0_block() -> None:
    (block,) = build_dev_a_plan(ROOT)
    assert (block.name, block.split, block.repeat, len(block.cases)) == ("S0", "t009_dev_a", 0, 60)


POOL = [
    PoolEntry("cand-a", "proposer", S0_SKILL_ID, "skills/S0/SKILL.md"),
    PoolEntry("cand-b", "proposer", S0_SKILL_ID, "skills/S0/SKILL.md"),
    PoolEntry("ctrl-x", "control", S0_SKILL_ID, "skills/S0/SKILL.md"),
]


def test_main_plan_runs_controls_on_oracle_only() -> None:
    keys = canonical_main_blocks(POOL)
    assert ("cand-a", "proposer", "t009_dev_b", 0) in keys
    assert ("cand-a", "proposer", "t009_oracle", 0) in keys
    assert ("ctrl-x", "control", "t009_oracle", 0) in keys
    assert ("ctrl-x", "control", "t009_dev_b", 0) not in keys
    assert ("S0", "s0", "t009_dev_b", 0) in keys
    assert [k[3] for k in keys if k[0] == "S0" and k[2] == "t009_oracle"] == [0, 1, 2, 3, 4]
    plan = build_main_plan(POOL, ROOT)
    episodes = sum(len(b.cases) for b in plan)
    assert episodes == 2 * (420 + 300) + 300 + 420 + 5 * 300
    assert plan == build_main_plan(POOL, ROOT)


def test_main_plan_refuses_a_skill_file_that_does_not_match() -> None:
    bad = [PoolEntry("cand-a", "proposer", "000000000000", "skills/S0/SKILL.md")]
    with pytest.raises(ValueError, match="skill_id"):
        build_main_plan(bad, ROOT)


def test_freeze_writes_the_plan_and_its_sha256_and_refuses_a_different_file(
    tmp_path: Path,
) -> None:
    out = tmp_path / "plan.json"
    assert main(["--phase", "calibration", "--out", str(out), "--root", str(ROOT)]) == 0
    data = out.read_bytes()
    assert data == plan_bytes(build_calibration_plan(ROOT))
    sha = (tmp_path / "plan.json.sha256").read_text().split()
    import hashlib

    assert sha[0] == hashlib.sha256(data).hexdigest() and sha[1] == "plan.json"
    assert load_plan(out) == build_calibration_plan(ROOT)
    assert main(["--phase", "calibration", "--out", str(out), "--root", str(ROOT)]) == 0  # 같음
    out.write_text(json.dumps({"blocks": []}))
    assert main(["--phase", "calibration", "--out", str(out), "--root", str(ROOT)]) == 1


def test_main_phase_cannot_be_frozen_before_the_pool_exists(tmp_path: Path) -> None:
    assert (
        main(["--phase", "main", "--out", str(tmp_path / "m.json"), "--root", str(tmp_path)]) == 1
    )


def test_phases_and_default_paths() -> None:
    assert PHASES == ("calibration", "dev_a", "main")
    assert plan_path("calibration") == Path("plans/ticket-009-calibration-plan.json")
