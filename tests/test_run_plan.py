"""Ticket 005 실행 계획 (§2, §4): 12,540 에피소드, 블록 63개, 블록 순서, 해시."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from evidencegate.analysis import load_pool
from evidencegate.eval_constants import PILOT_BLOCKS
from evidencegate.eval_data import split_cases
from evidencegate.run_plan import (
    Block,
    build_plan,
    canonical_blocks,
    main,
    plan_bytes,
    plan_sha256,
)
from evidencegate.skills import load_skill

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan() -> tuple[Block, ...]:
    return build_plan(load_pool(REPO_ROOT), REPO_ROOT)


def test_plan_has_12540_episodes_in_63_blocks(plan: tuple[Block, ...]) -> None:
    assert len(plan) == 63
    assert sum(len(b.cases) for b in plan) == 12_540


def test_block_composition(plan: tuple[Block, ...]) -> None:
    """후보 19 x 3 split(반복 0), S0 Dev-B 반복 0, S0 Oracle 반복 0~4. S0 Dev-A는 없다(완료)."""
    kinds = Counter((b.group, b.split, b.repeat) for b in plan)
    assert (
        kinds[("proposer", "dev_a", 0)]
        == kinds[("proposer", "dev_b", 0)]
        == kinds[("proposer", "oracle", 0)]
        == 12
    )
    assert (
        kinds[("control", "dev_a", 0)]
        == kinds[("control", "dev_b", 0)]
        == kinds[("control", "oracle", 0)]
        == 7
    )
    assert kinds[("s0", "dev_b", 0)] == 1
    assert [kinds[("s0", "oracle", r)] for r in range(5)] == [1] * 5
    assert ("s0", "dev_a", 0) not in kinds
    assert sum(kinds.values()) == 63


def test_every_block_covers_its_split_in_cell_and_seed_order(plan: tuple[Block, ...]) -> None:
    for b in plan:
        assert list(b.cases) == split_cases(b.split)


def test_pilot_blocks_come_first(plan: tuple[Block, ...]) -> None:
    assert PILOT_BLOCKS == (("cand-000", "dev_a"), ("mixed-scale-on-traffic", "dev_a"))
    assert [(b.name, b.split) for b in plan[:2]] == list(PILOT_BLOCKS)
    assert [b.index for b in plan] == list(range(63))


def test_remaining_order_is_random_random_0_over_the_canonical_list(
    plan: tuple[Block, ...],
) -> None:
    pool = load_pool(REPO_ROOT)
    rest = [k for k in canonical_blocks(pool) if (k[0], k[2]) not in PILOT_BLOCKS]
    random.Random(0).shuffle(rest)
    assert [(b.name, b.split, b.repeat) for b in plan[2:]] == [(n, s, r) for n, _, s, r in rest]


def test_canonical_order_is_candidates_by_split_then_s0() -> None:
    keys = canonical_blocks(load_pool(REPO_ROOT))
    assert keys[0] == ("cand-000", "proposer", "dev_a", 0)
    assert keys[1] == ("cand-000", "proposer", "dev_b", 0)
    assert keys[-6:] == [("S0", "s0", "dev_b", 0)] + [("S0", "s0", "oracle", r) for r in range(5)]


def test_skill_paths_and_ids_match_the_frozen_files(plan: tuple[Block, ...]) -> None:
    for b in plan:
        assert load_skill(REPO_ROOT / b.skill_path).skill_id == b.skill_id
    s0 = [b for b in plan if b.group == "s0"]
    assert {b.skill_id for b in s0} == {"bf65e569ae37"}
    assert {b.skill_path for b in s0} == {"skills/S0/SKILL.md"}


def test_plan_is_deterministic_and_hashed(plan: tuple[Block, ...]) -> None:
    again = build_plan(load_pool(REPO_ROOT), REPO_ROOT)
    assert plan_bytes(again) == plan_bytes(plan)
    assert plan_sha256(plan) == hashlib.sha256(plan_bytes(plan)).hexdigest()
    data = json.loads(plan_bytes(plan))
    assert data["episodes"] == 12_540 and len(data["blocks"]) == 63
    assert data["blocks"][0]["cases"][0] == list(split_cases("dev_a")[0])


def test_plan_build_refuses_a_skill_file_that_does_not_match(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(REPO_ROOT / "skills", tmp_path / "skills")
    path = tmp_path / "skills" / "controls" / "pos-timing-only" / "SKILL.md"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="skill_id"):
        build_plan(load_pool(REPO_ROOT), tmp_path)


def test_main_writes_plan_and_hash_and_refuses_a_different_existing_plan(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"
    assert main(["--out", str(out), "--root", str(REPO_ROOT)]) == 0
    digest = (tmp_path / "plan.json.sha256").read_text().split()[0]
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    assert main(["--out", str(out), "--root", str(REPO_ROOT)]) == 0  # 같은 계획이면 그대로 통과
    out.write_text("{}")
    assert main(["--out", str(out), "--root", str(REPO_ROOT)]) == 1
