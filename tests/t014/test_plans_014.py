"""Ticket 014 분할과 단계별 실행 계획(사전 등록 2·8절). LLM 없음."""

import json
from pathlib import Path

import pytest

from evidencegate.eval_data import CELLS, cell_of
from evidencegate.run_plan import load_plan
from evidencegate.t009.splits import SPLITS as SPLITS_009
from evidencegate.t014 import plans, splits
from evidencegate.t014.settings import CALIBRATION_REPEATS, ORACLE_REPEATS

ROOT = Path(__file__).resolve().parents[2]


def test_splits_use_the_preregistered_seed_ranges() -> None:
    for name, first, per_cell, size in (
        ("t014_dev_a", 1_100_000, 10, 60),
        ("t014_calibration", 1_200_000, 150, 900),
        ("t014_dev_b", 1_300_000, 150, 900),
        ("t014_oracle", 1_400_000, 100, 600),
    ):
        cases = splits.split_cases(name)
        assert len(cases) == len(set(cases)) == size
        seeds = sorted({c[2] for c in cases})
        assert seeds == list(range(first, first + per_cell))
        assert {cell_of(c) for c in cases} == set(CELLS)
    with pytest.raises(ValueError, match="unknown"):
        splits.split_cases("t009_dev_b")


def test_new_seed_ranges_do_not_overlap_earlier_ones() -> None:
    ours = {c[2] for name in splits.SPLITS for c in splits.split_cases(name)}
    earlier = {o + i for o, n in SPLITS_009.values() for i in range(n)}
    assert not ours & earlier
    assert min(ours) > 900_069  # 009 보정 끝(900,020 + 49)보다 뒤


def test_cell_cases_split_a_split_into_the_six_cells_in_order() -> None:
    parts = [splits.cell_cases("t014_oracle", cell) for cell in CELLS]
    assert [len(p) for p in parts] == [100] * 6
    assert sorted(c for p in parts for c in p) == splits.split_cases("t014_oracle")
    assert all(p == sorted(p) for p in parts)


def test_stage0_and_stage1_plans_are_cell_blocks_of_s0() -> None:
    p0 = plans.build_plan(0, ROOT)
    assert len(p0) == 6 and {b.split for b in p0} == {"t014_dev_a"}
    assert {b.repeat for b in p0} == {0} and all(len(b.cases) == 10 for b in p0)
    p1 = plans.build_plan(1, ROOT)
    assert len(p1) == 5 * 6 and all(len(b.cases) == 150 for b in p1)
    assert {(b.repeat, cell_of(b.cases[0])) for b in p1} == {
        (r, c) for r in CALIBRATION_REPEATS for c in CELLS
    }
    assert [b.index for b in p1] == list(range(30))
    assert {b.name for b in p0 + p1} == {"S0"}
    # 섞기: 정해진 순서(반복, 칸)와 다르고, 다시 만들면 같다
    assert [(b.repeat, b.cases[0]) for b in p1] != sorted((b.repeat, b.cases[0]) for b in p1)
    assert plans.build_plan(1, ROOT) == p1


def write_pool(root: Path, names: list[str]) -> None:
    from evidencegate.skills import load_skill

    s0 = load_skill(ROOT / "skills/S0/SKILL.md")
    entries = []
    for i, name in enumerate(names):
        folder = root / "skills/candidates-014" / name
        folder.mkdir(parents=True)
        text = s0.text + f"\n<!-- {name} -->\n"
        (folder / "SKILL.md").write_text(text)
        entries.append(
            {
                "name": name,
                "group": "proposer",
                "skill_id": load_skill(folder / "SKILL.md").skill_id,
                "skill_path": f"skills/candidates-014/{name}/SKILL.md",
                "call_index": i,
            }
        )
    (root / "skills/S0").mkdir(parents=True)
    (root / "skills/S0/SKILL.md").write_bytes((ROOT / "skills/S0/SKILL.md").read_bytes())
    path = root / "skills/candidates-014/POOL-014.json"
    path.write_text(json.dumps({"candidates": entries}))


def test_stage2_plan_is_pool_and_s0_by_cell_on_dev_b(tmp_path: Path) -> None:
    write_pool(tmp_path, ["cand-a", "cand-b"])
    p2 = plans.build_plan(2, tmp_path)
    assert len(p2) == 3 * 6 and {b.split for b in p2} == {"t014_dev_b"}
    assert {b.name for b in p2} == {"S0", "cand-a", "cand-b"} and {b.repeat for b in p2} == {0}
    assert {b.group for b in p2 if b.name != "S0"} == {"proposer"}


def test_stage2_plan_needs_the_frozen_pool(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="POOL-014"):
        plans.build_plan(2, tmp_path)


def test_stage4_plan_is_targets_and_s0_by_cell_and_repeat(tmp_path: Path) -> None:
    write_pool(tmp_path, ["cand-a", "cand-b", "cand-c"])
    targets = plans.Targets(proposer=("cand-c", "cand-a"), controls=False)
    p4 = plans.build_stage4_plan(targets, tmp_path)
    assert len(p4) == 3 * 6 * 3 and all(len(b.cases) == 100 for b in p4)
    assert {b.name for b in p4} == {"S0", "cand-a", "cand-c"}
    assert {b.repeat for b in p4} == set(ORACLE_REPEATS)


def test_stage4_plan_with_controls_adds_the_seven_controls(tmp_path: Path) -> None:
    write_pool(tmp_path, ["cand-a"])
    controls = ROOT / "skills/controls"
    for path in controls.glob("*/SKILL.md"):
        dest = tmp_path / "skills/controls" / path.parent.name / "SKILL.md"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(path.read_bytes())
    p4 = plans.build_stage4_plan(plans.Targets(proposer=("cand-a",), controls=True), tmp_path)
    names = {b.name for b in p4 if b.group == "control"}
    assert len(names) == 7 and len(p4) == (1 + 7 + 1) * 6 * 3


def test_write_plan_is_write_once_and_check_rebuilds(tmp_path: Path) -> None:
    (tmp_path / "skills/S0").mkdir(parents=True)
    (tmp_path / "skills/S0/SKILL.md").write_bytes((ROOT / "skills/S0/SKILL.md").read_bytes())
    sha = plans.write_plan(tmp_path, 1, plans.build_plan(1, tmp_path))
    path = tmp_path / "plans/ticket-014-stage1-plan.json"
    assert load_plan(path) == plans.build_plan(1, tmp_path)
    assert (path.with_name(path.name + ".sha256")).read_text().split()[0] == sha
    assert plans.check_plan(tmp_path, 1).ok
    with pytest.raises(FileExistsError, match="differs"):
        plans.write_plan(tmp_path, 1, plans.build_plan(0, tmp_path))
    path.write_text(path.read_text().replace('"repeat":4', '"repeat":3', 1))
    check = plans.check_plan(tmp_path, 1)
    assert not check.ok and "sha256" in check.detail


def test_build_plan_refuses_stages_without_episodes() -> None:
    with pytest.raises(ValueError, match="stage 3"):
        plans.build_plan(3, ROOT)


def test_stage4_plan_is_refused_without_oracle_targets(tmp_path: Path) -> None:
    write_pool(tmp_path, ["cand-a"])
    with pytest.raises(ValueError, match="no Oracle targets"):
        plans.build_stage4_plan(plans.Targets(proposer=(), controls=False), tmp_path)
