"""P1 결정성과 수량 (Ticket 008 §4 P1). LLM은 쓰지 않는다.

순수 함수는 가짜 데이터로, 환경 점검은 실제 데이터로(없으면 건너뜀) 시험한다.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.p1 import (
    PAPER_TABLE1,
    SIX_TYPES,
    GameInfo,
    compare_with_paper,
    count_table,
    determinism_check,
    expert_run,
    location_targets,
    main,
    overlap,
    pilot_pool,
    scan_split,
)

DATA = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))
JSON = DATA / "json_2.1.1"
needs_data = pytest.mark.skipif(not JSON.exists(), reason="needs ALFWorld data")


def add_trial(
    root: Path,
    split: str,
    folder: str,
    trial: str,
    task_type: str,
    *,
    game: bool = True,
    solvable: bool = True,
    content: str = "g",
) -> None:
    d = root / split / folder / trial
    d.mkdir(parents=True)
    (d / "traj_data.json").write_text(json.dumps({"task_type": task_type}))
    if game:
        (d / "game.tw-pddl").write_text(json.dumps({"solvable": solvable, "x": content}))


def fake_json(tmp_path: Path) -> Path:
    root = tmp_path / "json_2.1.1"
    add_trial(
        root,
        "train",
        "pick_heat_then_place_in_recep-Apple-1",
        "t1",
        "pick_heat_then_place_in_recep",
    )
    add_trial(
        root,
        "train",
        "pick_heat_then_place_in_recep-Apple-1",
        "t2",
        "pick_heat_then_place_in_recep",
    )
    add_trial(
        root, "train", "look_at_obj_in_light-Book-2", "t1", "look_at_obj_in_light", solvable=False
    )
    add_trial(
        root,
        "train",
        "pick_and_place_with_movable_recep-Pen-Cup-3",
        "t1",
        "pick_and_place_with_movable_recep",
        game=False,
    )
    add_trial(
        root,
        "valid_train",
        "pick_heat_then_place_in_recep-Apple-1",
        "t1",
        "pick_heat_then_place_in_recep",
    )
    add_trial(
        root,
        "valid_train",
        "pick_two_obj_and_place-Key-4",
        "t9",
        "pick_two_obj_and_place",
        content="other",
    )
    return root


def test_six_types_are_alfworld_task_types_in_id_order() -> None:
    assert SIX_TYPES == (
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    )
    assert sum(PAPER_TABLE1["train"].values()) == 3553
    assert sum(PAPER_TABLE1["valid_seen"].values()) == 140
    assert sum(PAPER_TABLE1["valid_unseen"].values()) == 134


def test_scan_reads_task_type_game_presence_solvability_and_hashes(tmp_path: Path) -> None:
    root = fake_json(tmp_path)
    infos = scan_split(root, "train")
    assert [i.rel for i in infos] == [
        "train/look_at_obj_in_light-Book-2/t1",
        "train/pick_and_place_with_movable_recep-Pen-Cup-3/t1",
        "train/pick_heat_then_place_in_recep-Apple-1/t1",
        "train/pick_heat_then_place_in_recep-Apple-1/t2",
    ]
    heat = infos[2]
    assert heat.task_type == "pick_heat_then_place_in_recep" and heat.has_game and heat.solvable
    assert heat.playable and len(heat.game_sha256 or "") == 64
    assert not infos[0].playable  # 풀 수 없는 게임
    assert not infos[1].has_game and not infos[1].playable  # movable: 게임 파일 없음


def test_count_table_counts_playable_six_types_and_reports_the_rest(tmp_path: Path) -> None:
    root = fake_json(tmp_path)
    table = count_table({s: scan_split(root, s) for s in ("train", "valid_train")})
    assert table["train"]["playable"]["pick_heat_then_place_in_recep"] == 2
    assert table["train"]["playable"]["look_at_obj_in_light"] == 0
    assert table["train"]["unsolvable"] == {"look_at_obj_in_light": 1}
    assert table["train"]["no_game_file"] == {"pick_and_place_with_movable_recep": 1}
    assert table["train"]["trials"] == 4


def test_compare_with_paper_marks_cells_that_differ() -> None:
    table: dict[str, Any] = {
        split: {"playable": dict(PAPER_TABLE1[split])}
        for split in ("train", "valid_seen", "valid_unseen")
    }
    assert compare_with_paper(table) == []
    table["train"]["playable"]["pick_heat_then_place_in_recep"] = 458
    assert compare_with_paper(table) == [
        {"split": "train", "task_type": "pick_heat_then_place_in_recep", "paper": 459, "data": 458}
    ]


def test_overlap_by_path_and_by_game_hash(tmp_path: Path) -> None:
    root = fake_json(tmp_path)
    result = overlap(scan_split(root, "train"), scan_split(root, "valid_train"))
    assert result["by_relative_path"] == ["pick_heat_then_place_in_recep-Apple-1/t1"]
    assert result["by_game_sha256"] == ["pick_heat_then_place_in_recep-Apple-1/t1"]
    assert result["valid_train_games"] == 2


def info(rel: str, task: str, sha: str = "a") -> GameInfo:
    return GameInfo(rel, task, task, True, True, sha * 64, "b" * 64)


def test_pilot_pool_is_a_seeded_sample_per_type_from_playable_train_games() -> None:
    games = [info(f"train/{t}-{i}/t", t) for t in SIX_TYPES for i in range(40)]
    pool = pilot_pool(games, per_type=5, seed=0)
    assert len(pool) == 30
    for t in SIX_TYPES:
        assert sum(1 for g in pool if g.task_type == t) == 5
    assert pool == pilot_pool(list(reversed(games)), per_type=5, seed=0)  # 입력 순서와 무관
    assert pool != pilot_pool(games, per_type=5, seed=1)
    with pytest.raises(ValueError, match="only"):
        pilot_pool(games[:3], per_type=5, seed=0)


def test_pilot_pool_refuses_games_outside_train() -> None:
    with pytest.raises(ValueError, match="train"):
        pilot_pool([info("valid_seen/x/t", SIX_TYPES[0])], per_type=1, seed=0)


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("put apple 1 in/on countertop 2", "countertop 2"),
        ("move apple 1 to countertop 2", "countertop 2"),  # alfworld 0.4.2의 놓기 형식
        ("heat apple 1 with microwave 1", "microwave 1"),
        ("cool mug 2 with fridge 1", "fridge 1"),
        ("clean fork 1 with sinkbasin 1", "sinkbasin 1"),
        ("take apple 1 from countertop 2", "countertop 2"),
        ("open fridge 1", "fridge 1"),
        ("close fridge 1", "fridge 1"),
        ("use desklamp 1", "desklamp 1"),
        ("go to fridge 1", None),
        ("look", None),
        ("inventory", None),
    ],
)
def test_location_targets_parse_the_receptacle_a_command_acts_on(
    command: str, target: str | None
) -> None:
    assert location_targets(command) == target


@needs_data
def test_expert_run_and_determinism_on_two_real_games() -> None:
    games = sorted(JSON.glob("train/pick_heat_then_place_in_recep-*/*/game.tw-pddl"))[:2]
    run = expert_run(games[0])
    assert run["won"] and 0 < run["steps"] <= 150
    assert run["fallback_steps"] == 0  # 전문가 계획이 빈 걸음 수(첫 가능한 명령으로 대체)
    assert set(run["infos_keys"]) >= {"admissible_commands", "won", "extra.expert_plan"}
    check = determinism_check(games, resets=3)
    assert [g["same_first"] and g["same_replay"] for g in check] == [True, True]


@needs_data
def test_expert_run_is_reproducible_with_a_seed() -> None:
    # handcoded 전문가는 전역 random을 쓴다(시드 없음). 같은 시드면 같은 행동 열이어야 한다.
    game = sorted(JSON.glob("train/pick_two_obj_and_place-Bread-None-Fridge-*/*/game.tw-pddl"))[0]
    runs = [expert_run(game, seed=0)["actions"] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_main_refuses_to_run_without_a_fixed_hash_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 전문가는 집합 순회 순서에도 기대므로 기록은 PYTHONHASHSEED=0으로만 만든다.
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    assert main(["--out-dir", str(tmp_path)]) == 2
    assert "PYTHONHASHSEED=0" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
