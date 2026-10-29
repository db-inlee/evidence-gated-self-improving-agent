"""P0 환경 기록과 스모크 실행 (Ticket 008 §4 P0).

데이터가 필요한 테스트는 `~/.cache/alfworld`(또는 ALFWORLD_DATA)가 없으면 건너뛴다.
"""

import json
import os
from pathlib import Path

import pytest

from alfworld_pilot.p0 import (
    SPLITS,
    count_games,
    dir_bytes,
    environment_record,
    smoke_run,
)

DATA = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))
needs_data = pytest.mark.skipif(not (DATA / "json_2.1.1").exists(), reason="needs ALFWorld data")


def fake_cache(tmp_path: Path) -> Path:
    root = tmp_path / "alfworld"
    for split, games in (("train", 3), ("valid_seen", 1), ("valid_unseen", 2), ("valid_train", 0)):
        (root / "json_2.1.1" / split).mkdir(parents=True)
        for i in range(games):
            trial = root / "json_2.1.1" / split / f"task-{i}" / "trial_T1"
            trial.mkdir(parents=True)
            (trial / "game.tw-pddl").write_text("x" * 10)
            (trial / "traj_data.json").write_text("{}")
    (root / "detectors").mkdir()
    (root / "detectors" / "w.pth").write_bytes(b"0" * 100)
    (root / "logic").mkdir()
    (root / "logic" / "alfred.pddl").write_text("p")
    return root


def test_dir_bytes_sums_every_file(tmp_path: Path) -> None:
    root = fake_cache(tmp_path)
    assert dir_bytes(root / "detectors") == 100
    assert dir_bytes(root / "json_2.1.1" / "valid_seen") == 10 + 2


def test_count_games_counts_game_files_per_split(tmp_path: Path) -> None:
    root = fake_cache(tmp_path)
    assert SPLITS == ("train", "valid_seen", "valid_unseen", "valid_train")
    assert count_games(root / "json_2.1.1") == {
        "train": 3,
        "valid_seen": 1,
        "valid_unseen": 2,
        "valid_train": 0,
    }


def test_environment_record_has_versions_sizes_counts_and_no_home_path(tmp_path: Path) -> None:
    root = fake_cache(tmp_path)
    record = environment_record(root, install_route="arm64-native")
    assert record["install_route"] == "arm64-native"
    assert record["python"]["version"].startswith("3.12")
    assert record["python"]["machine"] in {"arm64", "x86_64"}
    assert set(record["packages"]) >= {"alfworld", "textworld", "fast-downward-textworld"}
    assert record["data"]["bytes"]["detectors"] == 100
    assert record["data"]["games"]["train"] == 3
    assert record["data"]["location"] == root.as_posix()  # 임시 폴더는 홈 밖이라 그대로
    assert str(Path.home()) not in json.dumps(record)


@needs_data
def test_smoke_run_plays_one_text_game_to_the_end_with_the_handcoded_expert() -> None:
    games = sorted(
        (DATA / "json_2.1.1" / "train").glob("pick_heat_then_place_in_recep-*/*/game.tw-pddl")
    )
    result = smoke_run(games[0], max_steps=50)
    assert result["done"] and result["won"], result
    assert 0 < result["steps"] <= 50
    assert result["expert_plan_available"]
    assert result["admissible_commands_available"]
    assert str(Path.home()) not in json.dumps(result)


def test_paths_under_home_are_written_with_a_tilde() -> None:
    from alfworld_pilot.p0 import _display

    assert _display(Path.home() / ".cache" / "alfworld") == "~/.cache/alfworld"
    assert _display(Path("/opt/data")) == "/opt/data"
