"""P4 게임 선택 옵션 (2026-11-03 결정 1): --task-type, --offset, --list.

- 기존 --per-type 선택 결과는 그대로다.
- --task-type T --offset 12 --per-type 12 는 그 유형의 경로순 13~24번째다.
- llm_seed 규칙은 그대로다.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.agent import derive_llm_seed
from alfworld_pilot.p1 import SIX_TYPES
from alfworld_pilot.runner import first_per_type, main, plan_episodes, select_games

ROOT = Path(__file__).parents[1]
POOL: list[dict[str, Any]] = json.loads((ROOT / "records" / "p1_pilot_pool.json").read_text())[
    "games"
]
FIXED = json.loads((ROOT / "fixed" / "p3_fixed.json").read_text())


def reference_first(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """수정 전 first_per_type와 같은 규칙을 따로 쓴 것."""
    out = []
    for t in sorted(
        {g["task_type"] for g in pool},
        key=lambda t: min(g["rel"] for g in pool if g["task_type"] == t),
    ):
        out += sorted((g for g in pool if g["task_type"] == t), key=lambda g: g["rel"])[:n]
    return out


@pytest.mark.parametrize("n", [1, 2, 12, 30])
def test_per_type_selection_is_unchanged(n: int) -> None:
    assert select_games(POOL, n) == reference_first(POOL, n)
    assert first_per_type(POOL, n) == select_games(POOL, n)


def test_p3_fixed_episodes_are_unchanged() -> None:
    plan = plan_episodes(select_games(POOL, 2), repeats=1)
    assert [{"key": e.key, "llm_seed": e.llm_seed} for e in plan] == FIXED["p3"]["episodes"]


def test_task_type_with_offset_picks_the_13th_to_24th_games_of_that_type() -> None:
    t = "pick_heat_then_place_in_recep"
    chosen = select_games(POOL, 12, task_type=t, offset=12)
    ordered = sorted((g for g in POOL if g["task_type"] == t), key=lambda g: g["rel"])
    assert chosen == ordered[12:24]
    assert not {g["rel"] for g in chosen} & {g["rel"] for g in select_games(POOL, 12)}


def test_offset_without_task_type_applies_to_every_type() -> None:
    chosen = select_games(POOL, 3, offset=2)
    assert len(chosen) == 18
    for t in SIX_TYPES:
        ordered = sorted((g for g in POOL if g["task_type"] == t), key=lambda g: g["rel"])
        assert [g for g in chosen if g["task_type"] == t] == ordered[2:5]


def test_seed_rule_is_unchanged_for_selected_games() -> None:
    plan = plan_episodes(select_games(POOL, 12, task_type=SIX_TYPES[3], offset=12), repeats=3)
    assert len(plan) == 36
    assert all(e.llm_seed == derive_llm_seed(e.rel, e.repeat) for e in plan)


def test_bad_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown task type"):
        select_games(POOL, 1, task_type="pick_and_place_with_movable_recep")
    with pytest.raises(ValueError, match="only"):
        select_games(POOL, 12, task_type=SIX_TYPES[0], offset=25)
    with pytest.raises(ValueError):
        select_games(POOL, 1, offset=-1)


def test_list_prints_the_plan_without_contacting_the_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.chdir(ROOT)
    out = tmp_path / "e.jsonl"
    argv = ["--task-type", SIX_TYPES[3], "--offset", "12", "--per-type", "12", "--repeats", "3"]
    assert main([*argv, "--out", str(out), "--list"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 36
    first = json.loads(lines[0])
    assert set(first) == {"key", "task_type", "llm_seed"}
    assert not out.exists()


def test_selection_does_not_depend_on_pool_order() -> None:
    import random

    shuffled = list(POOL)
    random.Random(0).shuffle(shuffled)
    assert select_games(shuffled, 12) == select_games(POOL, 12)
    t = SIX_TYPES[3]
    assert select_games(shuffled, 12, task_type=t, offset=12) == select_games(
        POOL, 12, task_type=t, offset=12
    )
