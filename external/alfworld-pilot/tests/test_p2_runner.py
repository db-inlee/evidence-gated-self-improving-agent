"""P2 실행기: 이어서 실행, 중단 규칙, 원칙 7 점검(시작과 블록마다), 기록 형식, 변환기 자리."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.agent import EpisodeTrace, Outcome, PolicySettings, StepTrace
from alfworld_pilot.power import PowerCheckError
from alfworld_pilot.runner import (
    BLOCK_SIZE,
    Episode,
    RunStopped,
    plan_episodes,
    run_plan,
    to_v1_like,
)


def games(n_per_type: int = 2) -> list[dict[str, Any]]:
    types = ("pick_and_place_simple", "pick_heat_then_place_in_recep")
    return [
        {"rel": f"train/{t}-X-{i}/trial_{i}", "task_type": t, "game_sha256": "a" * 64}
        for t in types
        for i in range(n_per_type)
    ]


def test_plan_is_games_by_repeats_in_a_fixed_order() -> None:
    plan = plan_episodes(games(), repeats=2)
    assert len(plan) == 8
    assert plan[0] == Episode(
        rel="train/pick_and_place_simple-X-0/trial_0",
        task_type="pick_and_place_simple",
        repeat=0,
        llm_seed=plan[0].llm_seed,
    )
    assert plan == plan_episodes(list(reversed(games())), repeats=2)
    assert plan[0].key == "train/pick_and_place_simple-X-0/trial_0#0"


@dataclass
class Fakes:
    power_calls: int = 0
    power_fail_at: int | None = None
    episodes_run: list[str] = field(default_factory=list)
    crash_at: int | None = None

    def check(self) -> None:
        self.power_calls += 1
        if self.power_fail_at is not None and self.power_calls >= self.power_fail_at:
            raise PowerCheckError("lid: closed")

    def episode(self, ep: Episode) -> EpisodeTrace:
        if self.crash_at is not None and len(self.episodes_run) == self.crash_at:
            raise ConnectionError("ollama down")
        self.episodes_run.append(ep.key)
        step = StepTrace(command="look", observation="ok", admissible=True)
        return EpisodeTrace(
            steps=(step,),
            won=ep.repeat == 0,
            stop_reason="won" if ep.repeat == 0 else "max_steps",
            first_observation="room",
            format_errors=0,
            text_turns=0,
            llm_calls=1,
            prompt_tokens=(10,),
            completion_tokens=(2,),
            truncated_calls=0,
            max_omitted=0,
            wall_s=0.5,
        )


def run(tmp_path: Path, fakes: Fakes, plan: Sequence[Episode]) -> list[dict[str, Any]]:
    out = tmp_path / "episodes.jsonl"
    try:
        run_plan(
            plan,
            out,
            run_episode=fakes.episode,
            check_power=fakes.check,
            run_config={"policy": PolicySettings().config(), "skill_sha256": "s" * 64},
        )
    finally:
        pass
    return [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []


def test_records_every_episode_with_config_and_raw_steps(tmp_path: Path) -> None:
    plan = plan_episodes(games(), repeats=1)
    records = run(tmp_path, Fakes(), plan)
    assert [r["key"] for r in records] == [e.key for e in plan]
    r = records[0]
    assert r["won"] is True and r["steps"] == [
        {"command": "look", "observation": "ok", "admissible": True}
    ]
    assert r["llm_seed"] == plan[0].llm_seed
    assert r["config"]["skill_sha256"] == "s" * 64
    assert r["config"]["policy"]["instructions_sha256"]


def test_power_is_checked_at_start_and_before_every_block(tmp_path: Path) -> None:
    plan = plan_episodes(games(n_per_type=BLOCK_SIZE), repeats=1)  # 2 블록
    fakes = Fakes()
    run(tmp_path, fakes, plan)
    assert fakes.power_calls == 1 + len(plan) // BLOCK_SIZE


def test_power_failure_stops_before_the_next_episode_and_resume_continues(
    tmp_path: Path,
) -> None:
    plan = plan_episodes(games(n_per_type=BLOCK_SIZE), repeats=1)
    fakes = Fakes(power_fail_at=2)
    with pytest.raises(RunStopped, match="lid"):
        run(tmp_path, fakes, plan)
    done = [
        json.loads(line)["key"] for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert done == [e.key for e in plan[:BLOCK_SIZE]]
    resumed = Fakes()
    records = run(tmp_path, resumed, plan)
    assert resumed.episodes_run == [e.key for e in plan[BLOCK_SIZE:]]
    assert [r["key"] for r in records] == [e.key for e in plan]


def test_power_failure_at_start_runs_nothing(tmp_path: Path) -> None:
    fakes = Fakes(power_fail_at=1)
    with pytest.raises(RunStopped):
        run(tmp_path, fakes, plan_episodes(games(), repeats=1))
    assert fakes.episodes_run == []


def test_server_error_stops_without_a_partial_record(tmp_path: Path) -> None:
    plan = plan_episodes(games(), repeats=1)
    fakes = Fakes(crash_at=2)
    with pytest.raises(RunStopped, match="ollama down"):
        run(tmp_path, fakes, plan)
    lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_resume_refuses_a_different_config(tmp_path: Path) -> None:
    plan = plan_episodes(games(), repeats=1)
    run(tmp_path, Fakes(), plan[:1])
    with pytest.raises(RunStopped, match="config"):
        run_plan(
            plan,
            tmp_path / "episodes.jsonl",
            run_episode=Fakes().episode,
            check_power=Fakes().check,
            run_config={"policy": PolicySettings().config(), "skill_sha256": "t" * 64},
        )


def test_to_v1_like_maps_cells_games_and_success() -> None:
    record = {
        "key": "train/a/t#1",
        "rel": "train/a/t",
        "task_type": "pick_heat_then_place_in_recep",
        "repeat": 1,
        "won": True,
        "e_a": False,
    }
    assert to_v1_like(record) == {
        "family": "pick_heat_then_place_in_recep",
        "incident": "train/a/t",
        "repeat": 1,
        "correct": True,
        "evidence_safe": False,
    }


def test_outcome_type_is_shared() -> None:
    assert Outcome(observation="o", won=False, done=False, admissible=True).observation == "o"


def test_main_refuses_without_hash_seed_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from alfworld_pilot.runner import main

    monkeypatch.setenv("PYTHONHASHSEED", "1")
    out = tmp_path / "e.jsonl"
    assert main(["--per-type", "1", "--out", str(out)]) == 2
    assert "PYTHONHASHSEED=0" in capsys.readouterr().err
    assert not out.exists()


def test_s0_skill_is_short_english_and_hashed() -> None:
    skill = Path(__file__).parents[1] / "skills" / "S0" / "SKILL.md"
    text = skill.read_text()
    assert text.isascii() and len(text.split()) < 120
    for hidden in ("admissible", "expert", "facts"):
        assert hidden not in text.lower()
