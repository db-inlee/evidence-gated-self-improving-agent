"""하네스 진단 H1·H2 (exploratory diagnostic, not used for G1-G6; 2026-11-03 결정 3)."""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.agent import NUDGE, Outcome, PolicySettings, run_episode
from alfworld_pilot.diag import (
    DIAG_PER_TYPE,
    DiagWorker,
    diag_plan,
    diag_run_config,
    h1_settings,
)
from alfworld_pilot.runner import plan_episodes, select_games

ROOT = Path(__file__).parents[1]
POOL = json.loads((ROOT / "records" / "p1_pilot_pool.json").read_text())["games"]


def text() -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": "Task completed."}}


def act(command: str) -> dict[str, Any]:
    call = {"function": {"name": "act", "arguments": {"command": command}}}
    return {"message": {"role": "assistant", "content": "", "tool_calls": [call]}}


@dataclass
class Chat:
    replies: list[dict[str, Any]]
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        return self.replies.pop(0)


@dataclass
class Env:
    def reset(self) -> str:
        return "room"

    def step(self, command: str) -> Outcome:
        won = command == "take pen 1 from shelf 2"
        return Outcome("ok", won, won, True)


def test_h1_keeps_nudging_through_many_text_replies() -> None:
    chat = Chat([text()] * 5 + [act("take pen 1 from shelf 2")])
    trace = run_episode(Env(), "s", chat, seed=0, settings=h1_settings())
    assert trace.won and trace.text_turns == 5
    assert all(b["messages"][-1] == {"role": "user", "content": NUDGE} for b in chat.bodies[1:])


def test_h1_still_stops_at_the_call_and_step_caps() -> None:
    s = h1_settings()
    assert s.max_llm_calls == PolicySettings().max_llm_calls == 100
    assert s.max_steps == PolicySettings().max_steps == 50
    trace = run_episode(Env(), "s", Chat([text()] * 100), seed=0, settings=s)
    assert trace.stop_reason == "max_llm_calls" and trace.llm_calls == 100
    # 연속 텍스트 상한만 바뀌고 나머지 정책 설정은 S0와 같다
    a, b = h1_settings().config(), PolicySettings().config()
    assert a["stop"] != b["stop"]
    assert {k: v for k, v in a.items() if k != "stop"} == {
        k: v for k, v in b.items() if k != "stop"
    }


def test_plan_is_the_first_six_p4a_games_per_type_at_repeat_zero() -> None:
    plan = diag_plan(POOL)
    assert DIAG_PER_TYPE == 6 and len(plan) == 36
    assert plan == plan_episodes(select_games(POOL, 6), repeats=1)
    assert {e.repeat for e in plan} == {0}
    p4a = {e.key: e.llm_seed for e in plan_episodes(select_games(POOL, 12), repeats=3)}
    assert all(p4a[e.key] == e.llm_seed for e in plan)  # P4-a 반복 0과 짝


def test_run_config_labels_the_variant_and_is_not_used_for_gates() -> None:
    h1 = diag_run_config("H1", skill_sha256="s" * 64, server={"x": 1})
    h2 = diag_run_config("H2", skill_sha256="s" * 64, server={"x": 1})
    for c in (h1, h2):
        assert c["diagnostic"]["label"] == "exploratory diagnostic, not used for G1-G6"
    assert (
        h1["diagnostic"]["variant"] == "H1" and h1["diagnostic"]["command_list_in_prompt"] is False
    )
    assert h2["diagnostic"]["command_list_in_prompt"] is True
    assert h1["policy"]["stop"]["max_consecutive_text_turns"] == 101
    assert h2["policy"] == PolicySettings().config()
    with pytest.raises(ValueError):
        diag_run_config("H3", skill_sha256="s", server={})


def test_workers_are_picklable_and_validate_the_variant() -> None:
    w = DiagWorker("H2", "skill", "/x")
    # spawn 작업자로 보낼 수 있는지 확인한다.
    # 같은 프로세스에서 만든 자기 객체만 되돌린다(외부 데이터 아님).
    assert pickle.loads(pickle.dumps(w)) == w
    with pytest.raises(ValueError):
        DiagWorker("H9", "skill", "/x")


DATA = Path.home() / ".cache" / "alfworld" / "json_2.1.1"


@pytest.mark.skipif(not DATA.exists(), reason="needs ALFWorld data")
def test_h2_worker_puts_the_real_command_list_on_the_latest_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import alfworld_pilot.diag as diag

    bodies: list[dict[str, Any]] = []

    class FakeOllama:
        def __init__(self, base_url: str) -> None:
            self.replies = [act("look"), text(), text()]

        def chat(self, body: dict[str, Any]) -> dict[str, Any]:
            bodies.append(json.loads(json.dumps(body)))
            return self.replies.pop(0)

    monkeypatch.setattr(diag, "OllamaChat", FakeOllama)
    plan = diag_plan(POOL)
    trace = DiagWorker("H2", "skill", str(DATA))(plan[0])
    first = bodies[0]["messages"][1]["content"]
    assert "\n\nAvailable commands: " in first and "look" in first.split("Available commands: ")[1]
    second = [m["content"] for m in bodies[1]["messages"]]
    assert sum("Available commands: " in c for c in second) == 1
    assert "Available commands" not in trace.steps[0].observation
