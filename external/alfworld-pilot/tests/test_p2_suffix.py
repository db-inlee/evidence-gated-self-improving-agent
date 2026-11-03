"""H2용 observation_suffix 훅 (2026-11-03 결정 2, agent.py amendment).

- 보내는 요청의 **가장 최근 관측**에만 접미사를 붙인다. 이력·기록에는 원래 관측만 남는다.
- 기본값 None 경로는 golden(test_p2_diagnostics)이 수정 전과 같음을 확인한다.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.agent import NUDGE, Outcome, PolicySettings, run_episode

FIRST = "You are in the middle of a room.\n\nYour task is to: put a pen in shelf."


def call(command: str) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "act", "arguments": {"command": command}}}],
        }
    }


def text() -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": "hmm"}}


@dataclass
class Chat:
    replies: list[dict[str, Any]]
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(json.loads(json.dumps(body)))
        return self.replies.pop(0)


@dataclass
class Env:
    n: int = 0

    def reset(self) -> str:
        return FIRST

    def step(self, command: str) -> Outcome:
        self.n += 1
        won = command == "take pen 1 from shelf 2"
        return Outcome(f"obs {self.n}", won, won, True)


def suffix_from(env: Env) -> Any:
    return lambda: f"Available commands: list{env.n}"


def run(replies: list[dict[str, Any]], **kw: Any) -> tuple[Any, Chat]:
    env = Env()
    chat = Chat(replies)
    trace = run_episode(
        env, "# S", chat, seed=0, settings=PolicySettings(**kw), observation_suffix=suffix_from(env)
    )
    return trace, chat


def contents(body: dict[str, Any]) -> list[str]:
    return [m.get("content", "") for m in body["messages"]]


def test_suffix_goes_on_the_first_observation_before_any_step() -> None:
    _, chat = run([call("take pen 1 from shelf 2")])
    first = chat.bodies[0]["messages"][1]
    assert first == {"role": "user", "content": FIRST + "\n\nAvailable commands: list0"}


def test_suffix_goes_only_on_the_latest_observation() -> None:
    _, chat = run([call("look"), call("look"), call("take pen 1 from shelf 2")])
    third = contents(chat.bodies[2])
    assert third[1] == FIRST  # 이전 관측에는 붙지 않는다
    assert "obs 1" in third and "obs 2\n\nAvailable commands: list2" in third
    assert sum("Available commands" in c for c in third) == 1


def test_suffix_skips_the_nudge_and_lands_on_the_latest_observation() -> None:
    _, chat = run([call("look"), text(), call("take pen 1 from shelf 2")])
    third = contents(chat.bodies[2])
    assert third[-1] == NUDGE
    assert "obs 1\n\nAvailable commands: list1" in third


def test_trace_and_history_keep_raw_observations() -> None:
    trace, _ = run([call("look"), call("take pen 1 from shelf 2")])
    assert [s.observation for s in trace.steps] == ["obs 1", "obs 2"]
    assert trace.first_observation == FIRST


@pytest.mark.parametrize("budget", range(2000, 2600, 9))
def test_history_budget_leaves_room_for_the_suffix(budget: int) -> None:
    replies = [call("look") for _ in range(12)]
    trace, chat = run(replies, max_steps=12, history_budget_override=budget)
    for body in chat.bodies:
        assert sum(len(json.dumps(m)) for m in body["messages"]) <= budget
    assert trace.truncated_calls > 0


DATA = Path.home() / ".cache" / "alfworld" / "json_2.1.1"


@pytest.mark.skipif(not DATA.exists(), reason="needs ALFWorld data")
def test_real_env_exposes_admissible_commands_as_a_copy() -> None:
    from alfworld_pilot.agent import AlfworldEnv

    pool = json.loads((Path(__file__).parents[1] / "records" / "p1_pilot_pool.json").read_text())
    env = AlfworldEnv(DATA / pool["games"][0]["rel"] / "game.tw-pddl", max_steps=50)
    env.reset()
    commands = env.admissible_commands()
    assert "look" in commands
    commands.append("x")
    assert "x" not in env.admissible_commands()
    env.close()
