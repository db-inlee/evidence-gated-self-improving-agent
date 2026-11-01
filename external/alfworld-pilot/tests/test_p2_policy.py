"""P2 S0 에이전트 루프(LLM 없이 가짜 채팅으로). 정책 설정, 형식 오류, 이력 자르기, 비노출."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.agent import (
    INSTRUCTIONS,
    NUDGE,
    TEMPLATES,
    EpisodeTrace,
    Outcome,
    PolicySettings,
    build_messages,
    derive_llm_seed,
    instructions_sha256,
    run_episode,
)

SKILL = "# S0\nExplore, then act."
FIRST = "You are in the middle of a room.\n\nYour task is to: put a pen in shelf."


def call(command: Any) -> dict[str, Any]:
    return {"function": {"name": "act", "arguments": {"command": command}}}


def tool_reply(*calls: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": "", "tool_calls": list(calls)},
        "prompt_eval_count": 100,
        "eval_count": 10,
    }


def text_reply(text: str = "thinking") -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": 90,
        "eval_count": 5,
    }


@dataclass
class FakeChat:
    replies: list[dict[str, Any]]
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(json.loads(json.dumps(body)))
        return self.replies.pop(0)


@dataclass
class FakeEnv:
    win_on: str = "take pen 1 from shelf 2"
    commands: list[str] = field(default_factory=list)

    def reset(self) -> str:
        return FIRST

    def step(self, command: str) -> Outcome:
        self.commands.append(command)
        won = command == self.win_on
        obs = "You pick up the pen 1 from the shelf 2." if won else "Nothing happens."
        return Outcome(observation=obs, won=won, done=won, admissible=command != "fly")


def settings(**overrides: Any) -> PolicySettings:
    return PolicySettings(**overrides)


def test_fixed_policy_settings_and_config() -> None:
    s = PolicySettings()
    assert (s.model, s.think, s.temperature, s.num_ctx) == ("gpt-oss:20b", "low", 0.7, 8192)
    assert (s.max_steps, s.max_consecutive_format_errors, s.max_consecutive_text_turns) == (
        50,
        3,
        2,
    )
    config = s.config()
    assert config["instructions_sha256"] == instructions_sha256()
    assert config["history"]["rule"].startswith("keep system + first observation")
    assert config["command_list_in_prompt"] is False
    assert config["expert_plan_in_prompt"] is False


def test_seed_rule_is_stable_and_paired_by_game_and_repeat() -> None:
    a = derive_llm_seed("train/x/t1", 0)
    assert a == derive_llm_seed("train/x/t1", 0)
    assert a != derive_llm_seed("train/x/t1", 1)
    assert 0 <= a < 2**31


def test_templates_follow_alfworld_042_command_forms() -> None:
    for form in (
        "go to {recep}",
        "take {obj} from {recep}",
        "move {obj} to {recep}",
        "open {recep}",
        "close {recep}",
        "heat {obj} with {recep}",
        "cool {obj} with {recep}",
        "clean {obj} with {recep}",
        "use {obj}",
        "examine {x}",
        "look",
        "inventory",
    ):
        assert form in TEMPLATES
    assert "put " not in TEMPLATES and "in/on" not in TEMPLATES
    assert TEMPLATES in INSTRUCTIONS


def test_one_tool_call_is_one_step_and_a_win_ends_the_episode() -> None:
    chat = FakeChat(
        [tool_reply(call("go to shelf 2")), tool_reply(call("take pen 1 from shelf 2"))]
    )
    env = FakeEnv()
    trace = run_episode(env, SKILL, chat, seed=7, settings=settings())
    assert env.commands == ["go to shelf 2", "take pen 1 from shelf 2"]
    assert trace.won and trace.stop_reason == "won" and len(trace.steps) == 2
    assert trace.steps[1].observation == "You pick up the pen 1 from the shelf 2."
    body = chat.bodies[0]
    assert body["options"] == {"temperature": 0.7, "seed": 7, "num_ctx": 8192}
    assert body["think"] == "low" and body["model"] == "gpt-oss:20b"
    assert body["messages"][0] == {"role": "system", "content": INSTRUCTIONS + "\n\n" + SKILL}
    assert body["messages"][1] == {"role": "user", "content": FIRST}
    assert [t["function"]["name"] for t in body["tools"]] == ["act"]
    # 두 번째 요청에는 행동과 관측이 tool 메시지로 들어간다
    assert chat.bodies[1]["messages"][-1] == {
        "role": "tool",
        "tool_name": "act",
        "content": "Nothing happens.",
    }
    assert trace.llm_calls == 2 and trace.prompt_tokens == (100, 100)


def test_text_replies_get_a_nudge_and_two_in_a_row_stop() -> None:
    chat = FakeChat([text_reply(), text_reply()])
    trace = run_episode(FakeEnv(), SKILL, chat, seed=0, settings=settings())
    assert trace.stop_reason == "no_tool_call" and not trace.won
    assert chat.bodies[1]["messages"][-1] == {"role": "user", "content": NUDGE}
    assert trace.text_turns == 2


@pytest.mark.parametrize(
    "bad",
    [
        tool_reply(call("")),
        tool_reply(call("go to shelf 1\nlook")),
        tool_reply(call(3)),
        tool_reply({"function": {"name": "move", "arguments": {"command": "look"}}}),
        tool_reply(call("look"), call("inventory")),  # 한 턴에 행동 하나
        {"parse_error": "error parsing tool call"},
    ],
)
def test_format_errors_go_back_as_tool_results_and_three_in_a_row_stop(
    bad: dict[str, Any],
) -> None:
    chat = FakeChat([bad, bad, bad])
    env = FakeEnv()
    trace = run_episode(env, SKILL, chat, seed=0, settings=settings())
    assert env.commands == []  # 형식 오류 턴에는 환경을 움직이지 않는다
    assert trace.stop_reason == "format_error" and trace.format_errors == 3
    feedback = chat.bodies[1]["messages"][-1]
    assert feedback["role"] == "tool" and feedback["content"].startswith("Format error:")


def test_a_valid_call_resets_the_consecutive_format_error_count() -> None:
    bad = tool_reply(call(""))
    replies = [
        bad,
        bad,
        tool_reply(call("look")),
        bad,
        bad,
        tool_reply(call("take pen 1 from shelf 2")),
    ]
    trace = run_episode(FakeEnv(), SKILL, FakeChat(replies), seed=0, settings=settings())
    assert trace.won and trace.format_errors == 4


def test_step_cap_and_llm_call_cap() -> None:
    chat = FakeChat([tool_reply(call("look")) for _ in range(5)])
    trace = run_episode(FakeEnv(), SKILL, chat, seed=0, settings=settings(max_steps=3))
    assert trace.stop_reason == "max_steps" and len(trace.steps) == 3
    # 텍스트와 정상 행동이 번갈아 나와도 LLM 호출 상한에서 멈춘다
    # v1처럼 텍스트 응답은 연속 형식 오류 수를 바꾸지 않는다.
    # 그래서 형식 오류와 섞으면 형식 오류 쪽이 먼저 멈춘다.
    mixed = [text_reply(), tool_reply(call("look"))] * 10
    trace = run_episode(
        FakeEnv(), SKILL, FakeChat(mixed), seed=0, settings=settings(max_llm_calls=6)
    )
    assert trace.stop_reason == "max_llm_calls" and trace.llm_calls == 6


def test_steps_record_admissibility_for_e_b_only() -> None:
    chat = FakeChat([tool_reply(call("fly")), tool_reply(call("take pen 1 from shelf 2"))])
    trace = run_episode(FakeEnv(), SKILL, chat, seed=0, settings=settings())
    assert [s.admissible for s in trace.steps] == [False, True]


def turns(n: int, obs_len: int = 50) -> list[tuple[str, str]]:
    return [(f"go to shelf {i}", "x" * obs_len) for i in range(n)]


def test_history_keeps_everything_when_it_fits() -> None:
    messages, omitted = build_messages("SYS", FIRST, turns(3), budget_chars=10_000)
    assert omitted == 0
    assert [m["role"] for m in messages] == ["system", "user"] + ["assistant", "tool"] * 3


def test_history_drops_the_oldest_turns_and_keeps_system_and_first_observation() -> None:
    history = turns(40, obs_len=100)
    messages, omitted = build_messages("SYS", FIRST, history, budget_chars=2_000)
    assert omitted > 0
    assert messages[0]["content"] == "SYS" and messages[1]["content"] == FIRST
    assert messages[2] == {"role": "user", "content": f"[{omitted} earlier steps omitted]"}
    kept = [m for m in messages if m["role"] == "assistant"]
    assert len(kept) == 40 - omitted
    assert kept[-1]["tool_calls"][0]["function"]["arguments"] == {"command": "go to shelf 39"}
    assert sum(len(json.dumps(m)) for m in messages) <= 2_000


def test_history_budget_follows_num_ctx() -> None:
    s = PolicySettings()
    assert s.history_budget_chars == (s.num_ctx - s.reserve_tokens) * s.chars_per_token


def test_truncations_are_counted_in_the_trace() -> None:
    replies = [tool_reply(call("look")) for _ in range(8)]
    trace = run_episode(
        FakeEnv(),
        SKILL,
        FakeChat(replies),
        seed=0,
        settings=settings(max_steps=8, history_budget_override=900),
    )
    assert trace.truncated_calls > 0 and trace.max_omitted > 0


def test_trace_is_plain_data() -> None:
    chat = FakeChat([tool_reply(call("take pen 1 from shelf 2"))])
    trace = run_episode(FakeEnv(), SKILL, chat, seed=0, settings=settings())
    assert isinstance(trace, EpisodeTrace)
    json.dumps(trace.as_dict())


# --- 실제 게임으로 비노출 확인 ---------------------------------------------------------------

DATA = Path.home() / ".cache" / "alfworld" / "json_2.1.1"
POOL = Path(__file__).parents[1] / "records" / "p1_pilot_pool.json"
EXPERT = Path(__file__).parents[1] / "records" / "p1_expert.json"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="needs ALFWorld data")


def raw_observations(game: Path, actions: Sequence[str]) -> list[str]:
    """어댑터와 따로, textworld를 직접 돌려 얻은 관측(첫 관측 포함)."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    env_id = textworld.gym.register_game(
        str(game), textworld.EnvInfos(), max_episode_steps=50, wrappers=[AlfredDemangler()]
    )
    env = textworld.gym.make(env_id)
    obs, _ = env.reset()
    out = [obs]
    for a in actions:
        obs, *_ = env.step(a)
        out.append(obs)
    env.close()
    return out


@needs_data
def test_real_env_prompts_hold_no_command_list_facts_or_expert_plan() -> None:
    from alfworld_pilot.agent import AlfworldEnv

    rel = json.loads(POOL.read_text())["games"][0]["rel"]
    actions: Sequence[str] = json.loads(EXPERT.read_text())["games"][rel]["actions"][:6]
    env = AlfworldEnv(DATA / rel / "game.tw-pddl", max_steps=50)
    seen_lists: list[list[str]] = []
    original_step = env.step

    def spy(command: str) -> Outcome:
        seen_lists.append(env.last_admissible_for_tests())
        return original_step(command)

    env.step = spy  # type: ignore[method-assign]
    chat = FakeChat([tool_reply(call(a)) for a in actions] + [text_reply(), text_reply()])
    trace = run_episode(env, SKILL, chat, seed=0, settings=settings())
    assert trace.steps
    prompts = json.dumps(chat.bodies)
    # system 뒤의 메시지는 첫 관측, 환경 관측, 재촉 문구, 형식 오류 문장뿐이고(목록을 끼워 넣지
    # 않는다), assistant의 명령은 실제로 낸 명령뿐이다
    allowed = {NUDGE} | set(raw_observations(DATA / rel / "game.tw-pddl", actions))
    issued = set(actions)
    for body in chat.bodies:
        for m in body["messages"][1:]:
            if m["role"] == "assistant":
                for c in m.get("tool_calls", []):
                    assert c["function"]["arguments"]["command"] in issued
            else:
                assert m["content"] in allowed, m
    # 관측에 나오지 않은 가능한 명령은 대화 어디에도 없다(단어 경계로 비교)
    others = json.dumps([m.get("tool_calls", "") for b in chat.bodies for m in b["messages"][1:]])
    leaked = {
        c
        for lst in seen_lists
        for c in lst
        if c not in issued and re.search(rf"(?<![\w]){re.escape(c)}(?![\w])", others)
    }
    assert leaked == set()
    for word in ("admissible", "expert", "inreceptacle", "atlocation", "facts", "gamefile"):
        assert word not in prompts.lower()
    assert env.request_infos_for_tests() == {
        "admissible_commands": True,
        "won": True,
        "facts": False,
        "extras": [],
    }
