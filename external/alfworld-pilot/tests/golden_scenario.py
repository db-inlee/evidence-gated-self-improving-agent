"""진단 기록 추가(2026-11-03 amendment) 전후 비교용 가짜 채팅 시나리오.

텍스트 턴(thinking 포함, 500자 넘는 내용), 형식 오류, 정상 행동, 성공을 모두 지난다.
`fixtures/p2_golden.json`은 수정 **전** 코드로 이 시나리오를 돌려 만든 기록이다.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from alfworld_pilot.agent import EpisodeTrace, Outcome, PolicySettings, run_episode

FIRST = "You are in the middle of a room.\n\nYour task is to: put a pen in shelf."
LONG_TEXT = "I think the pen is somewhere. " * 30  # 900자


def _call(command: Any) -> dict[str, Any]:
    return {"function": {"name": "act", "arguments": {"command": command}}}


def _text(content: str, thinking: str = "") -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": content, "thinking": thinking},
        "prompt_eval_count": 80,
        "eval_count": 3,
    }


def _bad() -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": "", "tool_calls": [_call("a\nb")]},
        "prompt_eval_count": 70,
        "eval_count": 2,
    }


def text_stop_replies() -> list[dict[str, Any]]:
    """정상 행동 하나 뒤 텍스트 응답 두 번 연속 -> no_tool_call 종료."""
    return [replies()[1], _text("Done?", "t" * 7), _text("I am not sure.")]


def format_stop_replies() -> list[dict[str, Any]]:
    """텍스트 하나(연속 형식 오류 수를 바꾸지 않음)를 끼운 형식 오류 세 번 -> format_error 종료."""
    return [_bad(), _text("hmm"), _bad(), _bad()]


SCENARIOS = {
    "mixed": lambda: replies(),
    "text_stop": text_stop_replies,
    "format_stop": format_stop_replies,
}


def replies() -> list[dict[str, Any]]:
    return [
        {
            "message": {"role": "assistant", "content": "Let me think.", "thinking": "abc" * 10},
            "prompt_eval_count": 90,
            "eval_count": 5,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("go to shelf 2")],
                "thinking": "go",
            },
            "prompt_eval_count": 100,
            "eval_count": 9,
        },
        {
            "message": {"role": "assistant", "content": "", "tool_calls": [_call("")]},
            "prompt_eval_count": 110,
            "eval_count": 4,
        },
        {
            "message": {"role": "assistant", "content": LONG_TEXT},
            "prompt_eval_count": 120,
            "eval_count": 50,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("take pen 1 from shelf 2")],
            },
            "prompt_eval_count": 130,
            "eval_count": 9,
        },
    ]


@dataclass
class ScriptedChat:
    replies: list[dict[str, Any]]
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(json.loads(json.dumps(body)))
        return self.replies.pop(0)


@dataclass
class ScriptedEnv:
    def reset(self) -> str:
        return FIRST

    def step(self, command: str) -> Outcome:
        won = command == "take pen 1 from shelf 2"
        obs = "You pick up the pen 1 from the shelf 2." if won else "You arrive at shelf 2."
        return Outcome(observation=obs, won=won, done=won, admissible=True)


def run(name: str = "mixed") -> tuple[EpisodeTrace, list[dict[str, Any]]]:
    chat = ScriptedChat(SCENARIOS[name]())
    trace = run_episode(ScriptedEnv(), "# S0", chat, seed=5, settings=PolicySettings())
    return trace, chat.bodies


def snapshot(name: str = "mixed") -> dict[str, Any]:
    """시간 필드(wall_s)를 뺀 기록과 LLM에 보낸 요청 본문."""
    trace, bodies = run(name)
    record = {k: v for k, v in trace.as_dict().items() if k != "wall_s"}
    data: dict[str, Any] = json.loads(json.dumps({"record": record, "bodies": bodies}))
    return data


if __name__ == "__main__":
    print(json.dumps({n: snapshot(n) for n in SCENARIOS}, indent=1, sort_keys=True))
