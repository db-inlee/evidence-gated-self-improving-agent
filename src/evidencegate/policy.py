"""정책 인터페이스와 궤적 기록.

정책은 `AgentEnv`만 받는다. 호출 하나는 `TurnRecord` 하나가 되고, 환경이 `{"error": ...}`를
돌려준 호출은 형식 오류로 센다(없는 도구, 잘못된 인자, 대상 외 서비스).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from evidencegate.skills import Skill
from opsworld.agent_env import AgentEnv

RESULT_SUMMARY_MAX_CHARS = 500
_TRUNCATION_MARK = "..."


class StopReason(StrEnum):
    ACTION = "action"
    MAX_STEPS = "max_steps"
    NO_TOOL_CALL = "no_tool_call"
    ERROR = "error"


@dataclass(frozen=True)
class TurnRecord:
    tool: str
    args: dict[str, Any]
    result_summary: str  # 결과의 compact JSON, 최대 RESULT_SUMMARY_MAX_CHARS자
    format_error: bool


@dataclass(frozen=True)
class PolicyTrace:
    turns: tuple[TurnRecord, ...]
    format_errors: int
    stop_reason: StopReason
    llm_config: dict[str, Any] | None = None  # Part B: 모델, 양자화, 엔진 버전, seed 등
    usage: dict[str, Any] | None = None  # Part B: 토큰 수, 캐시 토큰, 소요 시간


class Policy(Protocol):
    @property
    def policy_id(self) -> str: ...

    @property
    def config(self) -> dict[str, Any]:
        """정책 동작을 정하는 설정 전부. 실행 키의 config_id는 이 값의 해시다."""
        ...

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace: ...


def call_tool(env: AgentEnv, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], TurnRecord]:
    """환경을 호출하고 결과와 TurnRecord를 함께 돌려준다."""
    result = env.call(name, args)
    turn = TurnRecord(
        tool=name,
        args=dict(args) if isinstance(args, dict) else {"_raw": repr(args)},
        result_summary=summarize_result(result),
        format_error="error" in result,
    )
    return result, turn


def summarize_result(result: dict[str, Any]) -> str:
    text = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(text) <= RESULT_SUMMARY_MAX_CHARS:
        return text
    return text[: RESULT_SUMMARY_MAX_CHARS - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


def make_trace(turns: Sequence[TurnRecord], stop_reason: StopReason, **extra: Any) -> PolicyTrace:
    return PolicyTrace(
        turns=tuple(turns),
        format_errors=sum(t.format_error for t in turns),
        stop_reason=stop_reason,
        **extra,
    )


ScriptedCall = str | tuple[str, dict[str, Any]]


class ScriptedPolicy:
    """정해진 호출 순서를 그대로 재생한다. 조치가 받아들여지면 거기서 멈춘다.

    이름만 주면 인자는 `{"service_id": env.service_id}`다(설정에는 `null`로 남는다). 스킬, 알림,
    seed는 쓰지 않는다. 설정은 호출 목록이고, 이름은 policy_id에만 쓰인다.
    """

    def __init__(self, calls: Sequence[ScriptedCall], name: str) -> None:
        self.policy_id = f"scripted:{name}"
        self._calls = tuple(calls)
        self.config: dict[str, Any] = {
            "type": "scripted",
            "calls": [
                {"tool": c[0], "args": dict(c[1])}
                if isinstance(c, tuple)
                else {"tool": c, "args": None}
                for c in self._calls
            ],
        }

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        turns: list[TurnRecord] = []
        for call in self._calls:
            name, args = call if isinstance(call, tuple) else (call, {"service_id": env.service_id})
            _, turn = call_tool(env, name, args)
            turns.append(turn)
            if env.closed:
                return make_trace(turns, StopReason.ACTION)
        return make_trace(turns, StopReason.NO_TOOL_CALL)
