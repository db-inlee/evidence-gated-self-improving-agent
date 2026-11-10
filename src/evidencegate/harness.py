"""하네스 설정(Ticket 010 §1, 원칙 2·4·5). 정책이 LLM에 보내는 도구 정의와 tool 메시지에만 쓴다.

- `HarnessText`: 도구 순서(관측 도구 4개의 순열, 조치 도구 3개는 그 뒤에 지금 순서), 도구 설명
  덮어쓰기(도구 이름 → 새 설명), 표시 방식(`RENDERERS`의 이름). 모두 기본값이면 지금 코드와 같다
  (`config()`가 None, 정책 `config`에 키를 더하지 않음).
- 환경(`agent_env.TOOL_SPECS`, 도구 결과), 원장(`TurnRecord.result_summary`), 채점은 바꾸지 않는다.
  그래서 실행기의 `tool_spec_id`는 그대로이고, 하네스는 `config_id`로만 구별된다.
- 표시 방식은 도구 결과의 형식만 바꾼다. 값·키를 더하거나 빼지 않는다(proposer용 4개). 정보를 빼는
  표시 방식은 통제 후보 전용으로 따로 둔다(Ticket 010 §3, 아직 없음).
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from opsworld.agent_env import TOOL_SPECS, ToolSpec

OBSERVATION_TOOLS: tuple[str, ...] = tuple(s.name for s in TOOL_SPECS if s.kind == "observation")
ACTION_TOOLS: tuple[str, ...] = tuple(s.name for s in TOOL_SPECS if s.kind == "action")
TOOL_NAMES: tuple[str, ...] = tuple(s.name for s in TOOL_SPECS)
DEFAULT_RENDERER = "json_full"


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """중첩 dict·list를 점으로 이은 경로 → 스칼라. 빈 dict·list는 그대로 값으로 둔다."""
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key in sorted(value):
            out |= flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, item in enumerate(value):
            out |= flatten(item, f"{prefix}.{i}" if prefix else str(i))
        return out
    return {prefix: value}


def _json_full(result: Mapping[str, Any]) -> str:
    return json.dumps(result, sort_keys=True)  # 지금 정책의 tool 메시지와 같다


def _json_compact(result: Mapping[str, Any]) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _key_value_lines(result: Mapping[str, Any]) -> str:
    return "\n".join(f"{k}: {json.dumps(v)}" for k, v in flatten(dict(result)).items())


def _json_unsorted(result: Mapping[str, Any]) -> str:
    return json.dumps(result)


RENDERERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "json_full": _json_full,
    "json_compact": _json_compact,
    "key_value_lines": _key_value_lines,
    "json_unsorted": _json_unsorted,
}
PROPOSER_RENDERERS: tuple[str, ...] = (
    "json_full",
    "json_compact",
    "key_value_lines",
    "json_unsorted",
)


def render(result: Mapping[str, Any], renderer: str = DEFAULT_RENDERER) -> str:
    return RENDERERS[renderer](result)


@dataclass(frozen=True)
class HarnessText:
    tool_order: tuple[str, ...] | None = None  # 관측 도구 4개의 순열
    tool_descriptions: tuple[tuple[str, str], ...] = ()  # (도구 이름, 새 설명)
    observation_renderer: str = DEFAULT_RENDERER

    def __post_init__(self) -> None:
        if self.tool_order is not None and sorted(self.tool_order) != sorted(OBSERVATION_TOOLS):
            raise ValueError(f"tool_order must be a permutation of {OBSERVATION_TOOLS}")
        names = [name for name, _ in self.tool_descriptions]
        if len(set(names)) != len(names):
            raise ValueError("tool_descriptions names a tool more than once")
        for name, text in self.tool_descriptions:
            if name not in TOOL_NAMES:
                raise ValueError(f"unknown tool {name!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty description for {name}")
        if self.observation_renderer not in RENDERERS:
            raise ValueError(f"unknown observation_renderer {self.observation_renderer!r}")

    def order(self) -> tuple[str, ...]:
        return (*(self.tool_order or OBSERVATION_TOOLS), *ACTION_TOOLS)

    def overrides(self) -> dict[str, str]:
        """기본 설명과 다른 덮어쓰기만."""
        defaults = {s.name: s.description for s in TOOL_SPECS}
        return {n: t for n, t in sorted(self.tool_descriptions) if t != defaults[n]}

    def config(self) -> dict[str, Any] | None:
        """정책 `config["harness"]`. 모두 기본값이면 None(키를 더하지 않음)."""
        if (
            self.order() == TOOL_NAMES
            and not self.overrides()
            and (self.observation_renderer == DEFAULT_RENDERER)
        ):
            return None
        return {
            "tool_order": list(self.order()),
            "tool_descriptions": self.overrides(),
            "observation_renderer": self.observation_renderer,
        }


def apply_tool_specs(specs: Sequence[ToolSpec], harness: HarnessText | None) -> list[ToolSpec]:
    """LLM에 보낼 도구 정의 순서와 설명. 이름·종류·인자는 그대로다."""
    if harness is None or harness.config() is None:
        return list(specs)
    by_name = {s.name: s for s in specs}
    overrides = harness.overrides()
    return [
        replace(by_name[n], description=overrides.get(n, by_name[n].description))
        for n in harness.order()
    ]
