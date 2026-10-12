"""정책(에이전트)이 받는 유일한 환경 인터페이스.

정책은 `AgentEnv`만 받는다. 도구 명세, 도구 호출, 조치 호출만 제공하고 `Episode`/`Incident`나
숨은 값을 공개 속성으로 노출하지 않는다. 잘못된 호출은 예외 대신 `{"error": ...}`를 돌려주고
원장에 기록하지 않는다. 반환 dict는 복사본이라 정책이 고쳐도 원장은 바뀌지 않는다.

LLM이 읽는 텍스트(도구 설명)는 영어로 쓴다. 설명에는 근본 원인에 대한 단서를 넣지 않는다.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from opsworld.actions import observe_only, rollback_deployment, scale_service
from opsworld.tools import (
    Episode,
    EpisodeClosedError,
    UnknownServiceError,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

_SERVICE_ID_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"service_id": {"type": "string", "description": "Target service ID."}},
    "required": ["service_id"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: Literal["observation", "action"]
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(_SERVICE_ID_SCHEMA))


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "get_service_metrics",
        "observation",
        "Get baseline and current p95 latency (ms), baseline and current error rate (0-1), "
        "CPU usage (percent), and the minute the regression started.",
    ),
    ToolSpec(
        "get_recent_deployments",
        "observation",
        "List recent deployments of the service with version and deploy time (minute).",
    ),
    ToolSpec(
        "get_dependency_health",
        "observation",
        "Get the health status of each dependency of the service and the minute any outage "
        "started (null if none).",
    ),
    ToolSpec(
        "get_traffic_metrics",
        "observation",
        "Get baseline and current request rate (requests per second) of the service.",
    ),
    ToolSpec(
        "rollback_deployment",
        "action",
        "Roll back the most recent deployment of the service. "
        "Disposition action: ends the episode.",
    ),
    ToolSpec(
        "scale_service",
        "action",
        "Add capacity to the service. Disposition action: ends the episode.",
    ),
    ToolSpec(
        "observe_only",
        "action",
        "Take no corrective action and keep monitoring. Disposition action: ends the episode.",
    ),
)

_HANDLERS: dict[str, Callable[[Episode, str], object]] = {
    "get_service_metrics": get_service_metrics,
    "get_recent_deployments": get_recent_deployments,
    "get_dependency_health": get_dependency_health,
    "get_traffic_metrics": get_traffic_metrics,
    "rollback_deployment": rollback_deployment,
    "scale_service": scale_service,
    "observe_only": observe_only,
}


class AgentEnv:
    __slots__ = ("_episode",)

    def __init__(self, episode: Episode) -> None:
        self._episode = episode

    @property
    def service_id(self) -> str:
        """초기 알림에 쓰는 대상 서비스."""
        return self._episode.incident.service.service_id

    @property
    def closed(self) -> bool:
        return self._episode.closed

    def tool_specs(self) -> list[ToolSpec]:
        return list(_TOOL_SPECS)

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._episode.closed:
            return {"error": "episode closed"}
        handler = _HANDLERS.get(name)
        if handler is None:
            return {"error": f"unknown tool: {name!r}"}
        problem = _check_args(args)
        if problem is not None:
            return {"error": problem}
        try:
            result = handler(self._episode, args["service_id"])
        except UnknownServiceError:
            return {"error": f"unknown service: {args['service_id']!r}"}
        except EpisodeClosedError:
            return {"error": "episode closed"}
        if result is None:  # 조치
            return {"status": "recorded"}
        assert isinstance(result, dict)
        return copy.deepcopy(result)


def _check_args(args: object) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    extra = set(args) - {"service_id"}
    if extra:
        return f"unexpected arguments: {sorted(extra)}"
    if "service_id" not in args:
        return "missing required argument: service_id"
    if not isinstance(args["service_id"], str):
        return "service_id must be a string"
    return None
