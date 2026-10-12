"""관측 도구 4개와 에피소드.

도구는 `(episode, service_id)`를 받아 원시 관측 dict를 반환하고 같은 값을 원장에 기록한다.
숨은 값(`root_cause`, `has_bug`)은 어떤 출력에도 넣지 않는다.
"""

from enum import StrEnum
from typing import Any

from opsworld.ledger import Ledger
from opsworld.state import Incident


class ToolName(StrEnum):
    SERVICE_METRICS = "get_service_metrics"
    RECENT_DEPLOYMENTS = "get_recent_deployments"
    DEPENDENCY_HEALTH = "get_dependency_health"
    TRAFFIC_METRICS = "get_traffic_metrics"


class EpisodeClosedError(RuntimeError):
    """조치로 끝난 에피소드에서 도구나 조치를 다시 부를 때 발생한다."""


class UnknownServiceError(ValueError):
    """장애 대상이 아닌 서비스를 조회하거나 조치할 때 발생한다."""


class Episode:
    """장애 하나에 대한 도구 호출 기록. v0는 조치 1회로 종료한다."""

    def __init__(self, incident: Incident) -> None:
        self.incident = incident
        self.ledger = Ledger()
        self.action: str | None = None
        self.action_step: int | None = None  # 조치 이전 관측 수

    @property
    def closed(self) -> bool:
        return self.action is not None

    def check_open(self, service_id: str) -> None:
        if self.closed:
            raise EpisodeClosedError(f"episode already ended with {self.action!r}")
        if service_id != self.incident.service.service_id:
            raise UnknownServiceError(f"unknown service: {service_id!r}")

    def observe(self, tool: str, service_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.check_open(service_id)
        self.ledger.append(tool, payload)
        return payload

    def record_action(self, action: str, service_id: str) -> None:
        self.check_open(service_id)
        self.action = action
        self.action_step = len(self.ledger)


def get_service_metrics(episode: Episode, service_id: str) -> dict[str, Any]:
    incident = episode.incident
    service = incident.service
    payload: dict[str, Any] = {
        "service_id": service.service_id,
        "baseline_p95_ms": service.baseline_p95_ms,
        "p95_ms": service.p95_ms,
        "baseline_error_rate": service.baseline_error_rate,
        "error_rate": service.error_rate,
        "cpu_usage_percent": service.cpu_usage,
        "regression_started_at": incident.regression_started_at,
    }
    return episode.observe(ToolName.SERVICE_METRICS, service_id, payload)


def get_recent_deployments(episode: Episode, service_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "service_id": service_id,
        "deployments": [
            {
                "deployment_id": d.deployment_id,
                "version": d.version,
                "deployed_at": d.deployed_at,
            }
            for d in episode.incident.deployments
        ],
    }
    return episode.observe(ToolName.RECENT_DEPLOYMENTS, service_id, payload)


def get_dependency_health(episode: Episode, service_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "service_id": service_id,
        "dependencies": [
            {
                "dependency_id": d.dependency_id,
                "status": d.status.value,
                "outage_started_at": d.outage_started_at,
            }
            for d in episode.incident.dependencies
        ],
    }
    return episode.observe(ToolName.DEPENDENCY_HEALTH, service_id, payload)


def get_traffic_metrics(episode: Episode, service_id: str) -> dict[str, Any]:
    traffic = episode.incident.traffic
    payload: dict[str, Any] = {
        "service_id": service_id,
        "baseline_rps": traffic.baseline_rps,
        "current_rps": traffic.current_rps,
    }
    return episode.observe(ToolName.TRAFFIC_METRICS, service_id, payload)
