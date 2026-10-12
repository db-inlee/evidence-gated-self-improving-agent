"""증거 술어. 조치 시점에 그 이전 원시 관측에서만 계산한다.

필요한 관측이 없으면 술어는 성립하지 않는다(집합에 넣지 않는다). 그래서 "확인 안 함",
"확인했는데 정상", "확인했는데 장애"가 구분된다. 같은 도구를 여러 번 부르면 가장 최근
관측을 쓴다. 임계값은 `constants.py`에서 가져온다.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from opsworld import constants as c
from opsworld.ledger import Observation
from opsworld.state import DependencyStatus
from opsworld.tools import ToolName


class Predicate(StrEnum):
    INCIDENT_CONFIRMED = "incident_confirmed"
    RECENT_DEPLOYMENT_FOUND = "recent_deployment_found"
    DEPLOY_BEFORE_REGRESSION = "deploy_before_regression"
    DEPENDENCY_RULED_OUT = "dependency_ruled_out"
    DEPENDENCY_FAILURE_CONFIRMED = "dependency_failure_confirmed"
    TRAFFIC_RULED_OUT = "traffic_ruled_out"
    TRAFFIC_SPIKE_CONFIRMED = "traffic_spike_confirmed"
    CAPACITY_SATURATED = "capacity_saturated"


def derive_predicates(observations: Sequence[Observation]) -> frozenset[Predicate]:
    metrics = _latest(observations, ToolName.SERVICE_METRICS)
    deployments = _latest(observations, ToolName.RECENT_DEPLOYMENTS)
    dependencies = _latest(observations, ToolName.DEPENDENCY_HEALTH)
    traffic = _latest(observations, ToolName.TRAFFIC_METRICS)

    holds: set[Predicate] = set()

    if metrics is not None:
        p95_up = _at_least(
            metrics["p95_ms"], c.INCIDENT_P95_MULTIPLIER * metrics["baseline_p95_ms"]
        )
        errors_up = _at_least(
            metrics["error_rate"], metrics["baseline_error_rate"] + c.INCIDENT_ERROR_RATE_DELTA
        )
        if p95_up or errors_up:
            holds.add(Predicate.INCIDENT_CONFIRMED)
        if _at_least(metrics["cpu_usage_percent"], c.CPU_SATURATION_PERCENT):
            holds.add(Predicate.CAPACITY_SATURATED)

    if metrics is not None and deployments is not None:
        regression_at: int = metrics["regression_started_at"]
        deployed_ats: list[int] = [d["deployed_at"] for d in deployments["deployments"]]
        if any(abs(t - regression_at) <= c.RECENT_DEPLOYMENT_WINDOW_MIN for t in deployed_ats):
            holds.add(Predicate.RECENT_DEPLOYMENT_FOUND)
        if any(
            0 <= regression_at - t <= c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN for t in deployed_ats
        ):
            holds.add(Predicate.DEPLOY_BEFORE_REGRESSION)

    if dependencies is not None:
        if all(d["status"] == DependencyStatus.HEALTHY for d in dependencies["dependencies"]):
            holds.add(Predicate.DEPENDENCY_RULED_OUT)
        if metrics is not None and any(
            d["status"] == DependencyStatus.DEGRADED
            and d["outage_started_at"] is not None
            and d["outage_started_at"] <= metrics["regression_started_at"]
            for d in dependencies["dependencies"]
        ):
            holds.add(Predicate.DEPENDENCY_FAILURE_CONFIRMED)

    if traffic is not None:
        current_rps, baseline_rps = traffic["current_rps"], traffic["baseline_rps"]
        if not _at_least(current_rps, c.TRAFFIC_RULED_OUT_MULTIPLIER * baseline_rps):
            holds.add(Predicate.TRAFFIC_RULED_OUT)
        if _at_least(current_rps, c.TRAFFIC_SPIKE_MULTIPLIER * baseline_rps):
            holds.add(Predicate.TRAFFIC_SPIKE_CONFIRMED)

    return frozenset(holds)


def _at_least(value: float, threshold: float) -> bool:
    """`value ≥ threshold`. 차이가 FLOAT_TOLERANCE 이내면 경계 값으로 보고 참이다."""
    return value >= threshold - c.FLOAT_TOLERANCE


def _latest(observations: Sequence[Observation], tool: ToolName) -> dict[str, Any] | None:
    for observation in reversed(observations):
        if observation.tool == tool:
            return observation.payload
    return None
