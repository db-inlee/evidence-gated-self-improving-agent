"""OpsWorld 상태 모델.

`Incident.root_cause`, `Incident.level`, `Incident.l2_subtype`, `Incident.l3_subtypes`,
`Deployment.has_bug`는 숨은 값이다.
어떤 도구 출력에도 넣지 않는다.
"""

from dataclasses import dataclass
from enum import StrEnum


class Family(StrEnum):
    """장애 생성 레시피. v0는 L1만 있다."""

    BAD_DEPLOYMENT = "bad_deployment"
    DEPENDENCY_FAILURE = "dependency_failure"
    TRAFFIC_SPIKE = "traffic_spike"


class RootCause(StrEnum):
    """숨은 근본 원인. 채점기만 읽는다."""

    BAD_DEPLOYMENT = "bad_deployment"
    DEPENDENCY_FAILURE = "dependency_failure"
    TRAFFIC_SPIKE = "traffic_spike"


class Level(StrEnum):
    """난이도. 숨은 값이라 도구 출력·알림·프롬프트에 나오지 않는다."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"  # Ticket 006: 같은 family의 L2 하위 유형 둘을 합친 사후 채점 전용 분포


class DependencyStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ServiceState:
    service_id: str
    baseline_p95_ms: float
    p95_ms: float
    baseline_error_rate: float
    error_rate: float
    cpu_usage: float  # 0~100 퍼센트
    replicas: int


@dataclass(frozen=True)
class Deployment:
    deployment_id: str
    service_id: str
    version: str
    deployed_at: int  # 분
    has_bug: bool  # 숨은 값


@dataclass(frozen=True)
class DependencyState:
    dependency_id: str
    status: DependencyStatus
    outage_started_at: int | None  # 분, 장애가 없으면 None


@dataclass(frozen=True)
class TrafficState:
    baseline_rps: float
    current_rps: float


@dataclass(frozen=True)
class Incident:
    incident_id: str
    seed: int
    family: Family
    root_cause: RootCause  # 숨은 값
    regression_started_at: int  # 분
    service: ServiceState
    deployments: tuple[Deployment, ...]
    dependencies: tuple[DependencyState, ...]
    traffic: TrafficState
    level: Level = Level.L1  # 숨은 값
    l2_subtype: str | None = None  # 숨은 값. L2 하위 유형(L1이면 None)
    l3_subtypes: tuple[str, ...] | None = None  # 숨은 값. L3가 합친 L2 하위 유형(L1·L2면 None)
