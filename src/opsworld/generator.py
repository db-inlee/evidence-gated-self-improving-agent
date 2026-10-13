"""결정적 장애 생성기.

같은 `(family, seed, level)`은 항상 같은 `Incident`를 낸다. 무작위성은 `random.Random` 인스턴스로만
쓰고 전역 `random`, 시간, 환경 변수에 의존하지 않는다. 시드는 시각과 수치만 바꾸고 family의
관측 특징(Ticket 001 §2 표)은 항상 유지한다.

L2(Ticket 003)는 같은 family 안에서 방해 요소가 다른 하위 유형을 만든다. 정답 처분은 family가
정하고, 모든 L2 장애에는 v0 도구만으로 Evidence-Safe 정답 궤적이 있다. L1은 L2 추가 전과 완전히
같고(난수 흐름도 같음), L2는 별도 난수 흐름(`"<family>:<seed>:L2"`)을 쓴다. 하위 유형은
`seed % 하위 유형 수`로 정해 family 안에서 균등하다.
"""

import hashlib
import random

from opsworld import constants as c
from opsworld.state import (
    DependencyState,
    DependencyStatus,
    Deployment,
    Family,
    Incident,
    Level,
    RootCause,
    ServiceState,
    TrafficState,
)

_ROOT_CAUSE: dict[Family, RootCause] = {
    Family.BAD_DEPLOYMENT: RootCause.BAD_DEPLOYMENT,
    Family.DEPENDENCY_FAILURE: RootCause.DEPENDENCY_FAILURE,
    Family.TRAFFIC_SPIKE: RootCause.TRAFFIC_SPIKE,
}


# family별 L2 하위 유형 (Ticket 003 §1 표)
L2_SUBTYPES: dict[Family, tuple[str, ...]] = {
    Family.BAD_DEPLOYMENT: ("cpu_saturated", "extra_deployment"),
    Family.DEPENDENCY_FAILURE: ("deploy_in_window", "traffic_elevated"),
    Family.TRAFFIC_SPIKE: ("deploy_in_window", "extra_deployment"),
}


def generate_incident(family: Family, seed: int, level: Level = Level.L1) -> Incident:
    if level is Level.L2:
        return _generate_l2(family, seed)
    rng = random.Random(f"{family.value}:{seed}")
    regression_started_at = rng.randint(*c.REGRESSION_START_RANGE)
    service_id = rng.choice(c.SERVICE_IDS)

    saturated = family is Family.TRAFFIC_SPIKE
    service = _degraded_service(rng, service_id, saturated=saturated)
    traffic = _traffic(rng, spike=family is Family.TRAFFIC_SPIKE)
    dependencies = _dependencies(
        rng,
        regression_started_at,
        failing=family is Family.DEPENDENCY_FAILURE,
    )

    if family is Family.BAD_DEPLOYMENT:
        deployed_at = regression_started_at - rng.randint(*c.BUGGY_DEPLOY_LEAD_MIN_RANGE)
        deployment = _deployment(rng, service_id, deployed_at, has_bug=True)
    else:
        deployed_at = _distractor_deploy_time(rng, regression_started_at)
        deployment = _deployment(rng, service_id, deployed_at, has_bug=False)

    return Incident(
        incident_id=_incident_id(family, seed),
        seed=seed,
        family=family,
        root_cause=_ROOT_CAUSE[family],
        regression_started_at=regression_started_at,
        service=service,
        deployments=(deployment,),
        dependencies=dependencies,
        traffic=traffic,
    )


def _generate_l2(family: Family, seed: int) -> Incident:
    subtypes = L2_SUBTYPES[family]
    subtype = subtypes[seed % len(subtypes)]
    rng = random.Random(f"{family.value}:{seed}:L2")
    regression_started_at = rng.randint(*c.REGRESSION_START_RANGE)
    service_id = rng.choice(c.SERVICE_IDS)

    saturated = family is Family.TRAFFIC_SPIKE or subtype in ("cpu_saturated", "traffic_elevated")
    spike = family is Family.TRAFFIC_SPIKE or subtype == "traffic_elevated"
    service = _degraded_service(rng, service_id, saturated=saturated)
    traffic = _traffic(rng, spike=spike)
    dependencies = _dependencies(
        rng,
        regression_started_at,
        failing=family is Family.DEPENDENCY_FAILURE,
    )

    def in_window() -> int:
        return regression_started_at - rng.randint(*c.L2_IN_WINDOW_DEPLOY_LEAD_MIN_RANGE)

    deployments: list[Deployment] = []
    if family is Family.BAD_DEPLOYMENT:
        buggy_at = regression_started_at - rng.randint(*c.BUGGY_DEPLOY_LEAD_MIN_RANGE)
        deployments.append(_deployment(rng, service_id, buggy_at, has_bug=True))
        if subtype == "extra_deployment":
            # 버그 배포보다 앞선 버그 없는 배포: 가장 최근 배포는 항상 버그 배포다.
            extra_at = regression_started_at - rng.randint(*c.L2_EXTRA_DEPLOY_BEFORE_MIN_RANGE)
            deployments.append(_deployment(rng, service_id, extra_at, has_bug=False))
    elif subtype == "deploy_in_window":
        deployments.append(_deployment(rng, service_id, in_window(), has_bug=False))
    elif subtype == "extra_deployment":
        deployments.append(_deployment(rng, service_id, in_window(), has_bug=False))
        outside_at = _distractor_deploy_time(rng, regression_started_at)
        deployments.append(_deployment(rng, service_id, outside_at, has_bug=False))
    else:  # dependency_failure/traffic_elevated: 배포는 L1과 같은 방해 배포(창 밖)
        distractor_at = _distractor_deploy_time(rng, regression_started_at)
        deployments.append(_deployment(rng, service_id, distractor_at, has_bug=False))

    return Incident(
        incident_id=_incident_id(family, seed, Level.L2),
        seed=seed,
        family=family,
        root_cause=_ROOT_CAUSE[family],
        regression_started_at=regression_started_at,
        service=service,
        # 배포 목록은 최근 순(도구 출력 순서)
        deployments=tuple(sorted(deployments, key=lambda d: d.deployed_at, reverse=True)),
        dependencies=dependencies,
        traffic=traffic,
        level=Level.L2,
        l2_subtype=subtype,
    )


def _incident_id(family: Family, seed: int, level: Level = Level.L1) -> str:
    # family 이름이 곧 근본 원인이므로 ID에는 해시만 남긴다. L1은 L2 추가 전과 같은 ID다.
    key = f"{family.value}:{seed}" if level is Level.L1 else f"{family.value}:{seed}:{level.value}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return f"inc-{digest[:10]}"


def _degraded_service(rng: random.Random, service_id: str, *, saturated: bool) -> ServiceState:
    baseline_p95_ms = round(rng.uniform(*c.BASELINE_P95_MS_RANGE), 1)
    baseline_error_rate = round(rng.uniform(*c.BASELINE_ERROR_RATE_RANGE), 4)
    p95_ms = round(baseline_p95_ms * rng.uniform(*c.DEGRADED_P95_MULTIPLIER_RANGE), 1)
    error_rate = round(baseline_error_rate + rng.uniform(*c.DEGRADED_ERROR_RATE_DELTA_RANGE), 4)
    cpu_range = c.SATURATED_CPU_RANGE if saturated else c.NORMAL_CPU_RANGE
    cpu_usage = round(rng.uniform(*cpu_range), 1)
    replicas = rng.randint(*c.REPLICAS_RANGE)
    return ServiceState(
        service_id=service_id,
        baseline_p95_ms=baseline_p95_ms,
        p95_ms=p95_ms,
        baseline_error_rate=baseline_error_rate,
        error_rate=error_rate,
        cpu_usage=cpu_usage,
        replicas=replicas,
    )


def _traffic(rng: random.Random, *, spike: bool) -> TrafficState:
    baseline_rps = round(rng.uniform(*c.BASELINE_RPS_RANGE), 1)
    multiplier_range = (
        c.SPIKE_TRAFFIC_MULTIPLIER_RANGE if spike else c.NORMAL_TRAFFIC_MULTIPLIER_RANGE
    )
    current_rps = round(baseline_rps * rng.uniform(*multiplier_range), 1)
    return TrafficState(baseline_rps=baseline_rps, current_rps=current_rps)


def _dependencies(
    rng: random.Random, regression_started_at: int, *, failing: bool
) -> tuple[DependencyState, ...]:
    count = rng.randint(*c.DEPENDENCY_COUNT_RANGE)
    ids = rng.sample(c.DEPENDENCY_IDS, count)
    failing_index = rng.randrange(count) if failing else None
    dependencies = []
    for index, dependency_id in enumerate(ids):
        if index == failing_index:
            outage_started_at = regression_started_at - rng.randint(*c.OUTAGE_LEAD_MIN_RANGE)
            dependencies.append(
                DependencyState(dependency_id, DependencyStatus.DEGRADED, outage_started_at)
            )
        else:
            dependencies.append(DependencyState(dependency_id, DependencyStatus.HEALTHY, None))
    return tuple(dependencies)


def _distractor_deploy_time(rng: random.Random, regression_started_at: int) -> int:
    # 버그 없는 최근 배포: 저하 시작 이후, 또는 deploy_before_regression 창보다 앞선 시각.
    if rng.random() < c.DISTRACTOR_DEPLOY_AFTER_PROBABILITY:
        return regression_started_at + rng.randint(*c.DISTRACTOR_DEPLOY_AFTER_MIN_RANGE)
    return regression_started_at - rng.randint(*c.DISTRACTOR_DEPLOY_BEFORE_MIN_RANGE)


def _deployment(
    rng: random.Random, service_id: str, deployed_at: int, *, has_bug: bool
) -> Deployment:
    deployment_id = f"deploy-{rng.randrange(16**6):06x}"
    major = rng.randint(*c.VERSION_MAJOR_RANGE)
    minor = rng.randint(*c.VERSION_MINOR_RANGE)
    patch = rng.randint(*c.VERSION_PATCH_RANGE)
    version = f"v{major}.{minor}.{patch}"
    return Deployment(
        deployment_id=deployment_id,
        service_id=service_id,
        version=version,
        deployed_at=deployed_at,
        has_bug=has_bug,
    )
