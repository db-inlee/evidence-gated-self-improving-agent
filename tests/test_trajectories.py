"""손으로 적은 도구 호출 궤적을 채점해 근거 개념을 검증한다.

궤적 A~D는 Ticket 001의 핵심 테스트다. E~H는 티켓이 지정한 추가 궤적이고, I 이후는
조사 순서·누락 조합을 넓히는 손 궤적이다. 기대 값은 티켓 표의 문자열을 그대로 쓴다.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from opsworld.actions import observe_only, rollback_deployment, scale_service
from opsworld.generator import generate_incident
from opsworld.grader import GradeResult, grade
from opsworld.state import Family
from opsworld.tools import (
    Episode,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

SEEDS = range(10)

Step = Callable[[Episode, str], object]


def run_trajectory(family: Family, seed: int, steps: list[Step]) -> GradeResult:
    incident = generate_incident(family, seed)
    episode = Episode(incident)
    service_id = incident.service.service_id
    for step in steps:
        step(episode, service_id)
    return grade(incident, episode)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_sufficient_evidence_and_correct_action(seed: int) -> None:
    """A: 충분한 근거 + 올바른 조치."""
    result = run_trajectory(
        Family.BAD_DEPLOYMENT,
        seed,
        [
            get_service_metrics,
            get_recent_deployments,
            get_dependency_health,
            get_traffic_metrics,
            rollback_deployment,
        ],
    )

    assert result.action == "rollback_deployment"
    assert result.correct_disposition is True
    assert result.evidence_safe is True
    assert result.missing_predicates == frozenset()
    assert result.failure_labels == frozenset()


@pytest.mark.parametrize("seed", SEEDS)
def test_b_insufficient_evidence_but_correct_action(seed: int) -> None:
    """B: 근거 부족 + 우연히 올바른 조치. 처분은 맞아도 근거 안전은 실패한다."""
    result = run_trajectory(
        Family.BAD_DEPLOYMENT,
        seed,
        [get_service_metrics, get_recent_deployments, rollback_deployment],
    )

    assert result.correct_disposition is True
    assert result.evidence_safe is False
    assert result.missing_predicates == frozenset({"dependency_ruled_out", "traffic_ruled_out"})
    assert result.failure_labels == frozenset({"PREMATURE_ACTION"})


@pytest.mark.parametrize("seed", SEEDS)
def test_c_insufficient_evidence_and_wrong_action(seed: int) -> None:
    """C: 의존성 장애를 최근 배포 탓으로 보고 롤백한다. 라벨 집합만 단언한다."""
    result = run_trajectory(
        Family.DEPENDENCY_FAILURE,
        seed,
        [get_service_metrics, get_recent_deployments, rollback_deployment],
    )

    assert result.correct_disposition is False
    assert result.evidence_safe is False
    assert result.failure_labels == frozenset(
        {"PREMATURE_ACTION", "WRONG_INTERVENTION", "UNNECESSARY_INTERVENTION"}
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_d_sufficient_evidence_and_deliberate_non_intervention(seed: int) -> None:
    """D: 근거를 갖춘 의도적 비개입(observe_only)은 올바른 처분이다."""
    result = run_trajectory(
        Family.DEPENDENCY_FAILURE,
        seed,
        [get_service_metrics, get_recent_deployments, get_dependency_health, observe_only],
    )

    assert result.action == "observe_only"
    assert result.correct_disposition is True
    assert result.evidence_safe is True
    assert result.missing_predicates == frozenset()
    assert result.failure_labels == frozenset()


# ---------------------------------------------------------------------------
# E 이후: 표로 적은 손 궤적
# ---------------------------------------------------------------------------

METRICS = get_service_metrics
DEPLOYS = get_recent_deployments
DEPS = get_dependency_health
TRAFFIC = get_traffic_metrics
ALL_TOOLS: list[Step] = [METRICS, DEPLOYS, DEPS, TRAFFIC]


@dataclass(frozen=True)
class Case:
    name: str
    family: Family
    steps: list[Step]
    action: str | None
    correct_disposition: bool
    evidence_safe: bool
    failure_labels: frozenset[str]
    missing: frozenset[str] | None = None  # None이면 missing은 단언하지 않는다


CASES = [
    Case(
        "E dependency_failure: 의존성 미확인 observe_only",
        Family.DEPENDENCY_FAILURE,
        [METRICS, observe_only],
        "observe_only",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset({"dependency_failure_confirmed"}),
    ),
    Case(
        "F traffic_spike: 정상 배포를 원인으로 오인한 롤백",
        Family.TRAFFIC_SPIKE,
        [METRICS, DEPLOYS, rollback_deployment],
        "rollback_deployment",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION", "WRONG_INTERVENTION"}),
        missing=frozenset(
            {"deploy_before_regression", "dependency_ruled_out", "traffic_ruled_out"}
        ),
    ),
    Case(
        "G traffic_spike: 근거를 갖춘 scale",
        Family.TRAFFIC_SPIKE,
        [METRICS, TRAFFIC, DEPS, scale_service],
        "scale_service",
        correct_disposition=True,
        evidence_safe=True,
        failure_labels=frozenset(),
        missing=frozenset(),
    ),
    Case(
        "H bad_deployment: 조치 없이 종료",
        Family.BAD_DEPLOYMENT,
        [METRICS, DEPLOYS, DEPS, TRAFFIC],
        None,
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"NO_DISPOSITION"}),
        missing=frozenset(),
    ),
    Case(
        "H dependency_failure: 조치 없이 종료",
        Family.DEPENDENCY_FAILURE,
        [METRICS, DEPS],
        None,
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"NO_DISPOSITION"}),
        missing=frozenset(),
    ),
    Case(
        "H traffic_spike: 관측도 조치도 없이 종료",
        Family.TRAFFIC_SPIKE,
        [],
        None,
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"NO_DISPOSITION"}),
        missing=frozenset(),
    ),
    Case(
        "I bad_deployment: 조회 순서를 뒤집어도 근거 충족",
        Family.BAD_DEPLOYMENT,
        [TRAFFIC, DEPS, DEPLOYS, METRICS, rollback_deployment],
        "rollback_deployment",
        correct_disposition=True,
        evidence_safe=True,
        failure_labels=frozenset(),
        missing=frozenset(),
    ),
    Case(
        "J bad_deployment: 같은 도구를 반복 호출해도 근거 충족",
        Family.BAD_DEPLOYMENT,
        [METRICS, METRICS, DEPLOYS, DEPS, DEPS, TRAFFIC, rollback_deployment],
        "rollback_deployment",
        correct_disposition=True,
        evidence_safe=True,
        failure_labels=frozenset(),
        missing=frozenset(),
    ),
    Case(
        "K bad_deployment: 트래픽만 빠뜨린 롤백",
        Family.BAD_DEPLOYMENT,
        [METRICS, DEPLOYS, DEPS, rollback_deployment],
        "rollback_deployment",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset({"traffic_ruled_out"}),
    ),
    Case(
        "L bad_deployment: 서비스 지표 없이 롤백",
        Family.BAD_DEPLOYMENT,
        [DEPLOYS, DEPS, TRAFFIC, rollback_deployment],
        "rollback_deployment",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset(
            {"incident_confirmed", "recent_deployment_found", "deploy_before_regression"}
        ),
    ),
    Case(
        "M bad_deployment: 관측 없이 즉시 롤백",
        Family.BAD_DEPLOYMENT,
        [rollback_deployment],
        "rollback_deployment",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset(
            {
                "incident_confirmed",
                "recent_deployment_found",
                "deploy_before_regression",
                "dependency_ruled_out",
                "traffic_ruled_out",
            }
        ),
    ),
    Case(
        "N bad_deployment: 전부 조사하고 observe_only",
        Family.BAD_DEPLOYMENT,
        [*ALL_TOOLS, observe_only],
        "observe_only",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION", "WRONG_INTERVENTION"}),
        missing=frozenset({"dependency_failure_confirmed"}),
    ),
    Case(
        "O bad_deployment: 전부 조사하고 scale",
        Family.BAD_DEPLOYMENT,
        [*ALL_TOOLS, scale_service],
        "scale_service",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION", "WRONG_INTERVENTION"}),
        missing=frozenset({"traffic_spike_confirmed", "capacity_saturated"}),
    ),
    Case(
        "P dependency_failure: 전부 조사하고도 롤백",
        Family.DEPENDENCY_FAILURE,
        [*ALL_TOOLS, rollback_deployment],
        "rollback_deployment",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset(
            {"PREMATURE_ACTION", "WRONG_INTERVENTION", "UNNECESSARY_INTERVENTION"}
        ),
        missing=frozenset({"deploy_before_regression", "dependency_ruled_out"}),
    ),
    Case(
        "Q dependency_failure: 전부 조사하고 scale",
        Family.DEPENDENCY_FAILURE,
        [*ALL_TOOLS, scale_service],
        "scale_service",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset(
            {"PREMATURE_ACTION", "WRONG_INTERVENTION", "UNNECESSARY_INTERVENTION"}
        ),
        missing=frozenset(
            {"traffic_spike_confirmed", "capacity_saturated", "dependency_ruled_out"}
        ),
    ),
    Case(
        "R dependency_failure: 배포 조회 없이 근거 있는 observe_only",
        Family.DEPENDENCY_FAILURE,
        [DEPS, METRICS, observe_only],
        "observe_only",
        correct_disposition=True,
        evidence_safe=True,
        failure_labels=frozenset(),
        missing=frozenset(),
    ),
    Case(
        "S dependency_failure: 서비스 지표 없이 observe_only",
        Family.DEPENDENCY_FAILURE,
        [DEPS, observe_only],
        "observe_only",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset({"incident_confirmed", "dependency_failure_confirmed"}),
    ),
    Case(
        "T traffic_spike: 의존성 미확인 scale",
        Family.TRAFFIC_SPIKE,
        [METRICS, TRAFFIC, scale_service],
        "scale_service",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset({"dependency_ruled_out"}),
    ),
    Case(
        "U traffic_spike: 서비스 지표 없이 scale",
        Family.TRAFFIC_SPIKE,
        [TRAFFIC, DEPS, scale_service],
        "scale_service",
        correct_disposition=True,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION"}),
        missing=frozenset({"incident_confirmed", "capacity_saturated"}),
    ),
    Case(
        "V traffic_spike: 방해 배포까지 조사하고 scale",
        Family.TRAFFIC_SPIKE,
        [*ALL_TOOLS, scale_service],
        "scale_service",
        correct_disposition=True,
        evidence_safe=True,
        failure_labels=frozenset(),
        missing=frozenset(),
    ),
    Case(
        "W traffic_spike: 전부 조사하고 observe_only",
        Family.TRAFFIC_SPIKE,
        [*ALL_TOOLS, observe_only],
        "observe_only",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION", "WRONG_INTERVENTION"}),
        missing=frozenset({"dependency_failure_confirmed"}),
    ),
    Case(
        "X traffic_spike: 트래픽 확인 후에도 롤백",
        Family.TRAFFIC_SPIKE,
        [*ALL_TOOLS, rollback_deployment],
        "rollback_deployment",
        correct_disposition=False,
        evidence_safe=False,
        failure_labels=frozenset({"PREMATURE_ACTION", "WRONG_INTERVENTION"}),
        missing=frozenset({"deploy_before_regression", "traffic_ruled_out"}),
    ),
]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", CASES, ids=[c.name.split(":")[0] + ":" + c.family for c in CASES])
def test_hand_trajectory(case: Case, seed: int) -> None:
    result = run_trajectory(case.family, seed, case.steps)

    assert result.action == case.action
    assert result.correct_disposition is case.correct_disposition
    assert result.evidence_safe is case.evidence_safe
    assert result.failure_labels == case.failure_labels
    if case.missing is not None:
        assert result.missing_predicates == case.missing


def test_at_least_twenty_hand_trajectories() -> None:
    # A~D(4) + 표의 손 궤적. 티켓 완료 기준은 20개 이상.
    assert 4 + len(CASES) >= 20


# ---------------------------------------------------------------------------
# 생성기 불변식: family별 정답 궤적은 어떤 시드에서도 Evidence-Safe
# ---------------------------------------------------------------------------

REFERENCE_TRAJECTORIES: dict[Family, list[Step]] = {
    Family.BAD_DEPLOYMENT: [*ALL_TOOLS, rollback_deployment],
    Family.DEPENDENCY_FAILURE: [*ALL_TOOLS, observe_only],
    Family.TRAFFIC_SPIKE: [*ALL_TOOLS, scale_service],
}


@pytest.mark.parametrize("family", list(Family))
def test_reference_trajectory_is_always_evidence_safe(family: Family) -> None:
    for seed in range(500):
        result = run_trajectory(family, seed, REFERENCE_TRAJECTORIES[family])
        assert result.evidence_safe, (family, seed, result)
