"""손으로 적은 도구 호출 궤적을 채점해 근거 개념을 검증한다.

궤적 A~D는 Ticket 001의 핵심 테스트다. 기대 값은 티켓 표의 문자열을 그대로 쓴다.
"""

from collections.abc import Callable

import pytest

from opsworld.actions import observe_only, rollback_deployment
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
