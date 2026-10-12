"""채점기: 조치 이후 관측 무시, 처분표, 라벨 규칙의 전수 성질."""

from collections.abc import Callable
from itertools import combinations

import pytest

from opsworld.actions import (
    REQUIRED_EVIDENCE,
    Action,
    observe_only,
    rollback_deployment,
    scale_service,
)
from opsworld.generator import generate_incident
from opsworld.grader import CORRECT_DISPOSITION, FailureLabel, GradeResult, grade
from opsworld.state import Family, RootCause
from opsworld.tools import (
    Episode,
    ToolName,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

Step = Callable[[Episode, str], object]
TOOLS: list[Step] = [
    get_service_metrics,
    get_recent_deployments,
    get_dependency_health,
    get_traffic_metrics,
]
ACTIONS: list[Step] = [rollback_deployment, scale_service, observe_only]
V0_LABELS = {
    "PREMATURE_ACTION",
    "WRONG_INTERVENTION",
    "UNNECESSARY_INTERVENTION",
    "NO_DISPOSITION",
}


def run(family: Family, seed: int, steps: list[Step]) -> tuple[Episode, GradeResult]:
    incident = generate_incident(family, seed)
    episode = Episode(incident)
    for step in steps:
        step(episode, incident.service.service_id)
    return episode, grade(incident, episode)


# ---------------------------------------------------------------------------
# 조치 이후 관측은 근거가 아니다
# ---------------------------------------------------------------------------


def test_observations_forced_after_action_are_ignored() -> None:
    """궤적 B 뒤에 의존성·트래픽 관측을 원장에 억지로 붙여도 판정은 그대로다."""
    incident = generate_incident(Family.BAD_DEPLOYMENT, 3)
    service_id = incident.service.service_id
    episode = Episode(incident)
    get_service_metrics(episode, service_id)
    get_recent_deployments(episode, service_id)
    rollback_deployment(episode, service_id)

    # 같은 장애의 별도 에피소드에서 실제 도구 출력을 얻어 닫힌 원장에 직접 붙인다.
    donor = Episode(incident)
    episode.ledger.append(ToolName.DEPENDENCY_HEALTH, get_dependency_health(donor, service_id))
    episode.ledger.append(ToolName.TRAFFIC_METRICS, get_traffic_metrics(donor, service_id))
    assert len(episode.ledger) == 4

    result = grade(incident, episode)
    assert result.evidence_safe is False
    assert result.missing_predicates == {"dependency_ruled_out", "traffic_ruled_out"}
    assert result.failure_labels == {"PREMATURE_ACTION"}


# ---------------------------------------------------------------------------
# 처분표, 라벨 정의
# ---------------------------------------------------------------------------


def test_correct_disposition_table_matches_ticket() -> None:
    assert {cause.value: action.value for cause, action in CORRECT_DISPOSITION.items()} == {
        "bad_deployment": "rollback_deployment",
        "dependency_failure": "observe_only",
        "traffic_spike": "scale_service",
    }
    assert set(CORRECT_DISPOSITION) == set(RootCause)


def test_failure_label_enum_defines_all_eight() -> None:
    assert {label.value for label in FailureLabel} == V0_LABELS | {
        "INCOMPLETE_INVESTIGATION",
        "MISSING_PREREQUISITE",
        "INCOMPLETE_REMEDIATION",
        "INVESTIGATION_STALL",
    }


def test_no_disposition_result() -> None:
    _, result = run(Family.DEPENDENCY_FAILURE, 0, [get_service_metrics])
    assert result == GradeResult(
        correct_disposition=False,
        evidence_safe=False,
        action=None,
        missing_predicates=frozenset(),
        failure_labels=frozenset({"NO_DISPOSITION"}),
    )


def test_grading_against_a_different_incident_raises() -> None:
    episode, _ = run(Family.BAD_DEPLOYMENT, 0, [rollback_deployment])
    with pytest.raises(ValueError):
        grade(generate_incident(Family.BAD_DEPLOYMENT, 1), episode)


# ---------------------------------------------------------------------------
# 전수 성질: 모든 family x 도구 부분집합 x (조치 3개 + 조치 없음)
# ---------------------------------------------------------------------------

TOOL_SUBSETS = [list(s) for r in range(len(TOOLS) + 1) for s in combinations(TOOLS, r)]
ENDINGS: list[Step | None] = [*ACTIONS, None]


def all_episodes() -> list[tuple[Family, int, list[Step], GradeResult]]:
    cases = []
    for family in Family:
        for seed in range(3):
            for tools in TOOL_SUBSETS:
                for ending in ENDINGS:
                    steps = [*tools, ending] if ending is not None else tools
                    _, result = run(family, seed, steps)
                    cases.append((family, seed, steps, result))
    return cases


ALL_EPISODES = all_episodes()


def test_exhaustive_grid_size() -> None:
    assert len(ALL_EPISODES) == 3 * 3 * 16 * 4


def test_v0_emits_only_the_four_v0_labels() -> None:
    for *_, result in ALL_EPISODES:
        assert result.failure_labels <= V0_LABELS


def test_evidence_safe_iff_correct_and_nothing_missing() -> None:
    for *_, result in ALL_EPISODES:
        if result.action is None:
            continue
        expected = result.correct_disposition and not result.missing_predicates
        assert result.evidence_safe is expected


def test_premature_action_iff_required_evidence_missing() -> None:
    for *_, result in ALL_EPISODES:
        if result.action is None:
            continue
        assert ("PREMATURE_ACTION" in result.failure_labels) is bool(result.missing_predicates)


def test_missing_is_subset_of_required_evidence() -> None:
    for *_, result in ALL_EPISODES:
        if result.action is None:
            continue
        required = REQUIRED_EVIDENCE[Action(result.action)].missing(frozenset())
        assert result.missing_predicates <= required


def test_wrong_intervention_iff_disposition_incorrect() -> None:
    for *_, result in ALL_EPISODES:
        if result.action is None:
            continue
        assert ("WRONG_INTERVENTION" in result.failure_labels) is (not result.correct_disposition)


def test_unnecessary_intervention_only_for_disruptive_action_on_dependency_failure() -> None:
    for family, _, _, result in ALL_EPISODES:
        expected = family is Family.DEPENDENCY_FAILURE and result.action in {
            "rollback_deployment",
            "scale_service",
        }
        assert ("UNNECESSARY_INTERVENTION" in result.failure_labels) is expected


def test_no_disposition_iff_no_action() -> None:
    for *_, result in ALL_EPISODES:
        assert ("NO_DISPOSITION" in result.failure_labels) is (result.action is None)
        if result.action is None:
            assert result.failure_labels == {"NO_DISPOSITION"}
