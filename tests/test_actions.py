"""조치, 필수 증거식, 에피소드 종료 규칙."""

from collections.abc import Callable

import pytest

from opsworld.actions import (
    REQUIRED_EVIDENCE,
    Action,
    AllOf,
    Require,
    observe_only,
    rollback_deployment,
    scale_service,
)
from opsworld.generator import generate_incident
from opsworld.predicates import Predicate
from opsworld.state import Family
from opsworld.tools import (
    Episode,
    EpisodeClosedError,
    UnknownServiceError,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

Step = Callable[[Episode, str], object]

ACTIONS: list[tuple[Step, Action]] = [
    (rollback_deployment, Action.ROLLBACK_DEPLOYMENT),
    (scale_service, Action.SCALE_SERVICE),
    (observe_only, Action.OBSERVE_ONLY),
]
ACTION_FUNCS: list[Step] = [act for act, _ in ACTIONS]
TOOLS: list[Step] = [
    get_service_metrics,
    get_recent_deployments,
    get_dependency_health,
    get_traffic_metrics,
]


def new_episode() -> tuple[Episode, str]:
    incident = generate_incident(Family.DEPENDENCY_FAILURE, 7)
    return Episode(incident), incident.service.service_id


# ---------------------------------------------------------------------------
# 필수 증거 표 (Ticket 001 §6과 1:1)
# ---------------------------------------------------------------------------

TICKET_REQUIRED_EVIDENCE = {
    "rollback_deployment": {
        "incident_confirmed",
        "recent_deployment_found",
        "deploy_before_regression",
        "dependency_ruled_out",
        "traffic_ruled_out",
    },
    "scale_service": {
        "incident_confirmed",
        "traffic_spike_confirmed",
        "capacity_saturated",
        "dependency_ruled_out",
    },
    "observe_only": {"incident_confirmed", "dependency_failure_confirmed"},
}


def test_v0_has_exactly_three_actions() -> None:
    assert {a.value for a in Action} == set(TICKET_REQUIRED_EVIDENCE)


@pytest.mark.parametrize("action", list(Action))
def test_required_evidence_matches_ticket_table(action: Action) -> None:
    expr = REQUIRED_EVIDENCE[action]
    required = TICKET_REQUIRED_EVIDENCE[action.value]
    assert expr.missing(frozenset()) == required
    assert expr.is_satisfied(frozenset(required))


@pytest.mark.parametrize("action", list(Action))
def test_required_evidence_is_and_only(action: Action) -> None:
    """v0는 AND만 쓴다. 필수 술어 하나라도 빠지면 충족되지 않는다."""
    required = TICKET_REQUIRED_EVIDENCE[action.value]
    expr = REQUIRED_EVIDENCE[action]
    assert isinstance(expr, AllOf)
    assert all(isinstance(term, Require) for term in expr.terms)
    for dropped in required:
        present = frozenset(required - {dropped})
        assert not expr.is_satisfied(present)
        assert expr.missing(present) == {dropped}


def test_extra_predicates_do_not_hurt() -> None:
    expr = REQUIRED_EVIDENCE[Action.OBSERVE_ONLY]
    present = frozenset({*TICKET_REQUIRED_EVIDENCE["observe_only"], "traffic_ruled_out"})
    assert expr.is_satisfied(present)
    assert expr.missing(present) == frozenset()


def test_nested_all_of_collects_missing_from_every_term() -> None:
    expr = AllOf(
        (
            Require(Predicate.INCIDENT_CONFIRMED),
            AllOf((Require(Predicate.TRAFFIC_RULED_OUT), Require(Predicate.DEPENDENCY_RULED_OUT))),
        )
    )
    assert expr.missing(frozenset({"traffic_ruled_out"})) == {
        "incident_confirmed",
        "dependency_ruled_out",
    }


# ---------------------------------------------------------------------------
# 조치 기록과 에피소드 종료
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("act", "action"), ACTIONS)
def test_action_records_name_and_number_of_prior_observations(act: Step, action: Action) -> None:
    episode, service_id = new_episode()
    get_service_metrics(episode, service_id)
    get_dependency_health(episode, service_id)
    act(episode, service_id)
    assert episode.action == action
    assert episode.action_step == 2
    assert episode.closed


@pytest.mark.parametrize("act", ACTION_FUNCS)
def test_action_does_not_change_incident(act: Step) -> None:
    episode, service_id = new_episode()
    before = episode.incident
    act(episode, service_id)
    assert episode.incident is before
    assert episode.incident == generate_incident(Family.DEPENDENCY_FAILURE, 7)


@pytest.mark.parametrize(("first", "first_action"), ACTIONS)
@pytest.mark.parametrize("second", ACTION_FUNCS)
def test_second_action_raises_episode_closed(
    first: Step, first_action: Action, second: Step
) -> None:
    episode, service_id = new_episode()
    first(episode, service_id)
    with pytest.raises(EpisodeClosedError):
        second(episode, service_id)
    assert episode.action == first_action


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_after_action_raises_and_records_nothing(tool: Step) -> None:
    episode, service_id = new_episode()
    observe_only(episode, service_id)
    with pytest.raises(EpisodeClosedError):
        tool(episode, service_id)
    assert len(episode.ledger) == 0


@pytest.mark.parametrize("call", [*TOOLS, *ACTION_FUNCS])
def test_unknown_service_raises_and_records_nothing(call: Step) -> None:
    episode, _ = new_episode()
    with pytest.raises(UnknownServiceError):
        call(episode, "not-the-incident-service")
    assert len(episode.ledger) == 0
    assert not episode.closed
