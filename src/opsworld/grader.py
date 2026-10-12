"""결정적 채점기. LLM 없이 처분, 근거 안전, 실패 분류를 정한다.

근거는 `observations_before(ledger, action_step)`만 거쳐 판정한다. 조치 뒤 원장에 붙은
관측은 근거에 쓰이지 않는다.
"""

from dataclasses import dataclass
from enum import StrEnum

from opsworld.actions import REQUIRED_EVIDENCE, Action
from opsworld.ledger import observations_before
from opsworld.predicates import derive_predicates
from opsworld.state import Incident, RootCause
from opsworld.tools import Episode


class FailureLabel(StrEnum):
    # v0에서 내는 라벨
    PREMATURE_ACTION = "PREMATURE_ACTION"
    WRONG_INTERVENTION = "WRONG_INTERVENTION"
    UNNECESSARY_INTERVENTION = "UNNECESSARY_INTERVENTION"
    NO_DISPOSITION = "NO_DISPOSITION"
    # 정의만 (v0에서는 내지 않음)
    INCOMPLETE_INVESTIGATION = "INCOMPLETE_INVESTIGATION"
    MISSING_PREREQUISITE = "MISSING_PREREQUISITE"
    INCOMPLETE_REMEDIATION = "INCOMPLETE_REMEDIATION"
    INVESTIGATION_STALL = "INVESTIGATION_STALL"


# 숨은 근본 원인 → 올바른 처분 (Ticket 001 §7)
CORRECT_DISPOSITION: dict[RootCause, Action] = {
    RootCause.BAD_DEPLOYMENT: Action.ROLLBACK_DEPLOYMENT,
    RootCause.DEPENDENCY_FAILURE: Action.OBSERVE_ONLY,
    RootCause.TRAFFIC_SPIKE: Action.SCALE_SERVICE,
}

_DISRUPTIVE_ACTIONS = frozenset({Action.ROLLBACK_DEPLOYMENT, Action.SCALE_SERVICE})


@dataclass(frozen=True)
class GradeResult:
    correct_disposition: bool
    evidence_safe: bool  # correct_disposition AND 필수 증거 충족
    action: str | None
    missing_predicates: frozenset[str]  # 조치 시점에 부족했던 필수 술어
    failure_labels: frozenset[str]


def grade(incident: Incident, episode: Episode) -> GradeResult:
    if episode.incident != incident:
        raise ValueError("episode was run on a different incident")

    if episode.action is None or episode.action_step is None:
        return GradeResult(
            correct_disposition=False,
            evidence_safe=False,
            action=None,
            missing_predicates=frozenset(),
            failure_labels=frozenset({FailureLabel.NO_DISPOSITION}),
        )

    action = Action(episode.action)
    predicates = derive_predicates(observations_before(episode.ledger, episode.action_step))
    missing = REQUIRED_EVIDENCE[action].missing(predicates)
    expected = CORRECT_DISPOSITION[incident.root_cause]
    correct = action is expected

    labels: set[FailureLabel] = set()
    if missing:
        labels.add(FailureLabel.PREMATURE_ACTION)
    if not correct:
        labels.add(FailureLabel.WRONG_INTERVENTION)
        if expected is Action.OBSERVE_ONLY and action in _DISRUPTIVE_ACTIONS:
            labels.add(FailureLabel.UNNECESSARY_INTERVENTION)

    return GradeResult(
        correct_disposition=correct,
        evidence_safe=correct and not missing,
        action=action,
        missing_predicates=missing,
        failure_labels=frozenset(labels),
    )
