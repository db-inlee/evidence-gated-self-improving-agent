"""조치 3개와 조치별 필수 증거 논리식.

v0 조치는 처분만 기록하고 시뮬레이터 상태는 바꾸지 않는다. v0 에피소드는 조치 1회로 끝나므로
바뀐 상태를 관측할 방법이 없고, 채점은 처분(Correct Disposition)과 조치 시점 근거만 본다.
조치 후 상태 변경과 재관측은 NOTES.md의 후속 후보다.

필수 증거는 논리식 객체로 표현한다. v0는 AND(`AllOf`)만 쓴다. OR 조합은 `EvidenceExpr`의
하위 클래스로 추가할 수 있게 열어 두되, 추가할 때마다 왜 충분한지 기록한다.
"""

from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass
from enum import StrEnum

from opsworld.predicates import Predicate
from opsworld.tools import Episode


class Action(StrEnum):
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    SCALE_SERVICE = "scale_service"
    OBSERVE_ONLY = "observe_only"


class EvidenceExpr(ABC):
    """필수 증거 논리식."""

    @abstractmethod
    def is_satisfied(self, predicates: Set[str]) -> bool: ...

    @abstractmethod
    def missing(self, predicates: Set[str]) -> frozenset[str]:
        """식을 충족하지 못하게 만든 술어. 충족되면 빈 집합."""


@dataclass(frozen=True)
class Require(EvidenceExpr):
    predicate: Predicate

    def is_satisfied(self, predicates: Set[str]) -> bool:
        return self.predicate in predicates

    def missing(self, predicates: Set[str]) -> frozenset[str]:
        return frozenset() if self.is_satisfied(predicates) else frozenset({self.predicate})


@dataclass(frozen=True)
class AllOf(EvidenceExpr):
    terms: tuple[EvidenceExpr, ...]

    def is_satisfied(self, predicates: Set[str]) -> bool:
        return all(term.is_satisfied(predicates) for term in self.terms)

    def missing(self, predicates: Set[str]) -> frozenset[str]:
        return frozenset().union(*(term.missing(predicates) for term in self.terms))


def all_of(*predicates: Predicate) -> AllOf:
    return AllOf(tuple(Require(p) for p in predicates))


# Ticket 001 §6 표와 1:1.
REQUIRED_EVIDENCE: dict[Action, EvidenceExpr] = {
    Action.ROLLBACK_DEPLOYMENT: all_of(
        Predicate.INCIDENT_CONFIRMED,
        Predicate.RECENT_DEPLOYMENT_FOUND,
        Predicate.DEPLOY_BEFORE_REGRESSION,
        Predicate.DEPENDENCY_RULED_OUT,
        Predicate.TRAFFIC_RULED_OUT,
    ),
    Action.SCALE_SERVICE: all_of(
        Predicate.INCIDENT_CONFIRMED,
        Predicate.TRAFFIC_SPIKE_CONFIRMED,
        Predicate.CAPACITY_SATURATED,
        Predicate.DEPENDENCY_RULED_OUT,
    ),
    Action.OBSERVE_ONLY: all_of(
        Predicate.INCIDENT_CONFIRMED,
        Predicate.DEPENDENCY_FAILURE_CONFIRMED,
    ),
}


def rollback_deployment(episode: Episode, service_id: str) -> None:
    episode.record_action(Action.ROLLBACK_DEPLOYMENT, service_id)


def scale_service(episode: Episode, service_id: str) -> None:
    episode.record_action(Action.SCALE_SERVICE, service_id)


def observe_only(episode: Episode, service_id: str) -> None:
    episode.record_action(Action.OBSERVE_ONLY, service_id)
