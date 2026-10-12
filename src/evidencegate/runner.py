"""에피소드 실행기.

키 `(policy_id, skill_id, family, incident_seed, repeat)`마다 한 번만 실행하고, 결과는 저장소에
한 줄씩 남긴다. 저장소에 이미 있는 키는 건너뛰므로 중단 후 같은 명령으로 이어서 돌 수 있다.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from evidencegate.ids import config_id, tool_spec_id
from evidencegate.policy import Policy, PolicyTrace
from evidencegate.seeds import derive_llm_seed
from evidencegate.skills import Skill
from opsworld import agent_env
from opsworld.agent_env import AgentEnv
from opsworld.generator import generate_incident
from opsworld.grader import grade
from opsworld.state import Family
from opsworld.tools import Episode

if TYPE_CHECKING:
    from evidencegate.store import EpisodeStore

SCHEMA_VERSION = 1

# 모든 family에서 같은 형식. 서비스 ID만 바뀌고 정답 단서는 없다.
ALERT_TEMPLATE = (
    "Alert: service {service_id} is degraded (latency/error SLO breach). "
    "Investigate and take exactly one disposition action."
)


def make_alert(service_id: str) -> str:
    return ALERT_TEMPLATE.format(service_id=service_id)


@dataclass(frozen=True)
class Case:
    family: Family
    incident_seed: int


@dataclass(frozen=True)
class EpisodeKey:
    """결과에 영향을 주는 입력 전부. 하나라도 바뀌면 다른 키다(Ticket 002 고정 결정 9)."""

    policy_id: str
    config_id: str  # 정책 설정 해시
    tool_spec_id: str  # 도구 명세 해시
    skill_id: str
    family: str
    incident_seed: int
    repeat: int


@dataclass(frozen=True)
class EpisodeRecord:
    key: EpisodeKey
    policy_config: dict[str, Any]  # config_id의 원문
    llm_seed: int
    # GradeResult 필드. 집합은 정렬된 튜플로 저장한다.
    correct_disposition: bool
    evidence_safe: bool
    action: str | None
    missing_predicates: tuple[str, ...]
    failure_labels: tuple[str, ...]
    trace: PolicyTrace
    started_at: str  # UTC ISO 8601
    duration_s: float
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class SuiteRunResult:
    executed: int
    skipped: int


def episode_key(
    policy: Policy, skill: Skill, family: Family, incident_seed: int, repeat: int
) -> EpisodeKey:
    return EpisodeKey(
        policy_id=policy.policy_id,
        config_id=config_id(policy),
        tool_spec_id=tool_spec_id(agent_env.TOOL_SPECS),
        skill_id=skill.skill_id,
        family=family.value,
        incident_seed=incident_seed,
        repeat=repeat,
    )


def run_episode(
    policy: Policy, skill: Skill, family: Family, incident_seed: int, repeat: int
) -> EpisodeRecord:
    incident = generate_incident(family, incident_seed)
    episode = Episode(incident)
    env = AgentEnv(episode)
    llm_seed = derive_llm_seed(family, incident_seed, repeat)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    start = time.perf_counter()
    trace = policy.run(env, skill, make_alert(env.service_id), llm_seed)
    duration_s = time.perf_counter() - start

    result = grade(incident, episode)
    return EpisodeRecord(
        key=episode_key(policy, skill, family, incident_seed, repeat),
        policy_config=policy.config,
        llm_seed=llm_seed,
        correct_disposition=result.correct_disposition,
        evidence_safe=result.evidence_safe,
        action=None if result.action is None else str(result.action),
        missing_predicates=tuple(sorted(str(p) for p in result.missing_predicates)),
        failure_labels=tuple(sorted(str(label) for label in result.failure_labels)),
        trace=trace,
        started_at=started_at,
        duration_s=duration_s,
    )


def run_suite(
    policy: Policy,
    skill: Skill,
    cases: Sequence[Case],
    repeats: int,
    store: "EpisodeStore",
) -> SuiteRunResult:
    """반복 번호 바깥, 케이스 안쪽 순서로 돈다. 반복 하나가 끝날 때마다 완전한 결과가 생긴다."""
    executed = skipped = 0
    for repeat in range(repeats):
        for case in cases:
            key = episode_key(policy, skill, case.family, case.incident_seed, repeat)
            if key in store:
                skipped += 1
                continue
            store.append(run_episode(policy, skill, case.family, case.incident_seed, repeat))
            executed += 1
    return SuiteRunResult(executed=executed, skipped=skipped)
