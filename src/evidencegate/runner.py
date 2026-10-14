"""에피소드 실행기.

키 `(policy_id, skill_id, family, incident_seed, repeat)`마다 한 번만 실행하고, 결과는 저장소에
한 줄씩 남긴다. 저장소에 이미 있는 키는 건너뛰므로 중단 후 같은 명령으로 이어서 돌 수 있다.
"""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from opsworld.state import Family, Level
from opsworld.tools import Episode

if TYPE_CHECKING:
    from evidencegate.store import EpisodeStore

# 2: EpisodeKey에 level 추가(Ticket 003). 버전 1 레코드는 level="L1"로 읽는다.
SCHEMA_VERSION = 2

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
    level: Level = Level.L1


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
    level: str = Level.L1.value  # 같은 (family, seed)라도 L1과 L2는 다른 키


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


@dataclass(frozen=True)
class Progress:
    """`run_suite`가 에피소드 하나를 실행할 때마다 내는 진행 상황(건너뛴 키는 제외)."""

    done: int  # 이번 호출에서 실행을 마친 에피소드 수
    total: int  # 이번 호출에서 실행할 에피소드 수
    skipped: int  # 저장소에 이미 있어 건너뛴 수
    record: EpisodeRecord
    elapsed_s: float  # 이번 호출 시작부터 지난 시간

    @property
    def mean_s(self) -> float:
        return self.elapsed_s / self.done

    @property
    def eta_s(self) -> float:
        return self.mean_s * (self.total - self.done)


def episode_key(
    policy: Policy,
    skill: Skill,
    family: Family,
    incident_seed: int,
    repeat: int,
    level: Level = Level.L1,
) -> EpisodeKey:
    return EpisodeKey(
        policy_id=policy.policy_id,
        config_id=config_id(policy),
        tool_spec_id=tool_spec_id(agent_env.TOOL_SPECS),
        skill_id=skill.skill_id,
        family=family.value,
        incident_seed=incident_seed,
        repeat=repeat,
        level=level.value,
    )


def run_episode(
    policy: Policy,
    skill: Skill,
    family: Family,
    incident_seed: int,
    repeat: int,
    level: Level = Level.L1,
) -> EpisodeRecord:
    incident = generate_incident(family, incident_seed, level)
    episode = Episode(incident)
    env = AgentEnv(episode)
    llm_seed = derive_llm_seed(family, incident_seed, repeat)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    start = time.perf_counter()
    trace = policy.run(env, skill, make_alert(env.service_id), llm_seed)
    duration_s = time.perf_counter() - start

    result = grade(incident, episode)
    return EpisodeRecord(
        key=episode_key(policy, skill, family, incident_seed, repeat, level),
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
    on_progress: Callable[[Progress], None] | None = None,
    workers: int = 1,
) -> SuiteRunResult:
    """반복 번호 바깥, 케이스 안쪽 순서로 돈다. 반복 하나가 끝날 때마다 완전한 결과가 생긴다.

    workers > 1이면 에피소드를 동시에 돌린다(Ticket 004 §1). 저장소 쓰기와 진행률 콜백은 이
    함수를 부른 스레드에서만 한다(단일 쓰기). 끝나는 순서는 바뀔 수 있지만 키는 순차와 같다.
    워커에서 예외가 나면 아직 시작하지 않은 에피소드를 취소하고 예외를 다시 낸다. 그때까지 저장된
    에피소드는 남고, 실행 중이던 에피소드는 저장되지 않아 재개할 때 다시 돈다.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    todo = [
        (case, repeat)
        for repeat in range(repeats)
        for case in cases
        if episode_key(policy, skill, case.family, case.incident_seed, repeat, case.level)
        not in store
    ]
    skipped = repeats * len(cases) - len(todo)
    start = time.perf_counter()
    done = 0

    def finish(record: EpisodeRecord) -> None:
        nonlocal done
        store.append(record)
        done += 1
        if on_progress is not None:
            on_progress(Progress(done, len(todo), skipped, record, time.perf_counter() - start))

    def execute(item: tuple[Case, int]) -> EpisodeRecord:
        case, repeat = item
        return run_episode(policy, skill, case.family, case.incident_seed, repeat, case.level)

    if workers == 1:
        for item in todo:
            finish(execute(item))
        return SuiteRunResult(executed=len(todo), skipped=skipped)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, item) for item in todo]
        try:
            for future in as_completed(futures):
                finish(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return SuiteRunResult(executed=len(todo), skipped=skipped)
