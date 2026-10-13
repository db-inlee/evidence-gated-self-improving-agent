"""L2 장애 생성기: L1 불변, 하위 유형별 특징, 생성기 불변식, 애매 구간 없음, 숨은 값, 결정성."""

import hashlib
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from evidencegate.runner import make_alert
from opsworld import constants as c
from opsworld.actions import observe_only, rollback_deployment, scale_service
from opsworld.agent_env import AgentEnv
from opsworld.generator import L2_SUBTYPES, generate_incident
from opsworld.grader import grade
from opsworld.ledger import observations_before
from opsworld.predicates import derive_predicates
from opsworld.state import Family, Incident, Level
from opsworld.tools import (
    Episode,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

TOOLS = (get_service_metrics, get_recent_deployments, get_dependency_health, get_traffic_metrics)
CORRECT_ACTION: dict[Family, Callable[[Episode, str], object]] = {
    Family.BAD_DEPLOYMENT: rollback_deployment,
    Family.DEPENDENCY_FAILURE: observe_only,
    Family.TRAFFIC_SPIKE: scale_service,
}
SUBTYPE_NAMES = {name for names in L2_SUBTYPES.values() for name in names}


def l2_seeds(family: Family, subtype: str, count: int) -> list[int]:
    """하위 유형은 seed % (하위 유형 수)로 정해진다. 그 하위 유형이 나오는 시드만 고른다."""
    index = L2_SUBTYPES[family].index(subtype)
    return [index + k * len(L2_SUBTYPES[family]) for k in range(count)]


def observe_all(incident: Incident) -> tuple[Episode, list[dict[str, Any]], frozenset[str]]:
    episode = Episode(incident)
    outputs = [tool(episode, incident.service.service_id) for tool in TOOLS]
    predicates = frozenset(derive_predicates(observations_before(episode.ledger, len(TOOLS))))
    return episode, outputs, predicates


SUBTYPE_CASES = [(family, subtype) for family, names in L2_SUBTYPES.items() for subtype in names]


# ---------------------------------------------------------------------------
# L1 불변
# ---------------------------------------------------------------------------

# Ticket 003 이전 코드로 계산한 값: 모든 family x 시드 0..499의 L1 Incident(이전 필드) repr 해시.
L1_DIGEST_BEFORE_TICKET_003 = "77ad304131da1e836a843ec697c50060138b42bd0a35c9e9361ef1ae5fffa101"


def test_l1_incidents_are_unchanged_by_ticket_003() -> None:
    digest = hashlib.sha256()
    for family in Family:
        for seed in range(500):
            i = generate_incident(family, seed)
            old = (
                i.incident_id,
                i.seed,
                i.family,
                i.root_cause,
                i.regression_started_at,
                i.service,
                i.deployments,
                i.dependencies,
                i.traffic,
            )
            digest.update(repr(old).encode())
    assert digest.hexdigest() == L1_DIGEST_BEFORE_TICKET_003


def test_default_level_is_l1_without_subtype() -> None:
    incident = generate_incident(Family.TRAFFIC_SPIKE, 3)
    assert incident == generate_incident(Family.TRAFFIC_SPIKE, 3, level=Level.L1)
    assert incident.level is Level.L1
    assert incident.l2_subtype is None


# ---------------------------------------------------------------------------
# 하위 유형 선택
# ---------------------------------------------------------------------------


def test_subtypes_per_family_match_ticket() -> None:
    assert L2_SUBTYPES == {
        Family.BAD_DEPLOYMENT: ("cpu_saturated", "extra_deployment"),
        Family.DEPENDENCY_FAILURE: ("deploy_in_window", "traffic_elevated"),
        Family.TRAFFIC_SPIKE: ("deploy_in_window", "extra_deployment"),
    }


@pytest.mark.parametrize("family", list(Family))
def test_subtypes_are_balanced_within_a_family(family: Family) -> None:
    subtypes = [generate_incident(family, seed, level=Level.L2).l2_subtype for seed in range(40)]
    assert {s: subtypes.count(s) for s in set(subtypes)} == {s: 20 for s in L2_SUBTYPES[family]}


def test_l2_keeps_family_root_cause_and_records_level() -> None:
    for family in Family:
        incident = generate_incident(family, 0, level=Level.L2)
        assert incident.level is Level.L2
        assert incident.root_cause.value == family.value


def test_l1_and_l2_of_the_same_seed_differ_and_have_distinct_ids() -> None:
    for family in Family:
        l1 = generate_incident(family, 7)
        l2 = generate_incident(family, 7, level=Level.L2)
        assert l1 != l2
        assert l1.incident_id != l2.incident_id


# ---------------------------------------------------------------------------
# 하위 유형별 특징 (Ticket 003 §1 표)
# ---------------------------------------------------------------------------

SEEDS_PER_SUBTYPE = 300


def window_lead(incident: Incident, deployed_at: int) -> int:
    return incident.regression_started_at - deployed_at


def in_window(lead: int) -> bool:
    return 0 <= lead <= c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN


def test_bad_deployment_cpu_saturated() -> None:
    for seed in l2_seeds(Family.BAD_DEPLOYMENT, "cpu_saturated", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.BAD_DEPLOYMENT, seed, level=Level.L2)
        _, _, predicates = observe_all(incident)
        assert "capacity_saturated" in predicates
        assert "traffic_ruled_out" in predicates
        assert "traffic_spike_confirmed" not in predicates
        assert [d.has_bug for d in incident.deployments] == [True]


def test_bad_deployment_extra_deployment() -> None:
    for seed in l2_seeds(Family.BAD_DEPLOYMENT, "extra_deployment", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.BAD_DEPLOYMENT, seed, level=Level.L2)
        buggy = [d for d in incident.deployments if d.has_bug]
        extra = [d for d in incident.deployments if not d.has_bug]
        assert len(buggy) == 1 and len(extra) == 1
        assert in_window(window_lead(incident, buggy[0].deployed_at))
        # 방해 배포는 저하 20~28분 전(버그 없음): 최근 배포이지만 deploy_before_regression 창 밖
        extra_lead = window_lead(incident, extra[0].deployed_at)
        assert 20 <= extra_lead <= 28
        assert not in_window(extra_lead)


def test_bad_deployment_extra_deployment_most_recent_is_the_buggy_one() -> None:
    """롤백 도구는 가장 최근 배포를 되돌리므로, 이 하위 유형에서 가장 최근 배포는 버그 배포다."""
    for seed in l2_seeds(Family.BAD_DEPLOYMENT, "extra_deployment", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.BAD_DEPLOYMENT, seed, level=Level.L2)
        most_recent = max(incident.deployments, key=lambda d: d.deployed_at)
        assert most_recent.has_bug
        assert incident.deployments[0] is most_recent  # 도구 출력은 최근 순


@pytest.mark.parametrize("level", list(Level))
def test_bad_deployment_most_recent_deployment_is_always_buggy(level: Level) -> None:
    """롤백(가장 최근 배포를 되돌림)이 정답인 family: L1·L2 모두 가장 최근 배포가 버그 배포다."""
    for seed in range(300):
        incident = generate_incident(Family.BAD_DEPLOYMENT, seed, level=level)
        assert max(incident.deployments, key=lambda d: d.deployed_at).has_bug


def test_dependency_failure_deploy_in_window() -> None:
    for seed in l2_seeds(Family.DEPENDENCY_FAILURE, "deploy_in_window", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.DEPENDENCY_FAILURE, seed, level=Level.L2)
        _, _, predicates = observe_all(incident)
        assert "deploy_before_regression" in predicates
        assert not any(d.has_bug for d in incident.deployments)


def test_dependency_failure_traffic_elevated() -> None:
    for seed in l2_seeds(Family.DEPENDENCY_FAILURE, "traffic_elevated", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.DEPENDENCY_FAILURE, seed, level=Level.L2)
        _, _, predicates = observe_all(incident)
        assert {"traffic_spike_confirmed", "capacity_saturated"} <= predicates
        assert "dependency_failure_confirmed" in predicates


def test_traffic_spike_deploy_in_window() -> None:
    for seed in l2_seeds(Family.TRAFFIC_SPIKE, "deploy_in_window", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.TRAFFIC_SPIKE, seed, level=Level.L2)
        _, _, predicates = observe_all(incident)
        assert "deploy_before_regression" in predicates
        assert "traffic_ruled_out" not in predicates
        assert not any(d.has_bug for d in incident.deployments)


def test_traffic_spike_extra_deployment() -> None:
    for seed in l2_seeds(Family.TRAFFIC_SPIKE, "extra_deployment", SEEDS_PER_SUBTYPE):
        incident = generate_incident(Family.TRAFFIC_SPIKE, seed, level=Level.L2)
        assert len(incident.deployments) == 2
        assert not any(d.has_bug for d in incident.deployments)
        leads = [window_lead(incident, d.deployed_at) for d in incident.deployments]
        assert sum(in_window(lead) for lead in leads) == 1
        assert all(abs(lead) <= c.RECENT_DEPLOYMENT_WINDOW_MIN for lead in leads)


# ---------------------------------------------------------------------------
# 생성기 불변식: 모든 L2 장애는 v0 도구로 풀 수 있다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("family", "subtype"), SUBTYPE_CASES)
def test_reference_trajectory_is_evidence_safe(family: Family, subtype: str) -> None:
    for seed in l2_seeds(family, subtype, SEEDS_PER_SUBTYPE):
        incident = generate_incident(family, seed, level=Level.L2)
        assert incident.l2_subtype == subtype
        episode, _, _ = observe_all(incident)
        CORRECT_ACTION[family](episode, incident.service.service_id)
        result = grade(incident, episode)
        assert result.evidence_safe, (family, subtype, seed, result)


def test_rollback_cannot_be_evidence_safe_in_traffic_spike_deploy_in_window() -> None:
    for seed in l2_seeds(Family.TRAFFIC_SPIKE, "deploy_in_window", 50):
        incident = generate_incident(Family.TRAFFIC_SPIKE, seed, level=Level.L2)
        episode, _, _ = observe_all(incident)
        rollback_deployment(episode, incident.service.service_id)
        result = grade(incident, episode)
        assert "traffic_ruled_out" in result.missing_predicates


# ---------------------------------------------------------------------------
# 애매 구간 없음: 트래픽 비율은 1.5 미만 또는 3 이상
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", list(Level))
def test_no_ambiguous_traffic_ratio(level: Level) -> None:
    for family in Family:
        for seed in range(600):
            traffic = generate_incident(family, seed, level=level).traffic
            ratio = traffic.current_rps / traffic.baseline_rps
            assert ratio < c.TRAFFIC_RULED_OUT_MULTIPLIER or ratio >= c.TRAFFIC_SPIKE_MULTIPLIER, (
                family,
                seed,
                ratio,
            )


# ---------------------------------------------------------------------------
# 숨은 값 비노출: level, l2_subtype, 하위 유형 이름
# ---------------------------------------------------------------------------

HIDDEN_WORDS = {"level", "l2_subtype", "L1", "L2", *SUBTYPE_NAMES}


def all_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [*value.keys(), *[s for v in value.values() for s in all_strings(v)]]
    if isinstance(value, list):
        return [s for v in value for s in all_strings(v)]
    return [value] if isinstance(value, str) else []


@pytest.mark.parametrize(("family", "subtype"), SUBTYPE_CASES)
def test_tool_outputs_and_alert_do_not_reveal_level_or_subtype(
    family: Family, subtype: str
) -> None:
    for seed in l2_seeds(family, subtype, 30):
        incident = generate_incident(family, seed, level=Level.L2)
        env = AgentEnv(Episode(incident))
        args = {"service_id": env.service_id}
        outputs = [env.call(spec.name, args) for spec in env.tool_specs()[:4]]
        texts = [*[s for o in outputs for s in all_strings(o)], make_alert(env.service_id)]
        for text in texts:
            assert not any(word in text for word in HIDDEN_WORDS), (text, subtype)
        assert not any(word in incident.incident_id for word in HIDDEN_WORDS)


# ---------------------------------------------------------------------------
# 결정성
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(Family))
def test_l2_is_deterministic(family: Family) -> None:
    for seed in range(100):
        assert generate_incident(family, seed, level=Level.L2) == generate_incident(
            family, seed, level=Level.L2
        )


def test_l2_is_deterministic_across_processes() -> None:
    script = (
        "from opsworld.generator import generate_incident\n"
        "from opsworld.state import Family, Level\n"
        "seeds = range(20)\n"
        "print([repr(generate_incident(f, s, level=Level.L2)) for f in Family for s in seeds])\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1
