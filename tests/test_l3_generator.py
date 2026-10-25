"""L3 장애 생성기 (Ticket 006 §1~2): 구성, 술어 집합, 식별 가능성, 경계 여유, 숨은 값, 결정성.

L1·L2 불변(골든 해시)은 tests/test_l2_generator.py에 있다.
"""

import os
import random
import subprocess
import sys
from collections.abc import Callable

import pytest

from evidencegate.runner import make_alert
from opsworld import constants as c
from opsworld.actions import (
    REQUIRED_EVIDENCE,
    Action,
    observe_only,
    rollback_deployment,
    scale_service,
)
from opsworld.agent_env import AgentEnv
from opsworld.generator import L2_SUBTYPES, generate_incident
from opsworld.grader import grade
from opsworld.ledger import observations_before
from opsworld.predicates import Predicate, derive_predicates
from opsworld.state import Family, Incident, Level
from opsworld.tools import (
    Episode,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

TOOLS = (get_service_metrics, get_recent_deployments, get_dependency_health, get_traffic_metrics)

# Ticket 006 §3: L3-Oracle 400,000~400,099, L3-check 410,000~410,009. 그 밖에 시드 0..299도 본다.
L3_SEEDS = (*range(300), *range(400_000, 400_100), *range(410_000, 410_010))

CORRECT: dict[Family, Action] = {
    Family.BAD_DEPLOYMENT: Action.ROLLBACK_DEPLOYMENT,
    Family.DEPENDENCY_FAILURE: Action.OBSERVE_ONLY,
    Family.TRAFFIC_SPIKE: Action.SCALE_SERVICE,
}
ACT: dict[Action, Callable[[Episode, str], None]] = {
    Action.ROLLBACK_DEPLOYMENT: rollback_deployment,
    Action.OBSERVE_ONLY: observe_only,
    Action.SCALE_SERVICE: scale_service,
}

P = Predicate
# Ticket 006 §2-1 표: L3에서 성립하는 술어(나머지는 성립하지 않음)
EXPECTED_PREDICATES: dict[Family, frozenset[Predicate]] = {
    Family.BAD_DEPLOYMENT: frozenset(
        {
            P.INCIDENT_CONFIRMED,
            P.RECENT_DEPLOYMENT_FOUND,
            P.DEPLOY_BEFORE_REGRESSION,
            P.DEPENDENCY_RULED_OUT,
            P.TRAFFIC_RULED_OUT,
            P.CAPACITY_SATURATED,
        }
    ),
    Family.DEPENDENCY_FAILURE: frozenset(
        {
            P.INCIDENT_CONFIRMED,
            P.RECENT_DEPLOYMENT_FOUND,
            P.DEPLOY_BEFORE_REGRESSION,
            P.DEPENDENCY_FAILURE_CONFIRMED,
            P.TRAFFIC_SPIKE_CONFIRMED,
            P.CAPACITY_SATURATED,
        }
    ),
    Family.TRAFFIC_SPIKE: frozenset(
        {
            P.INCIDENT_CONFIRMED,
            P.RECENT_DEPLOYMENT_FOUND,
            P.DEPLOY_BEFORE_REGRESSION,
            P.DEPENDENCY_RULED_OUT,
            P.TRAFFIC_SPIKE_CONFIRMED,
            P.CAPACITY_SATURATED,
        }
    ),
}
# 표의 "불성립" 열. 정확한 집합 비교에 더해, 표에 적힌 함정 술어를 이름으로 한 번 더 확인한다.
EXPECTED_ABSENT: dict[Family, frozenset[Predicate]] = {
    Family.BAD_DEPLOYMENT: frozenset({P.TRAFFIC_SPIKE_CONFIRMED, P.DEPENDENCY_FAILURE_CONFIRMED}),
    Family.DEPENDENCY_FAILURE: frozenset({P.DEPENDENCY_RULED_OUT, P.TRAFFIC_RULED_OUT}),
    Family.TRAFFIC_SPIKE: frozenset({P.TRAFFIC_RULED_OUT}),
}


def l3(family: Family, seed: int) -> Incident:
    return generate_incident(family, seed, level=Level.L3)


def observe_all(incident: Incident) -> tuple[Episode, frozenset[str]]:
    episode = Episode(incident)
    for tool in TOOLS:
        tool(episode, incident.service.service_id)
    return episode, frozenset(derive_predicates(observations_before(episode.ledger, len(TOOLS))))


def leads(incident: Incident) -> list[int]:
    """배포마다 저하 시작보다 몇 분 앞서는가(음수 = 저하 뒤)."""
    return [incident.regression_started_at - d.deployed_at for d in incident.deployments]


def in_before_window(lead: int) -> bool:
    return 0 <= lead <= c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN


# ---------------------------------------------------------------------------
# 구성 (§1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(Family))
def test_l3_combines_both_l2_subtypes_and_records_them_as_hidden_values(family: Family) -> None:
    incident = l3(family, 0)
    assert incident.level is Level.L3
    assert incident.l2_subtype is None
    assert incident.l3_subtypes == L2_SUBTYPES[family]
    assert incident.root_cause.value == family.value


def test_l1_and_l2_have_no_l3_subtypes() -> None:
    for level in (Level.L1, Level.L2):
        assert generate_incident(Family.TRAFFIC_SPIKE, 4, level=level).l3_subtypes is None


@pytest.mark.parametrize("family", list(Family))
def test_l3_differs_from_l1_and_l2_of_the_same_seed_and_has_its_own_id(family: Family) -> None:
    incidents = [generate_incident(family, 7, level=level) for level in (Level.L1, Level.L2)]
    third = l3(family, 7)
    for other in incidents:
        assert other != third
        assert other.incident_id != third.incident_id


def test_bad_deployment_l3_has_buggy_and_extra_deployments_with_saturated_cpu() -> None:
    for seed in L3_SEEDS:
        incident = l3(Family.BAD_DEPLOYMENT, seed)
        assert [d.has_bug for d in incident.deployments] == [True, False]  # 최근 순
        buggy_lead, extra_lead = leads(incident)
        assert 1 <= buggy_lead <= 12
        assert 20 <= extra_lead <= 28
        assert incident.service.cpu_usage >= c.CPU_SATURATION_PERCENT


def test_dependency_failure_l3_has_in_window_and_outside_deployments_and_traffic_trap() -> None:
    """창 안 배포와 창 밖 방해 배포의 합집합 + CPU 포화 + 트래픽 급증 + 의존성 장애."""
    for seed in L3_SEEDS:
        incident = l3(Family.DEPENDENCY_FAILURE, seed)
        assert len(incident.deployments) == 2
        assert not any(d.has_bug for d in incident.deployments)
        assert sorted(in_before_window(lead) for lead in leads(incident)) == [False, True]
        assert sum(d.outage_started_at is not None for d in incident.dependencies) == 1


def test_traffic_spike_l3_has_the_l2_extra_deployment_structure() -> None:
    """deploy_in_window ⊂ extra_deployment: 창 안 배포 1개 + 창 밖 방해 배포 1개(T1, 대조 칸)."""
    for seed in L3_SEEDS:
        incident = l3(Family.TRAFFIC_SPIKE, seed)
        assert len(incident.deployments) == 2
        assert not any(d.has_bug for d in incident.deployments)
        assert sorted(in_before_window(lead) for lead in leads(incident)) == [False, True]


@pytest.mark.parametrize("family", list(Family))
def test_l3_uses_its_own_random_stream_in_the_l2_order(family: Family) -> None:
    """§1: 난수 흐름 "<family>:<seed>:L3", 소비 순서는 저하 시작 → 서비스 → ... (L2와 같음)."""
    for seed in L3_SEEDS:
        rng = random.Random(f"{family.value}:{seed}:L3")
        incident = l3(family, seed)
        assert incident.regression_started_at == rng.randint(*c.REGRESSION_START_RANGE)
        assert incident.service.service_id == rng.choice(c.SERVICE_IDS)
        assert incident.service.baseline_p95_ms == round(rng.uniform(*c.BASELINE_P95_MS_RANGE), 1)


@pytest.mark.parametrize("family", list(Family))
def test_deployments_are_listed_most_recent_first(family: Family) -> None:
    for seed in L3_SEEDS:
        times = [d.deployed_at for d in l3(family, seed).deployments]
        assert times == sorted(times, reverse=True)


# ---------------------------------------------------------------------------
# 불변식 1~2: 술어 집합, 정답 조치만 근거를 갖출 수 있다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(Family))
def test_predicate_set_is_exactly_the_ticket_table(family: Family) -> None:
    for seed in L3_SEEDS:
        _, predicates = observe_all(l3(family, seed))
        assert predicates == EXPECTED_PREDICATES[family], (family, seed)
        assert not predicates & EXPECTED_ABSENT[family]


@pytest.mark.parametrize("family", list(Family))
def test_only_the_correct_action_can_have_its_required_evidence(family: Family) -> None:
    for seed in L3_SEEDS:
        _, predicates = observe_all(l3(family, seed))
        for action, expr in REQUIRED_EVIDENCE.items():
            missing = expr.missing(predicates)
            if action is CORRECT[family]:
                assert not missing, (family, seed, action)
            else:
                assert missing, (family, seed, action)


def test_the_traps_named_in_the_ticket_block_the_wrong_actions() -> None:
    """§2-2의 문장: 어떤 술어가 빠져 오답 조치가 근거를 갖출 수 없는가."""
    rollback = REQUIRED_EVIDENCE[Action.ROLLBACK_DEPLOYMENT]
    scale = REQUIRED_EVIDENCE[Action.SCALE_SERVICE]
    for seed in L3_SEEDS:
        _, dep = observe_all(l3(Family.DEPENDENCY_FAILURE, seed))
        assert {P.DEPLOY_BEFORE_REGRESSION, P.TRAFFIC_SPIKE_CONFIRMED, P.CAPACITY_SATURATED} <= dep
        assert P.DEPENDENCY_RULED_OUT in rollback.missing(dep)
        assert P.DEPENDENCY_RULED_OUT in scale.missing(dep)
        _, traffic = observe_all(l3(Family.TRAFFIC_SPIKE, seed))
        assert P.TRAFFIC_RULED_OUT in rollback.missing(traffic)
        _, bad = observe_all(l3(Family.BAD_DEPLOYMENT, seed))
        assert P.TRAFFIC_SPIKE_CONFIRMED in scale.missing(bad)


@pytest.mark.parametrize("family", list(Family))
def test_reference_trajectory_is_evidence_safe(family: Family) -> None:
    for seed in L3_SEEDS:
        incident = l3(family, seed)
        episode, _ = observe_all(incident)
        ACT[CORRECT[family]](episode, incident.service.service_id)
        result = grade(incident, episode)
        assert result.correct_disposition and result.evidence_safe, (family, seed, result)


# ---------------------------------------------------------------------------
# 불변식 3: bad L3에서 가장 최근 배포는 버그 배포이고 다른 배포와 8분 이상 떨어져 있다
# ---------------------------------------------------------------------------


def test_bad_deployment_most_recent_is_buggy_and_separated() -> None:
    for seed in L3_SEEDS:
        incident = l3(Family.BAD_DEPLOYMENT, seed)
        most_recent, *others = incident.deployments
        assert most_recent.has_bug
        assert most_recent is max(incident.deployments, key=lambda d: d.deployed_at)
        for other in others:
            gap = most_recent.deployed_at - other.deployed_at
            assert gap >= c.L3_MIN_BUGGY_DEPLOY_SEPARATION_MIN, (seed, gap)


# ---------------------------------------------------------------------------
# 불변식 4: 경계 여유 (incident 값에서 직접 확인)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(Family))
def test_in_window_deployments_keep_the_margin_to_the_window_end(family: Family) -> None:
    for seed in L3_SEEDS:
        for lead in leads(l3(family, seed)):
            if in_before_window(lead):
                margin = c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN - lead
                assert margin >= c.L3_MARGIN_BEFORE_WINDOW_END_MIN, (family, seed, lead)


@pytest.mark.parametrize("family", list(Family))
def test_deployments_keep_the_margin_to_the_recent_deployment_boundary(family: Family) -> None:
    for seed in L3_SEEDS:
        for lead in leads(l3(family, seed)):
            margin = c.RECENT_DEPLOYMENT_WINDOW_MIN - abs(lead)
            assert margin >= c.L3_MARGIN_RECENT_DEPLOYMENT_MIN, (family, seed, lead)


def test_bad_deployment_extra_deployment_is_well_outside_the_before_window() -> None:
    for seed in L3_SEEDS:
        extra_lead = leads(l3(Family.BAD_DEPLOYMENT, seed))[1]
        margin = extra_lead - c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN
        assert margin >= c.L3_MARGIN_BAD_EXTRA_OUTSIDE_WINDOW_MIN, (seed, extra_lead)


@pytest.mark.parametrize("family", list(Family))
def test_traffic_ratio_keeps_the_margin_to_both_thresholds(family: Family) -> None:
    for seed in L3_SEEDS:
        traffic = l3(family, seed).traffic
        ratio = traffic.current_rps / traffic.baseline_rps
        if ratio < c.TRAFFIC_RULED_OUT_MULTIPLIER:
            assert c.TRAFFIC_RULED_OUT_MULTIPLIER - ratio >= c.L3_MARGIN_TRAFFIC_RULED_OUT
        else:
            assert ratio - c.TRAFFIC_SPIKE_MULTIPLIER >= c.L3_MARGIN_TRAFFIC_SPIKE, (seed, ratio)


@pytest.mark.parametrize("family", list(Family))
def test_cpu_keeps_the_margin_to_the_saturation_threshold(family: Family) -> None:
    for seed in L3_SEEDS:
        cpu = l3(family, seed).service.cpu_usage
        assert abs(cpu - c.CPU_SATURATION_PERCENT) >= c.L3_MARGIN_CPU_PERCENT, (family, seed, cpu)


def test_margin_constants_match_the_ticket_table() -> None:
    assert c.L3_MARGIN_BEFORE_WINDOW_END_MIN == 3
    assert c.L3_MARGIN_RECENT_DEPLOYMENT_MIN == 2
    assert c.L3_MARGIN_BAD_EXTRA_OUTSIDE_WINDOW_MIN == 5
    assert c.L3_MARGIN_TRAFFIC_RULED_OUT == 0.25
    assert c.L3_MARGIN_TRAFFIC_SPIKE == 0.3
    assert c.L3_MARGIN_CPU_PERCENT == 3.0
    assert c.L3_MIN_BUGGY_DEPLOY_SEPARATION_MIN == 8


# ---------------------------------------------------------------------------
# 숨은 값 비노출: level, l3_subtypes, 하위 유형 이름
# ---------------------------------------------------------------------------

HIDDEN_WORDS = {
    "level",
    "l2_subtype",
    "l3_subtypes",
    "L1",
    "L2",
    "L3",
    *{name for names in L2_SUBTYPES.values() for name in names},
}


def all_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [*value.keys(), *[s for v in value.values() for s in all_strings(v)]]
    if isinstance(value, list):
        return [s for v in value for s in all_strings(v)]
    return [value] if isinstance(value, str) else []


@pytest.mark.parametrize("family", list(Family))
def test_tool_outputs_and_alert_do_not_reveal_level_or_subtypes(family: Family) -> None:
    for seed in (*range(30), *range(400_000, 400_010)):
        incident = l3(family, seed)
        env = AgentEnv(Episode(incident))
        args = {"service_id": env.service_id}
        outputs = [env.call(spec.name, args) for spec in env.tool_specs()[:4]]
        keys_and_texts = [s for o in outputs for s in all_strings(o)]
        for text in [*keys_and_texts, make_alert(env.service_id)]:
            assert not any(word in text for word in HIDDEN_WORDS), (text, family, seed)
            assert "root_cause" not in text and "has_bug" not in text
        assert not any(word in incident.incident_id for word in HIDDEN_WORDS)


# ---------------------------------------------------------------------------
# 불변식 5: 결정성
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(Family))
def test_l3_is_deterministic(family: Family) -> None:
    for seed in L3_SEEDS:
        assert l3(family, seed) == l3(family, seed)


def test_l3_is_deterministic_across_processes() -> None:
    script = (
        "from opsworld.generator import generate_incident\n"
        "from opsworld.state import Family, Level\n"
        "seeds = [*range(20), *range(400_000, 400_005)]\n"
        "print([repr(generate_incident(f, s, level=Level.L3)) for f in Family for s in seeds])\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1
