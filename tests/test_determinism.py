"""시뮬레이터 출력의 성질: 결정성, 숨은 값 비노출, family 특징 유지."""

import os
import subprocess
import sys
from typing import Any

import pytest

from opsworld import constants as c
from opsworld.generator import generate_incident
from opsworld.state import DependencyStatus, Family, Incident, RootCause
from opsworld.tools import (
    Episode,
    get_dependency_health,
    get_recent_deployments,
    get_service_metrics,
    get_traffic_metrics,
)

SEEDS = range(100)
TOOLS = (get_service_metrics, get_recent_deployments, get_dependency_health, get_traffic_metrics)


def all_tool_outputs(incident: Incident) -> list[dict[str, Any]]:
    episode = Episode(incident)
    return [tool(episode, incident.service.service_id) for tool in TOOLS]


# ---------------------------------------------------------------------------
# 결정성
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(Family))
def test_same_family_and_seed_give_identical_incident(family: Family) -> None:
    for seed in SEEDS:
        assert generate_incident(family, seed) == generate_incident(family, seed)


@pytest.mark.parametrize("family", list(Family))
def test_same_family_and_seed_give_identical_tool_outputs(family: Family) -> None:
    for seed in SEEDS:
        first = all_tool_outputs(generate_incident(family, seed))
        second = all_tool_outputs(generate_incident(family, seed))
        assert first == second


@pytest.mark.parametrize("family", list(Family))
def test_seed_changes_the_incident(family: Family) -> None:
    incidents = {repr(generate_incident(family, seed)) for seed in SEEDS}
    assert len(incidents) == len(SEEDS)


def test_incidents_do_not_depend_on_process_hash_seed() -> None:
    """문자열 해시 무작위화(PYTHONHASHSEED)가 달라도 같은 장애가 나온다."""
    script = (
        "from opsworld.generator import generate_incident\n"
        "from opsworld.state import Family\n"
        "print([repr(generate_incident(f, s)) for f in Family for s in range(20)])\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1


def test_incident_ids_are_unique_across_families_and_seeds() -> None:
    ids = {generate_incident(family, seed).incident_id for family in Family for seed in SEEDS}
    assert len(ids) == len(Family) * len(SEEDS)


# ---------------------------------------------------------------------------
# 숨은 값 비노출
# ---------------------------------------------------------------------------

HIDDEN_KEYS = {"root_cause", "has_bug"}
ROOT_CAUSE_NAMES = {cause.value for cause in RootCause}


def walk(value: object) -> list[tuple[str | None, object]]:
    """중첩된 dict/list를 펼쳐 (키, 값) 목록으로 만든다."""
    items: list[tuple[str | None, object]] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            items.append((key, inner))
            items.extend(walk(inner))
    elif isinstance(value, list):
        for inner in value:
            items.append((None, inner))
            items.extend(walk(inner))
    return items


@pytest.mark.parametrize("family", list(Family))
def test_tool_outputs_have_no_hidden_keys(family: Family) -> None:
    for seed in SEEDS:
        for output in all_tool_outputs(generate_incident(family, seed)):
            keys = {key for key, _ in walk(output)}
            assert not keys & HIDDEN_KEYS


@pytest.mark.parametrize("family", list(Family))
def test_tool_outputs_do_not_mention_root_cause(family: Family) -> None:
    for seed in SEEDS:
        incident = generate_incident(family, seed)
        for output in all_tool_outputs(incident):
            strings = {v for _, v in walk(output) if isinstance(v, str)}
            assert not any(name in s for s in strings for name in ROOT_CAUSE_NAMES)


@pytest.mark.parametrize("family", list(Family))
def test_incident_id_does_not_mention_root_cause(family: Family) -> None:
    for seed in SEEDS:
        incident_id = generate_incident(family, seed).incident_id
        assert not any(name in incident_id for name in ROOT_CAUSE_NAMES)


# ---------------------------------------------------------------------------
# family 특징 (Ticket 001 §2 표). 시드는 수치만 바꾸고 특징은 유지한다.
# ---------------------------------------------------------------------------


def test_bad_deployment_features() -> None:
    for seed in SEEDS:
        incident = generate_incident(Family.BAD_DEPLOYMENT, seed)
        assert incident.root_cause is RootCause.BAD_DEPLOYMENT
        buggy = [d for d in incident.deployments if d.has_bug]
        assert len(buggy) == 1
        lead = incident.regression_started_at - buggy[0].deployed_at
        assert 0 <= lead <= c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN
        assert all(d.status is DependencyStatus.HEALTHY for d in incident.dependencies)
        traffic = incident.traffic
        assert traffic.current_rps < c.TRAFFIC_RULED_OUT_MULTIPLIER * traffic.baseline_rps


def test_dependency_failure_features() -> None:
    for seed in SEEDS:
        incident = generate_incident(Family.DEPENDENCY_FAILURE, seed)
        assert incident.root_cause is RootCause.DEPENDENCY_FAILURE
        degraded = [d for d in incident.dependencies if d.status is DependencyStatus.DEGRADED]
        assert len(degraded) == 1
        outage_started_at = degraded[0].outage_started_at
        assert outage_started_at is not None
        assert outage_started_at <= incident.regression_started_at
        assert_bug_free_distractor_deployment(incident)


def test_traffic_spike_features() -> None:
    for seed in SEEDS:
        incident = generate_incident(Family.TRAFFIC_SPIKE, seed)
        assert incident.root_cause is RootCause.TRAFFIC_SPIKE
        traffic = incident.traffic
        assert traffic.current_rps >= c.TRAFFIC_SPIKE_MULTIPLIER * traffic.baseline_rps
        assert incident.service.cpu_usage >= c.CPU_SATURATION_PERCENT
        assert all(d.status is DependencyStatus.HEALTHY for d in incident.dependencies)
        assert_bug_free_distractor_deployment(incident)


def assert_bug_free_distractor_deployment(incident: Incident) -> None:
    """최근 배포는 있지만 버그가 없고, 저하 시작 직전 창 밖에 있다."""
    assert incident.deployments
    for deployment in incident.deployments:
        assert not deployment.has_bug
        offset = incident.regression_started_at - deployment.deployed_at
        assert abs(offset) <= c.RECENT_DEPLOYMENT_WINDOW_MIN
        assert not 0 <= offset <= c.DEPLOY_BEFORE_REGRESSION_WINDOW_MIN
