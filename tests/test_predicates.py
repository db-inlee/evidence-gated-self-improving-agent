"""증거 술어 단위 테스트: 임계값 경계, 관측 없음, 원장 슬라이싱.

payload 키 이름은 `tools.py`의 출력과 같다. 임계값은 Ticket 001 §5 표의 숫자를 그대로 쓴다.
"""

from typing import Any

import pytest

from opsworld.ledger import Ledger, Observation, observations_before
from opsworld.predicates import derive_predicates
from opsworld.tools import ToolName

REGRESSION_AT = 300


def metrics(
    *,
    p95_ms: float = 100.0,
    baseline_p95_ms: float = 100.0,
    error_rate: float = 0.0,
    baseline_error_rate: float = 0.0,
    cpu: float = 50.0,
) -> tuple[str, dict[str, Any]]:
    return ToolName.SERVICE_METRICS, {
        "service_id": "svc",
        "baseline_p95_ms": baseline_p95_ms,
        "p95_ms": p95_ms,
        "baseline_error_rate": baseline_error_rate,
        "error_rate": error_rate,
        "cpu_usage_percent": cpu,
        "regression_started_at": REGRESSION_AT,
    }


def deployments(*deployed_ats: int) -> tuple[str, dict[str, Any]]:
    return ToolName.RECENT_DEPLOYMENTS, {
        "service_id": "svc",
        "deployments": [
            {"deployment_id": f"d{i}", "version": "v1.0.0", "deployed_at": t}
            for i, t in enumerate(deployed_ats)
        ],
    }


def dependencies(*entries: tuple[str, int | None]) -> tuple[str, dict[str, Any]]:
    return ToolName.DEPENDENCY_HEALTH, {
        "service_id": "svc",
        "dependencies": [
            {"dependency_id": f"dep{i}", "status": status, "outage_started_at": started}
            for i, (status, started) in enumerate(entries)
        ],
    }


def traffic(current: float, baseline: float = 100.0) -> tuple[str, dict[str, Any]]:
    return ToolName.TRAFFIC_METRICS, {
        "service_id": "svc",
        "baseline_rps": baseline,
        "current_rps": current,
    }


def preds(*observed: tuple[str, dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        derive_predicates(
            [Observation(i, tool, payload) for i, (tool, payload) in enumerate(observed)]
        )
    )


HEALTHY = ("healthy", None)


# ---------------------------------------------------------------------------
# 관측이 없으면 술어는 성립하지 않는다
# ---------------------------------------------------------------------------


def test_no_observations_give_no_predicates() -> None:
    assert preds() == frozenset()


def test_dependency_predicates_absent_without_dependency_observation() -> None:
    """의존성 도구를 부르지 않으면 ruled_out과 failure_confirmed가 둘 다 없다."""
    result = preds(metrics(p95_ms=500.0), deployments(REGRESSION_AT - 5), traffic(100.0))
    assert "dependency_ruled_out" not in result
    assert "dependency_failure_confirmed" not in result


def test_traffic_predicates_absent_without_traffic_observation() -> None:
    result = preds(metrics(p95_ms=500.0), dependencies(HEALTHY))
    assert "traffic_ruled_out" not in result
    assert "traffic_spike_confirmed" not in result


def test_deployment_predicates_need_metrics_for_regression_time() -> None:
    assert preds(deployments(REGRESSION_AT - 5)) == frozenset()


def test_dependency_failure_needs_metrics_for_regression_time() -> None:
    assert preds(dependencies(("degraded", REGRESSION_AT - 5))) == frozenset()


# ---------------------------------------------------------------------------
# incident_confirmed, capacity_saturated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p95_ms", "expected"),
    [(200.0, True), (199.9, False)],
)
def test_incident_confirmed_by_p95_at_two_times_baseline(p95_ms: float, expected: bool) -> None:
    assert ("incident_confirmed" in preds(metrics(p95_ms=p95_ms))) is expected


@pytest.mark.parametrize(
    ("error_rate", "expected"),
    [(0.02, True), (0.019, False)],
)
def test_incident_confirmed_by_error_rate_two_points_above_baseline(
    error_rate: float, expected: bool
) -> None:
    result = preds(metrics(error_rate=error_rate, baseline_error_rate=0.0))
    assert ("incident_confirmed" in result) is expected


def test_healthy_metrics_do_not_confirm_incident() -> None:
    assert "incident_confirmed" not in preds(metrics())


@pytest.mark.parametrize(("cpu", "expected"), [(85.0, True), (84.9, False)])
def test_capacity_saturated_at_85_percent(cpu: float, expected: bool) -> None:
    assert ("capacity_saturated" in preds(metrics(cpu=cpu))) is expected


# ---------------------------------------------------------------------------
# recent_deployment_found, deploy_before_regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(-30, True), (-31, False), (30, True), (31, False), (0, True)],
)
def test_recent_deployment_within_30_minutes_either_side(offset: int, expected: bool) -> None:
    # offset > 0 이면 저하 시작 이후 배포
    result = preds(metrics(), deployments(REGRESSION_AT + offset))
    assert ("recent_deployment_found" in result) is expected


@pytest.mark.parametrize(
    ("lead", "expected"),
    [(0, True), (15, True), (16, False), (-1, False)],
)
def test_deploy_before_regression_within_0_to_15_minutes_before(lead: int, expected: bool) -> None:
    # lead = 저하 시작 - 배포 시각. 음수면 저하 시작 이후 배포
    result = preds(metrics(), deployments(REGRESSION_AT - lead))
    assert ("deploy_before_regression" in result) is expected


def test_any_matching_deployment_is_enough() -> None:
    result = preds(metrics(), deployments(REGRESSION_AT - 200, REGRESSION_AT - 5))
    assert {"recent_deployment_found", "deploy_before_regression"} <= result


def test_no_deployments_listed() -> None:
    result = preds(metrics(), deployments())
    assert "recent_deployment_found" not in result
    assert "deploy_before_regression" not in result


# ---------------------------------------------------------------------------
# dependency_ruled_out, dependency_failure_confirmed
# ---------------------------------------------------------------------------


def test_all_healthy_dependencies_rule_out_dependency() -> None:
    result = preds(metrics(), dependencies(HEALTHY, HEALTHY))
    assert "dependency_ruled_out" in result
    assert "dependency_failure_confirmed" not in result


@pytest.mark.parametrize(
    ("outage_started_at", "expected"),
    [(REGRESSION_AT - 10, True), (REGRESSION_AT, True), (REGRESSION_AT + 1, False)],
)
def test_dependency_failure_confirmed_when_outage_starts_no_later_than_regression(
    outage_started_at: int, expected: bool
) -> None:
    result = preds(metrics(), dependencies(HEALTHY, ("degraded", outage_started_at)))
    assert ("dependency_failure_confirmed" in result) is expected
    assert "dependency_ruled_out" not in result


def test_degraded_dependency_without_outage_time_is_not_confirmed() -> None:
    result = preds(metrics(), dependencies(("degraded", None)))
    assert "dependency_failure_confirmed" not in result
    assert "dependency_ruled_out" not in result


# ---------------------------------------------------------------------------
# traffic_ruled_out, traffic_spike_confirmed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "ruled_out", "spike"),
    [
        (149.9, True, False),
        (150.0, False, False),  # 1.5~3배: 어느 쪽도 아님
        (299.9, False, False),
        (300.0, False, True),
    ],
)
def test_traffic_thresholds(current: float, ruled_out: bool, spike: bool) -> None:
    result = preds(traffic(current, baseline=100.0))
    assert ("traffic_ruled_out" in result) is ruled_out
    assert ("traffic_spike_confirmed" in result) is spike


# ---------------------------------------------------------------------------
# 같은 도구 반복 호출
# ---------------------------------------------------------------------------


def test_latest_observation_of_a_tool_wins() -> None:
    result = preds(traffic(500.0), traffic(100.0))
    assert "traffic_ruled_out" in result
    assert "traffic_spike_confirmed" not in result


# ---------------------------------------------------------------------------
# 원장과 observations_before
# ---------------------------------------------------------------------------


def filled_ledger(count: int) -> Ledger:
    ledger = Ledger()
    for _ in range(count):
        ledger.append(*traffic(100.0))
    return ledger


def test_ledger_numbers_observations_from_zero() -> None:
    assert [o.step for o in filled_ledger(3).observations] == [0, 1, 2]


@pytest.mark.parametrize(("action_step", "expected_steps"), [(0, []), (2, [0, 1]), (3, [0, 1, 2])])
def test_observations_before_keeps_only_earlier_steps(
    action_step: int, expected_steps: list[int]
) -> None:
    before = observations_before(filled_ledger(3), action_step)
    assert [o.step for o in before] == expected_steps


def test_observations_after_action_step_are_dropped() -> None:
    ledger = Ledger()
    ledger.append(*traffic(100.0))
    ledger.append(*traffic(500.0))  # 조치(action_step=1) 뒤 관측
    assert preds_from(observations_before(ledger, 1)) == frozenset({"traffic_ruled_out"})


def test_ledger_keeps_its_own_copy_of_payload() -> None:
    ledger = Ledger()
    tool, payload = traffic(100.0)
    ledger.append(tool, payload)
    payload["current_rps"] = 999.0
    assert ledger.observations[0].payload["current_rps"] == 100.0


def preds_from(observations: tuple[Observation, ...]) -> frozenset[str]:
    return frozenset(derive_predicates(observations))


# ---------------------------------------------------------------------------
# 경계값의 부동소수점 오차: 도구 출력은 반올림된 십진 값이므로, 표에서 경계에 놓인 값은
# 이진 부동소수점 오차와 무관하게 표대로 판정돼야 한다.
# ---------------------------------------------------------------------------

BASELINE_ERROR_RATES = [b / 10000 for b in range(200)]
BASELINE_VALUES = [b / 10 for b in range(1000, 3000)]


def test_error_rate_exactly_two_points_above_baseline_confirms_incident() -> None:
    for baseline in BASELINE_ERROR_RATES:
        error_rate = round(baseline + 0.02, 4)
        result = preds(metrics(error_rate=error_rate, baseline_error_rate=baseline))
        assert "incident_confirmed" in result, (baseline, error_rate)


def test_p95_exactly_two_times_baseline_confirms_incident() -> None:
    for baseline in BASELINE_VALUES:
        p95 = round(baseline * 2, 1)
        result = preds(metrics(p95_ms=p95, baseline_p95_ms=baseline))
        assert "incident_confirmed" in result, (baseline, p95)


def test_traffic_exactly_three_times_baseline_is_spike() -> None:
    for baseline in BASELINE_VALUES:
        current = round(baseline * 3, 1)
        assert "traffic_spike_confirmed" in preds(traffic(current, baseline)), (baseline, current)


def test_traffic_exactly_one_and_half_times_baseline_is_not_ruled_out() -> None:
    for baseline in BASELINE_VALUES:
        current = round(baseline * 1.5, 2)
        assert "traffic_ruled_out" not in preds(traffic(current, baseline)), (baseline, current)
