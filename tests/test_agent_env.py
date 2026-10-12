"""AgentEnv 파사드: 정책이 볼 수 있는 경계."""

from typing import Any

import pytest

from opsworld.agent_env import AgentEnv, ToolSpec
from opsworld.generator import generate_incident
from opsworld.state import Family
from opsworld.tools import Episode

OBSERVATIONS = [
    "get_service_metrics",
    "get_recent_deployments",
    "get_dependency_health",
    "get_traffic_metrics",
]
ACTIONS = ["rollback_deployment", "scale_service", "observe_only"]


def new_env(family: Family = Family.DEPENDENCY_FAILURE, seed: int = 0) -> tuple[AgentEnv, Episode]:
    episode = Episode(generate_incident(family, seed))
    return AgentEnv(episode), episode


def args(env: AgentEnv) -> dict[str, Any]:
    return {"service_id": env.service_id}


def walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            keys.add(key)
            keys |= walk_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            keys |= walk_keys(inner)
    return keys


# ---------------------------------------------------------------------------
# 경계: Episode/Incident와 숨은 값에 닿는 공개 속성이 없다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["episode", "incident", "root_cause", "has_bug"])
def test_env_has_no_attribute_leading_to_hidden_state(name: str) -> None:
    env, _ = new_env()
    assert not hasattr(env, name)


def test_env_public_surface_is_exactly_the_facade() -> None:
    env, _ = new_env()
    public = {name for name in dir(env) if not name.startswith("_")}
    assert public == {"service_id", "tool_specs", "call", "closed"}


def test_service_id_matches_incident() -> None:
    env, episode = new_env()
    assert env.service_id == episode.incident.service.service_id


@pytest.mark.parametrize("family", list(Family))
def test_no_call_result_contains_hidden_keys(family: Family) -> None:
    for seed in range(20):
        for action in ACTIONS:
            env, _ = new_env(family, seed)
            results = [env.call(name, args(env)) for name in OBSERVATIONS]
            results.append(env.call(action, args(env)))
            for result in results:
                assert not walk_keys(result) & {"root_cause", "has_bug"}


# ---------------------------------------------------------------------------
# 도구 명세
# ---------------------------------------------------------------------------


def test_tool_specs_cover_four_observations_and_three_actions() -> None:
    env, _ = new_env()
    specs = env.tool_specs()
    assert [s.name for s in specs] == OBSERVATIONS + ACTIONS
    assert all(isinstance(s, ToolSpec) for s in specs)


def test_tool_specs_are_english_json_schema_with_service_id() -> None:
    env, _ = new_env()
    for spec in env.tool_specs():
        assert spec.description.isascii() and spec.description
        assert spec.parameters == {
            "type": "object",
            "properties": {"service_id": {"type": "string", "description": "Target service ID."}},
            "required": ["service_id"],
            "additionalProperties": False,
        }


def test_tool_specs_do_not_mention_root_causes() -> None:
    env, _ = new_env()
    text = " ".join(s.description.lower() for s in env.tool_specs())
    for word in ("bad_deployment", "dependency_failure", "traffic_spike", "has_bug", "root"):
        assert word not in text


def test_tool_specs_are_the_same_for_every_incident() -> None:
    specs = {repr(new_env(f, s)[0].tool_specs()) for f in Family for s in range(5)}
    assert len(specs) == 1


# ---------------------------------------------------------------------------
# 호출
# ---------------------------------------------------------------------------


def test_observation_call_returns_tool_output_and_records_it() -> None:
    env, episode = new_env()
    result = env.call("get_traffic_metrics", args(env))
    assert set(result) == {"service_id", "baseline_rps", "current_rps"}
    assert len(episode.ledger) == 1
    assert episode.ledger.observations[0].payload == result


def test_action_call_returns_recorded_and_closes_env() -> None:
    env, episode = new_env()
    env.call("get_service_metrics", args(env))
    assert env.call("observe_only", args(env)) == {"status": "recorded"}
    assert env.closed
    assert episode.action == "observe_only"
    assert episode.action_step == 1


def test_returned_dict_is_a_copy() -> None:
    env, episode = new_env()
    result = env.call("get_dependency_health", args(env))
    result["dependencies"][0]["status"] = "tampered"
    result["service_id"] = "tampered"
    payload = episode.ledger.observations[0].payload
    assert payload["service_id"] == env.service_id
    assert payload["dependencies"][0]["status"] != "tampered"


@pytest.mark.parametrize(
    ("name", "call_args"),
    [
        ("get_logs", {"service_id": "SERVICE"}),  # 없는 도구
        ("get_service_metrics", {}),  # 필수 인자 누락
        ("get_service_metrics", {"service_id": 42}),  # 타입 오류
        ("get_service_metrics", {"service_id": "SERVICE", "verbose": True}),  # 추가 인자
        ("get_service_metrics", {"service_id": "no-such-service"}),  # 대상 외 서비스
        ("rollback_deployment", {"service": "SERVICE"}),  # 조치 인자 이름 오류
    ],
)
def test_invalid_call_returns_error_and_records_nothing(
    name: str, call_args: dict[str, Any]
) -> None:
    env, episode = new_env()
    resolved = {k: (env.service_id if v == "SERVICE" else v) for k, v in call_args.items()}
    result = env.call(name, resolved)
    assert set(result) == {"error"} and isinstance(result["error"], str)
    assert len(episode.ledger) == 0
    assert not env.closed


def test_non_dict_args_return_error() -> None:
    env, episode = new_env()
    result = env.call("get_service_metrics", "checkout-api")  # type: ignore[arg-type]
    assert set(result) == {"error"}
    assert len(episode.ledger) == 0


@pytest.mark.parametrize("name", OBSERVATIONS + ACTIONS)
def test_call_after_action_returns_episode_closed(name: str) -> None:
    env, episode = new_env()
    env.call("observe_only", args(env))
    assert env.call(name, args(env)) == {"error": "episode closed"}
    assert len(episode.ledger) == 0
    assert episode.action == "observe_only"
