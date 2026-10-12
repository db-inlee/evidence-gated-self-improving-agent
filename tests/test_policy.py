"""Policy 프로토콜, PolicyTrace, ScriptedPolicy."""

import json

from evidencegate.policy import (
    RESULT_SUMMARY_MAX_CHARS,
    Policy,
    PolicyTrace,
    ScriptedPolicy,
    StopReason,
)
from evidencegate.skills import Skill
from opsworld.agent_env import AgentEnv
from opsworld.generator import generate_incident
from opsworld.state import Family
from opsworld.tools import Episode

SKILL = Skill.from_text("# Skill\n")
ALERT = "Alert: test"


def new_env(family: Family = Family.BAD_DEPLOYMENT, seed: int = 0) -> tuple[AgentEnv, Episode]:
    episode = Episode(generate_incident(family, seed))
    return AgentEnv(episode), episode


def run(policy: Policy, env: AgentEnv) -> PolicyTrace:
    return policy.run(env, SKILL, ALERT, llm_seed=123)


def test_policy_id_has_scripted_prefix() -> None:
    assert ScriptedPolicy(["observe_only"], name="A").policy_id == "scripted:A"


def test_replays_calls_in_order_and_stops_at_action() -> None:
    env, episode = new_env()
    policy = ScriptedPolicy(
        ["get_service_metrics", "get_recent_deployments", "rollback_deployment"], name="B"
    )
    trace = run(policy, env)

    assert [t.tool for t in trace.turns] == [
        "get_service_metrics",
        "get_recent_deployments",
        "rollback_deployment",
    ]
    assert trace.stop_reason is StopReason.ACTION
    assert trace.format_errors == 0
    assert not any(t.format_error for t in trace.turns)
    assert episode.action == "rollback_deployment"
    assert trace.llm_config is None and trace.usage is None


def test_default_args_target_the_alerted_service() -> None:
    env, _ = new_env()
    trace = run(ScriptedPolicy(["get_traffic_metrics"], name="x"), env)
    assert trace.turns[0].args == {"service_id": env.service_id}


def test_calls_after_the_action_are_not_replayed() -> None:
    env, episode = new_env()
    trace = run(ScriptedPolicy(["observe_only", "get_service_metrics"], name="x"), env)
    assert len(trace.turns) == 1
    assert trace.stop_reason is StopReason.ACTION
    assert len(episode.ledger) == 0


def test_script_without_action_stops_with_no_tool_call() -> None:
    env, episode = new_env()
    trace = run(ScriptedPolicy(["get_service_metrics"], name="x"), env)
    assert trace.stop_reason is StopReason.NO_TOOL_CALL
    assert episode.action is None


def test_unknown_tool_counts_as_one_format_error_and_leaves_ledger_unchanged() -> None:
    env, episode = new_env()
    trace = run(ScriptedPolicy(["get_logs", "get_service_metrics"], name="x"), env)
    assert trace.format_errors == 1
    assert [t.format_error for t in trace.turns] == [True, False]
    assert "error" in json.loads(trace.turns[0].result_summary)
    assert len(episode.ledger) == 1


def test_bad_arguments_count_as_one_format_error_and_leave_ledger_unchanged() -> None:
    env, episode = new_env()
    policy = ScriptedPolicy([("get_service_metrics", {"service": env.service_id})], name="x")
    trace = run(policy, env)
    assert trace.format_errors == 1
    assert trace.turns[0].args == {"service": env.service_id}
    assert len(episode.ledger) == 0


def test_result_summary_is_compact_json_within_limit() -> None:
    env, _ = new_env(Family.DEPENDENCY_FAILURE)
    trace = run(ScriptedPolicy(["get_dependency_health", "get_service_metrics"], name="x"), env)
    for turn in trace.turns:
        assert len(turn.result_summary) <= RESULT_SUMMARY_MAX_CHARS
    assert json.loads(trace.turns[1].result_summary)["service_id"] == env.service_id


def test_long_results_are_truncated() -> None:
    env, _ = new_env()
    long_name = "x" * (RESULT_SUMMARY_MAX_CHARS * 2)
    trace = run(ScriptedPolicy([long_name], name="x"), env)
    summary = trace.turns[0].result_summary
    assert len(summary) == RESULT_SUMMARY_MAX_CHARS
    assert summary.endswith("...")
