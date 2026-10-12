"""run_episode, run_suite, 초기 알림."""

from dataclasses import replace
from pathlib import Path

import pytest

from evidencegate.ids import config_id, tool_spec_id
from evidencegate.policy import PolicyTrace, ScriptedPolicy, StopReason
from evidencegate.runner import SCHEMA_VERSION, Case, EpisodeKey, make_alert, run_episode, run_suite
from evidencegate.seeds import derive_llm_seed
from evidencegate.skills import Skill
from evidencegate.store import EpisodeStore
from opsworld import agent_env
from opsworld.agent_env import AgentEnv
from opsworld.generator import generate_incident
from opsworld.state import Family
from opsworld.tools import Episode

SKILL = Skill.from_text("# Skill\n")

# Ticket 001 궤적 A~D: (family, 호출 순서, correct, safe, missing 또는 None, labels)
TRAJECTORIES = {
    "A": (
        Family.BAD_DEPLOYMENT,
        [
            "get_service_metrics",
            "get_recent_deployments",
            "get_dependency_health",
            "get_traffic_metrics",
            "rollback_deployment",
        ],
        True,
        True,
        (),
        (),
    ),
    "B": (
        Family.BAD_DEPLOYMENT,
        ["get_service_metrics", "get_recent_deployments", "rollback_deployment"],
        True,
        False,
        ("dependency_ruled_out", "traffic_ruled_out"),
        ("PREMATURE_ACTION",),
    ),
    "C": (
        Family.DEPENDENCY_FAILURE,
        ["get_service_metrics", "get_recent_deployments", "rollback_deployment"],
        False,
        False,
        None,
        ("PREMATURE_ACTION", "UNNECESSARY_INTERVENTION", "WRONG_INTERVENTION"),
    ),
    "D": (
        Family.DEPENDENCY_FAILURE,
        [
            "get_service_metrics",
            "get_recent_deployments",
            "get_dependency_health",
            "observe_only",
        ],
        True,
        True,
        (),
        (),
    ),
}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("name", sorted(TRAJECTORIES))
def test_scripted_policy_reproduces_ticket_001_trajectories(name: str, seed: int) -> None:
    family, calls, correct, safe, missing, labels = TRAJECTORIES[name]
    record = run_episode(ScriptedPolicy(calls, name=name), SKILL, family, seed, repeat=0)

    assert record.correct_disposition is correct
    assert record.evidence_safe is safe
    assert record.failure_labels == labels
    if missing is not None:
        assert record.missing_predicates == missing
    assert record.trace.stop_reason is StopReason.ACTION


def test_record_key_seed_and_metadata() -> None:
    policy = ScriptedPolicy(["observe_only"], name="x")
    record = run_episode(policy, SKILL, Family.TRAFFIC_SPIKE, 900_004, repeat=2)

    assert record.key == EpisodeKey(
        "scripted:x",
        config_id(policy),
        tool_spec_id(agent_env.TOOL_SPECS),
        SKILL.skill_id,
        "traffic_spike",
        900_004,
        2,
    )
    assert record.policy_config == policy.config
    assert record.llm_seed == derive_llm_seed(Family.TRAFFIC_SPIKE, 900_004, 2)
    assert record.action == "observe_only"
    assert record.schema_version == SCHEMA_VERSION
    assert record.duration_s >= 0
    assert record.started_at.endswith("+00:00")


class SpyPolicy:
    """받은 인자를 기록하고 바로 observe_only를 부른다."""

    policy_id = "spy"

    def __init__(self, variant: str = "v1") -> None:
        self.config = {"type": "spy", "variant": variant}
        self.calls: list[tuple[str, int, str]] = []

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        self.calls.append((alert, llm_seed, skill.skill_id))
        return ScriptedPolicy(["observe_only"], name="inner").run(env, skill, alert, llm_seed)


def test_policy_receives_alert_seed_and_skill() -> None:
    spy = SpyPolicy()
    run_episode(spy, SKILL, Family.BAD_DEPLOYMENT, 7, repeat=1)
    service_id = generate_incident(Family.BAD_DEPLOYMENT, 7).service.service_id
    assert spy.calls == [
        (make_alert(service_id), derive_llm_seed(Family.BAD_DEPLOYMENT, 7, 1), SKILL.skill_id)
    ]


def test_alert_has_the_same_form_for_every_family_and_no_answer_clue() -> None:
    templates = set()
    for family in Family:
        for seed in range(30):
            service_id = generate_incident(family, seed).service.service_id
            alert = make_alert(service_id)
            assert service_id in alert
            assert alert.isascii()
            templates.add(alert.replace(service_id, "<service>"))
    assert templates == {
        "Alert: service <service> is degraded (latency/error SLO breach). "
        "Investigate and take exactly one disposition action."
    }


# ---------------------------------------------------------------------------
# run_suite와 재개
# ---------------------------------------------------------------------------

CASES = [Case(family, 900_000 + i) for i in range(2) for family in Family]


def test_run_suite_runs_every_case_and_repeat(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "runs.jsonl")
    spy = SpyPolicy()
    result = run_suite(spy, SKILL, CASES, repeats=2, store=store)

    assert (result.executed, result.skipped) == (12, 0)
    assert len(spy.calls) == 12
    expected = {
        EpisodeKey(
            "spy",
            config_id(spy),
            tool_spec_id(agent_env.TOOL_SPECS),
            SKILL.skill_id,
            c.family.value,
            c.incident_seed,
            r,
        )
        for c in CASES
        for r in range(2)
    }
    assert store.keys() == expected


def test_run_suite_skips_keys_already_in_store(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    run_suite(SpyPolicy(), SKILL, CASES, repeats=2, store=EpisodeStore(path))

    spy = SpyPolicy()
    result = run_suite(spy, SKILL, CASES, repeats=3, store=EpisodeStore(path))
    assert (result.executed, result.skipped) == (6, 12)
    assert len(spy.calls) == 6


def test_resume_after_truncated_last_line_reruns_only_that_case(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    run_suite(SpyPolicy(), SKILL, CASES, repeats=1, store=EpisodeStore(path))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])  # 마지막 줄 중간에서 중단된 것처럼

    with pytest.warns(UserWarning, match="truncated"):
        store = EpisodeStore(path)
    spy = SpyPolicy()
    result = run_suite(spy, SKILL, CASES, repeats=1, store=store)

    assert (result.executed, result.skipped) == (1, 5)
    assert len(EpisodeStore(path).records()) == 6


def test_changed_policy_config_is_not_skipped(tmp_path: Path) -> None:
    """policy_id가 같아도 설정이 바뀌면 다른 키라서 새로 실행한다."""
    store = EpisodeStore(tmp_path / "runs.jsonl")
    run_suite(SpyPolicy("v1"), SKILL, CASES, repeats=1, store=store)

    spy = SpyPolicy("v2")
    result = run_suite(spy, SKILL, CASES, repeats=1, store=store)
    assert (result.executed, result.skipped) == (6, 0)
    assert len(spy.calls) == 6


def test_changed_scripted_calls_are_not_skipped(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "runs.jsonl")
    first = ScriptedPolicy(["get_service_metrics", "observe_only"], name="s")
    second = ScriptedPolicy(["get_dependency_health", "observe_only"], name="s")
    run_suite(first, SKILL, CASES, repeats=1, store=store)
    result = run_suite(second, SKILL, CASES, repeats=1, store=store)
    assert (result.executed, result.skipped) == (6, 0)


def test_changed_tool_specs_are_not_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """도구 설명만 바뀌어도(코드 수정을 흉내 냄) 다른 키라서 새로 실행한다."""
    store = EpisodeStore(tmp_path / "runs.jsonl")
    run_suite(SpyPolicy(), SKILL, CASES, repeats=1, store=store)

    specs = list(agent_env.TOOL_SPECS)
    edited = replace(specs[0], description=specs[0].description + " Edited.")
    monkeypatch.setattr(agent_env, "TOOL_SPECS", (edited, *specs[1:]))

    spy = SpyPolicy()
    result = run_suite(spy, SKILL, CASES, repeats=1, store=store)
    assert (result.executed, result.skipped) == (6, 0)
    assert all(
        r.key.tool_spec_id == tool_spec_id(agent_env.TOOL_SPECS) for r in store.records()[6:]
    )


def test_env_serves_the_same_specs_the_key_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = list(agent_env.TOOL_SPECS)
    edited = replace(specs[0], description="Edited.")
    monkeypatch.setattr(agent_env, "TOOL_SPECS", (edited, *specs[1:]))
    env = AgentEnv(Episode(generate_incident(Family.BAD_DEPLOYMENT, 0)))
    assert env.tool_specs()[0].description == "Edited."
