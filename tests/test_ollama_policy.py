"""OllamaPolicy. 실제 Ollama 없이 httpx MockTransport로만 테스트한다."""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.ids import config_id
from evidencegate.ollama_policy import (
    INSTRUCTIONS,
    NUDGE,
    OllamaError,
    OllamaPolicy,
    OllamaSettings,
)
from evidencegate.policy import StopReason
from evidencegate.runner import EpisodeRecord, make_alert, run_episode
from evidencegate.seeds import derive_llm_seed
from evidencegate.skills import load_skill
from fake_ollama import (
    DIGEST,
    MODEL,
    FakeOllama,
    model_entry,
    reply_text,
    reply_tools,
    tool_call,
)
from opsworld import agent_env
from opsworld.generator import generate_incident
from opsworld.state import Family

S0 = load_skill(Path(__file__).resolve().parents[1] / "skills" / "S0" / "SKILL.md")


def connect(fake: FakeOllama, model: str = MODEL, **overrides: Any) -> OllamaPolicy:
    settings = OllamaSettings.for_model(model, **overrides)
    return OllamaPolicy.connect(settings, fake.client(), machine="test-mac", chip="Apple M3")


def run(
    fake: FakeOllama,
    family: Family = Family.BAD_DEPLOYMENT,
    seed: int = 900_000,
    repeat: int = 0,
    policy: OllamaPolicy | None = None,
) -> EpisodeRecord:
    return run_episode(policy or connect(fake), S0, family, seed, repeat)


def svc(family: Family = Family.BAD_DEPLOYMENT, seed: int = 900_000) -> dict[str, str]:
    return {"service_id": generate_incident(family, seed).service.service_id}


# ---------------------------------------------------------------------------
# 설정과 config_id
# ---------------------------------------------------------------------------


def test_think_defaults_per_model() -> None:
    assert OllamaSettings.for_model("gemma4:12b").think is False
    assert OllamaSettings.for_model("gpt-oss:20b").think == "low"
    with pytest.raises(ValueError):
        OllamaSettings.for_model("llama3:8b")
    assert OllamaSettings.for_model("llama3:8b", think=False).think is False


def test_fixed_generation_settings() -> None:
    s = OllamaSettings.for_model(MODEL)
    assert (s.temperature, s.num_ctx, s.keep_alive) == (0.7, 8192, "60m")
    assert (s.max_tool_calls, s.max_consecutive_text_turns, s.max_consecutive_format_errors) == (
        8,
        2,
        3,
    )


def test_connect_reads_version_digest_and_quantization() -> None:
    policy = connect(FakeOllama([]))
    assert policy.policy_id == "ollama:gemma4:12b"
    config = policy.config
    assert config["type"] == "ollama"
    assert config["model"] == MODEL
    assert config["model_digest"] == DIGEST
    assert config["quantization"] == "Q4_K_M"
    assert config["ollama_version"] == "0.33.3"
    assert config["options"] == {"temperature": 0.7, "num_ctx": 8192}
    assert config["think"] is False
    assert config["stop"] == {"max_tool_calls": 8, "max_consecutive_text_turns": 2}
    assert config["format_error_policy"] == {
        "feedback": "error_as_tool_result",
        "max_consecutive_format_errors": 3,
    }
    fixed = json.dumps({"instructions": INSTRUCTIONS, "nudge": NUDGE}, sort_keys=True)
    assert config["instructions_sha256"] == hashlib.sha256(fixed.encode()).hexdigest()


def test_connect_fails_when_model_is_not_pulled() -> None:
    with pytest.raises(OllamaError, match="not found"):
        connect(FakeOllama([], models=[model_entry("gpt-oss:20b", "sha256:x")]))


@pytest.mark.parametrize(
    "change",
    [
        {"models": [model_entry(MODEL, "sha256:other")]},
        {"version": "0.34.0"},
        {"models": [model_entry(MODEL, DIGEST, quant="Q8_0")]},
    ],
)
def test_config_id_changes_with_server_side_model_or_engine(change: dict[str, Any]) -> None:
    base = config_id(connect(FakeOllama([])))
    assert config_id(connect(FakeOllama([], **change))) != base


@pytest.mark.parametrize(
    "override",
    [{"temperature": 0.2}, {"num_ctx": 4096}, {"max_tool_calls": 6}, {"think": "low"}],
)
def test_config_id_changes_with_any_setting(override: dict[str, Any]) -> None:
    base = config_id(connect(FakeOllama([])))
    assert config_id(connect(FakeOllama([]), **override)) != base


def test_keep_alive_does_not_change_config_id() -> None:
    """keep_alive는 출력에 영향이 없으므로 키에서 뺀다."""
    base = config_id(connect(FakeOllama([])))
    assert config_id(connect(FakeOllama([]), keep_alive="5m")) == base


# ---------------------------------------------------------------------------
# 요청 형식
# ---------------------------------------------------------------------------


def test_first_request_has_fixed_prompt_tools_and_options() -> None:
    fake = FakeOllama([reply_tools(tool_call("observe_only", svc()))])
    run(fake, repeat=2)

    request = fake.chat_requests[0]
    assert request["model"] == MODEL
    assert request["stream"] is False
    assert request["think"] is False
    assert request["keep_alive"] == "60m"
    assert request["options"] == {
        "temperature": 0.7,
        "seed": derive_llm_seed(Family.BAD_DEPLOYMENT, 900_000, 2),
        "num_ctx": 8192,
    }
    system, user = request["messages"]
    assert system == {"role": "system", "content": INSTRUCTIONS + "\n\n" + S0.text}
    assert user == {"role": "user", "content": make_alert(svc()["service_id"])}
    assert [t["function"]["name"] for t in request["tools"]] == [
        s.name for s in agent_env.TOOL_SPECS
    ]
    first = request["tools"][0]
    assert first["type"] == "function"
    assert first["function"]["description"] == agent_env.TOOL_SPECS[0].description
    assert first["function"]["parameters"] == agent_env.TOOL_SPECS[0].parameters


def test_gpt_oss_sends_think_low() -> None:
    fake = FakeOllama(
        [reply_tools(tool_call("observe_only", svc()))],
        models=[model_entry("gpt-oss:20b", "sha256:g")],
    )
    run(fake, policy=connect(fake, "gpt-oss:20b"))
    assert fake.chat_requests[0]["think"] == "low"


def test_requests_never_carry_hidden_values() -> None:
    calls = [
        "get_service_metrics",
        "get_recent_deployments",
        "get_dependency_health",
        "get_traffic_metrics",
    ]
    for family in Family:
        fake = FakeOllama(
            [reply_tools(tool_call(c, svc(family))) for c in calls]
            + [reply_tools(tool_call("observe_only", svc(family)))]
        )
        run(fake, family=family)
        text = json.dumps(fake.chat_requests)
        for hidden in ("root_cause", "has_bug", *[f.value for f in Family]):
            assert hidden not in text, (family, hidden)


# ---------------------------------------------------------------------------
# 도구 호출 루프와 종료 조건
# ---------------------------------------------------------------------------


def test_tool_loop_feeds_results_back_and_stops_at_action() -> None:
    fake = FakeOllama(
        [
            reply_tools(tool_call("get_service_metrics", svc())),
            reply_tools(tool_call("get_recent_deployments", svc())),
            reply_tools(tool_call("rollback_deployment", svc())),
        ]
    )
    record = run(fake)

    assert record.trace.stop_reason is StopReason.ACTION
    assert [t.tool for t in record.trace.turns] == [
        "get_service_metrics",
        "get_recent_deployments",
        "rollback_deployment",
    ]
    assert record.action == "rollback_deployment"
    assert record.correct_disposition is True
    assert record.evidence_safe is False  # 궤적 B와 같은 조사
    assert len(fake.chat_requests) == 3

    second = fake.chat_requests[1]["messages"]
    assert second[2]["role"] == "assistant"
    assert second[2]["tool_calls"][0]["function"]["name"] == "get_service_metrics"
    assert second[3]["role"] == "tool"
    assert second[3]["tool_name"] == "get_service_metrics"
    assert json.loads(second[3]["content"])["service_id"] == svc()["service_id"]


def test_arguments_given_as_json_string_are_parsed() -> None:
    fake = FakeOllama([reply_tools(tool_call("observe_only", json.dumps(svc())))])
    record = run(fake)
    assert record.action == "observe_only"
    assert record.trace.format_errors == 0


def test_multiple_calls_in_one_reply_run_in_order_and_stop_at_action() -> None:
    fake = FakeOllama(
        [
            reply_tools(
                tool_call("get_service_metrics", svc()),
                tool_call("observe_only", svc()),
                tool_call("get_traffic_metrics", svc()),
            )
        ]
    )
    record = run(fake)
    assert [t.tool for t in record.trace.turns] == ["get_service_metrics", "observe_only"]
    assert record.trace.stop_reason is StopReason.ACTION


def test_stops_after_max_tool_calls() -> None:
    fake = FakeOllama([reply_tools(tool_call("get_service_metrics", svc())) for _ in range(20)])
    record = run(fake)
    assert record.trace.stop_reason is StopReason.MAX_STEPS
    assert len(record.trace.turns) == 8
    assert record.action is None
    assert record.failure_labels == ("NO_DISPOSITION",)


def test_two_text_only_replies_in_a_row_stop() -> None:
    fake = FakeOllama([reply_text("Let me think."), reply_text("Still thinking.")])
    record = run(fake)
    assert record.trace.stop_reason is StopReason.NO_TOOL_CALL
    assert record.trace.turns == ()
    second = fake.chat_requests[1]["messages"]
    assert second[2] == {"role": "assistant", "content": "Let me think."}
    assert second[3] == {"role": "user", "content": NUDGE}


def test_one_text_reply_then_tool_call_continues() -> None:
    fake = FakeOllama(
        [
            reply_text("Investigating."),
            reply_tools(tool_call("get_service_metrics", svc())),
            reply_text("Hmm."),
            reply_tools(tool_call("observe_only", svc())),
        ]
    )
    record = run(fake)
    assert record.trace.stop_reason is StopReason.ACTION
    assert len(record.trace.turns) == 2


# ---------------------------------------------------------------------------
# 형식 오류
# ---------------------------------------------------------------------------


def test_unknown_tool_is_fed_back_as_tool_error_and_counted() -> None:
    fake = FakeOllama(
        [
            reply_tools(tool_call("get_logs", svc())),
            reply_tools(tool_call("observe_only", svc())),
        ]
    )
    record = run(fake)
    assert record.trace.format_errors == 1
    assert record.trace.turns[0].format_error is True
    feedback = fake.chat_requests[1]["messages"][3]
    assert feedback["role"] == "tool"
    assert "error" in json.loads(feedback["content"])


def test_three_consecutive_format_errors_stop_with_error() -> None:
    fake = FakeOllama([reply_tools(tool_call("get_logs", svc())) for _ in range(5)])
    record = run(fake)
    assert record.trace.stop_reason is StopReason.ERROR
    assert record.trace.format_errors == 3
    assert len(record.trace.turns) == 3


def test_valid_call_resets_consecutive_format_errors() -> None:
    fake = FakeOllama(
        [
            reply_tools(tool_call("get_logs", svc())),
            reply_tools(tool_call("get_logs", svc())),
            reply_tools(tool_call("get_service_metrics", svc())),
            reply_tools(tool_call("get_logs", svc())),
            reply_tools(tool_call("get_logs", svc())),
            reply_tools(tool_call("observe_only", svc())),
        ]
    )
    record = run(fake)
    assert record.trace.stop_reason is StopReason.ACTION
    assert record.trace.format_errors == 4


def test_unparseable_arguments_are_a_format_error_without_calling_env() -> None:
    fake = FakeOllama(
        [
            reply_tools(tool_call("get_service_metrics", "{not json")),
            reply_tools(tool_call("observe_only", svc())),
        ]
    )
    record = run(fake)
    first = record.trace.turns[0]
    assert first.format_error is True
    assert first.args == {"_raw": "{not json"}
    assert record.missing_predicates == ("dependency_failure_confirmed", "incident_confirmed")


def test_server_side_tool_parse_error_counts_as_format_error() -> None:
    parse_error = httpx.Response(500, json={"error": "error parsing tool call: raw='{bad'"})
    fake = FakeOllama([parse_error, reply_tools(tool_call("observe_only", svc()))])
    record = run(fake)
    assert record.trace.format_errors == 1
    assert record.trace.turns[0].tool == "<unparsed>"
    assert record.action == "observe_only"
    assert fake.chat_requests[1]["messages"][2]["role"] == "tool"


def test_other_server_errors_abort_the_episode() -> None:
    fake = FakeOllama([httpx.Response(500, json={"error": "model runner crashed"})])
    with pytest.raises(OllamaError, match="crashed"):
        run(fake)


# ---------------------------------------------------------------------------
# 기록: llm_config, usage
# ---------------------------------------------------------------------------


def test_llm_config_records_model_engine_seed_and_machine() -> None:
    fake = FakeOllama([reply_tools(tool_call("observe_only", svc()))])
    record = run(fake, repeat=1)
    assert record.trace.llm_config == {
        "model": MODEL,
        "model_digest": DIGEST,
        "quantization": "Q4_K_M",
        "ollama_version": "0.33.3",
        "temperature": 0.7,
        "seed": derive_llm_seed(Family.BAD_DEPLOYMENT, 900_000, 1),
        "num_ctx": 8192,
        "think": False,
        "keep_alive": "60m",
        "machine": "test-mac",
        "chip": "Apple M3",
    }


def test_usage_records_tokens_cache_and_time_per_call() -> None:
    fake = FakeOllama(
        [
            reply_tools(tool_call("get_service_metrics", svc()), cached=0),
            reply_tools(tool_call("observe_only", svc()), cached=None),
        ]
    )
    record = run(fake)
    usage = record.trace.usage
    assert usage is not None
    calls = usage["calls"]
    assert len(calls) == 2
    assert calls[0]["prompt_eval_count"] == 120
    assert calls[0]["prompt_eval_cached_count"] == 0
    assert calls[1]["prompt_eval_cached_count"] is None  # 서버가 주지 않으면 None
    assert calls[0]["eval_count"] == 15
    assert calls[0]["total_duration_ns"] == 2_000_000_000
    assert all(c["wall_s"] >= 0 for c in calls)
    assert usage["llm_calls"] == 2
