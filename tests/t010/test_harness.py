"""Ticket 010 §1·§8: 하네스 설정(도구 순서, 도구 설명, 표시 방식)과 정책 연결.

기본 하네스는 지금 코드와 바이트까지 같아야 한다(원칙 2). 기준 요청 본문
`data/default_harness_requests.json`은 하네스 도입 전 코드(커밋 adb9939)가 보낸 /api/chat 요청을
그대로 저장한 것이다. 가짜 전송만 쓴다(Ollama 호출 없음).
"""

import copy
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.harness import (
    OBSERVATION_TOOLS,
    PROPOSER_RENDERERS,
    RENDERERS,
    HarnessText,
    apply_tool_specs,
    flatten,
    render,
)
from evidencegate.ids import content_id, tool_spec_id
from evidencegate.machine import Machine
from evidencegate.ollama_policy import OllamaPolicy, OllamaSettings
from evidencegate.runner import EpisodeRecord, run_episode
from evidencegate.skills import load_skill
from opsworld.agent_env import TOOL_SPECS, AgentEnv
from opsworld.generator import generate_incident
from opsworld.state import Family, Level
from opsworld.tools import Episode

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).parent / "data" / "default_harness_requests.json").read_text())
DIGEST = "f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe"
S0 = load_skill(ROOT / "skills" / "S0" / "SKILL.md")
SCRIPT = [
    "get_service_metrics",
    "get_recent_deployments",
    "get_dependency_health",
    "get_traffic_metrics",
    "observe_only",
]
# tests/test_preflight.py의 REFERENCE(S0 Dev-A 기록의 정책 설정)와 같은 값
REFERENCE: dict[str, Any] = {
    "type": "ollama",
    "model": "gpt-oss:20b",
    "model_digest": DIGEST,
    "quantization": "MXFP4",
    "ollama_version": "0.40.3",
    "instructions_sha256": "332b776ce67826fba8829790fc46090df243c2f9f38602e1c8c72faea4bfaed3",
    "options": {"temperature": 0.7, "num_ctx": 8192},
    "think": "low",
    "stop": {"max_tool_calls": 8, "max_consecutive_text_turns": 2},
    "format_error_policy": {"feedback": "error_as_tool_result", "max_consecutive_format_errors": 3},
    "seed_rule": "derive_llm_seed(family, incident_seed, repeat): sha256, first 8 bytes mod 2^31",
    "machine": {"hw_model": "Mac15,3", "chip": "Apple M3"},
    "parallel": {"client_concurrency": 2, "server_num_parallel": 2},
}


class Capture:
    """정해진 도구 호출 순서로 답하고 /api/chat 요청 본문 바이트를 모은다."""

    def __init__(self) -> None:
        self.bodies: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.3"})
        if path == "/api/tags":
            entry = {
                "name": "gpt-oss:20b",
                "model": "gpt-oss:20b",
                "digest": DIGEST,
                "details": {"quantization_level": "MXFP4"},
            }
            return httpx.Response(200, json={"models": [entry]})
        self.bodies.append(request.content.decode("utf-8"))
        body = json.loads(request.content)
        service = body["messages"][1]["content"].split("service ")[1].split(" ")[0]
        name = SCRIPT[len(self.bodies) - 1]
        call = {"function": {"name": name, "arguments": {"service_id": service}}}
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return httpx.Response(
            200,
            json={"model": "gpt-oss:20b", "message": message, "done": True,
                  "prompt_eval_count": 1, "eval_count": 1},
        )  # fmt: skip


def policy(capture: Capture, harness: HarnessText | None = None) -> OllamaPolicy:
    client = httpx.Client(transport=httpx.MockTransport(capture.handler), base_url="http://o.test")
    settings = OllamaSettings.for_model(
        "gpt-oss:20b", concurrency=2, server_num_parallel=2, harness=harness
    )
    return OllamaPolicy.connect(settings, client, machine=Machine("Mac15,3", "Apple M3"))


def episode(case: dict[str, Any], harness: HarnessText | None) -> tuple[Capture, EpisodeRecord]:
    capture = Capture()
    record = run_episode(
        policy(capture, harness),
        S0,
        Family(case["family"]),
        case["incident_seed"],
        0,
        Level(case["level"]),
    )
    return capture, record


# 원칙 2: 기본 하네스 동일성 --------------------------------------------------------------


@pytest.mark.parametrize("harness", [None, HarnessText()])
def test_default_harness_keeps_the_config_and_its_id(harness: HarnessText | None) -> None:
    config = policy(Capture(), harness).config
    assert config == REFERENCE
    assert content_id(config) == "7c66ee5bf13c"
    assert tool_spec_id(TOOL_SPECS) == "e9b4996f7529"


@pytest.mark.parametrize("harness", [None, HarnessText()])
def test_default_harness_sends_the_same_request_bytes(harness: HarnessText | None) -> None:
    for case in GOLDEN["cases"]:
        capture, _ = episode(case, harness)
        assert capture.bodies == case["requests"]


def test_overrides_equal_to_the_defaults_are_the_default() -> None:
    same = HarnessText(
        tool_order=OBSERVATION_TOOLS,
        tool_descriptions=(("get_traffic_metrics", TOOL_SPECS[3].description),),
        observation_renderer="json_full",
    )
    assert same.config() is None
    assert policy(Capture(), same).config == REFERENCE


# 비기본 필드 -------------------------------------------------------------------------


ORDER = (
    "get_traffic_metrics",
    "get_service_metrics",
    "get_recent_deployments",
    "get_dependency_health",
)
ACTIONS = ("rollback_deployment", "scale_service", "observe_only")
TRAFFIC_TEXT = (
    "Get baseline and current request rate (requests per second) of the service, "
    "to see whether demand changed."
)


@pytest.mark.parametrize(
    ("harness", "expected"),
    [
        (HarnessText(tool_order=ORDER), {"tool_order": [*ORDER, *ACTIONS]}),
        (
            HarnessText(tool_descriptions=(("get_traffic_metrics", TRAFFIC_TEXT),)),
            {"tool_descriptions": {"get_traffic_metrics": TRAFFIC_TEXT}},
        ),
        (
            HarnessText(observation_renderer="key_value_lines"),
            {"observation_renderer": "key_value_lines"},
        ),
    ],
)  # fmt: skip
def test_each_non_default_field_changes_the_config_id(
    harness: HarnessText, expected: dict[str, Any]
) -> None:
    config = policy(Capture(), harness).config
    assert content_id(config) != "7c66ee5bf13c"
    assert {k: v for k, v in config.items() if k != "harness"} == REFERENCE
    assert config["harness"] == {
        "tool_order": list(TOOL_SPECS_NAMES),
        "tool_descriptions": {},
        "observation_renderer": "json_full",
        **expected,
    }


TOOL_SPECS_NAMES = tuple(s.name for s in TOOL_SPECS)


def test_tool_order_and_descriptions_apply_to_the_tool_definitions_only() -> None:
    harness = HarnessText(
        tool_order=ORDER, tool_descriptions=(("get_traffic_metrics", TRAFFIC_TEXT),)
    )
    capture, record = episode(GOLDEN["cases"][0], harness)
    tools = json.loads(capture.bodies[0])["tools"]
    assert [t["function"]["name"] for t in tools] == [
        *ORDER,
        "rollback_deployment",
        "scale_service",
        "observe_only",
    ]
    traffic = next(t for t in tools if t["function"]["name"] == "get_traffic_metrics")
    assert traffic["function"]["description"] == TRAFFIC_TEXT
    default_spec = next(s for s in TOOL_SPECS if s.name == "get_traffic_metrics")
    assert traffic["function"]["parameters"] == default_spec.parameters
    assert [s.description for s in TOOL_SPECS][3] != TRAFFIC_TEXT  # 전역 명세는 그대로
    _, base = episode(GOLDEN["cases"][0], None)
    assert record.trace.turns == base.trace.turns  # 원장(원시 결과)은 그대로
    assert (record.correct_disposition, record.evidence_safe) == (
        base.correct_disposition,
        base.evidence_safe,
    )


@pytest.mark.parametrize("renderer", ["json_compact", "key_value_lines", "json_unsorted"])
def test_the_renderer_applies_to_tool_messages_only(renderer: str) -> None:
    capture, record = episode(GOLDEN["cases"][2], HarnessText(observation_renderer=renderer))
    last = json.loads(capture.bodies[-1])["messages"]
    tool_messages = [m for m in last if m["role"] == "tool"]
    _, base = episode(GOLDEN["cases"][2], None)
    case = GOLDEN["cases"][2]
    env = AgentEnv(
        Episode(
            generate_incident(Family(case["family"]), case["incident_seed"], Level(case["level"]))
        )
    )
    raw = [env.call(name, {"service_id": env.service_id}) for name in SCRIPT[:4]]  # 키 순서 그대로
    assert [m["content"] for m in tool_messages] == [render(r, renderer) for r in raw]
    assert record.trace.turns == base.trace.turns
    assert (
        json.loads(capture.bodies[0])["tools"]
        == json.loads(GOLDEN["cases"][2]["requests"][0])["tools"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"tool_order": ("get_service_metrics", "get_recent_deployments", "get_dependency_health")},
        {"tool_order": ("rollback_deployment", *ORDER[1:])},
        {"tool_order": ("get_service_metrics",) * 4},
        {"tool_descriptions": (("restart_service", "x"),)},
        {"tool_descriptions": (("get_traffic_metrics", ""),)},
        {"tool_descriptions": (("get_traffic_metrics", "a"), ("get_traffic_metrics", "b"))},
        {"observation_renderer": "changed_fields_only"},
    ],
)  # fmt: skip
def test_invalid_harness_values_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HarnessText(**bad)


# 표시 방식(원칙 5) -----------------------------------------------------------------------


def test_the_proposer_renderers_are_the_four_in_the_ticket() -> None:
    assert PROPOSER_RENDERERS == ("json_full", "json_compact", "key_value_lines", "json_unsorted")
    assert set(PROPOSER_RENDERERS) <= set(RENDERERS)


def raw_results() -> list[dict[str, Any]]:
    out = []
    for family in Family:
        for level in (Level.L1, Level.L2, Level.L3):
            for seed in (500_000, 500_001, 900_005):
                for spec in TOOL_SPECS:
                    env = AgentEnv(Episode(generate_incident(family, seed, level)))
                    out.append(env.call(spec.name, {"service_id": env.service_id}))
                env = AgentEnv(Episode(generate_incident(family, seed, level)))
                out.append(env.call("get_service_metrics", {"service_id": "nope"}))  # 오류 결과
    return out


def parse_lines(text: str) -> dict[str, Any]:
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(": ")
        out[key] = json.loads(value)
    return out


HIDDEN = ("root_cause", "has_bug", "l2_subtype", "incident_id")


@pytest.mark.parametrize("renderer", PROPOSER_RENDERERS)
def test_renderers_add_no_information(renderer: str) -> None:
    for result in raw_results():
        before = copy.deepcopy(result)
        text = render(result, renderer)
        assert result == before  # 결과를 바꾸지 않는다
        if renderer == "key_value_lines":
            assert parse_lines(text) == flatten(result)  # 같은 경로·값, 더도 덜도 없음
        else:
            assert json.loads(text) == result
        for word in HIDDEN:
            assert word not in text
        assert set(_tokens(text)) <= set(_tokens(json.dumps(result)))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)


def test_flatten_paths() -> None:
    nested = {"b": 1, "a": {"y": [1, {"z": None}], "x": "s"}, "c": [], "d": {}}
    assert flatten(nested) == {"a.x": "s", "a.y.0": 1, "a.y.1.z": None, "b": 1, "c": [], "d": {}}
    assert render(nested, "key_value_lines").splitlines()[0] == 'a.x: "s"'


def test_apply_tool_specs_keeps_names_parameters_and_kinds() -> None:
    harness = HarnessText(tool_order=ORDER, tool_descriptions=(("observe_only", "Keep watching."),))
    specs = apply_tool_specs(TOOL_SPECS, harness)
    assert sorted(s.name for s in specs) == sorted(TOOL_SPECS_NAMES)
    for s in specs:
        original = next(o for o in TOOL_SPECS if o.name == s.name)
        assert (s.kind, s.parameters) == (original.kind, original.parameters)
    assert apply_tool_specs(TOOL_SPECS, None) == list(TOOL_SPECS)
