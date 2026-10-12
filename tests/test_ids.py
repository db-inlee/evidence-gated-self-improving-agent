"""실행 키에 들어가는 내용 ID: tool_spec_id, config_id."""

import hashlib
import json
from dataclasses import asdict, replace

from evidencegate.ids import config_id, content_id, tool_spec_id
from evidencegate.policy import ScriptedPolicy
from opsworld import agent_env


def test_content_id_is_sha256_prefix_of_sorted_key_json() -> None:
    obj = {"b": 1, "a": [1, {"d": 2, "c": 3}]}
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert content_id(obj) == hashlib.sha256(text.encode()).hexdigest()[:12]


def test_content_id_ignores_key_order() -> None:
    assert content_id({"a": 1, "b": 2}) == content_id({"b": 2, "a": 1})


def test_tool_spec_id_hashes_full_specs() -> None:
    specs = agent_env.TOOL_SPECS
    assert tool_spec_id(specs) == content_id([asdict(s) for s in specs])


def test_tool_spec_id_changes_when_any_description_changes() -> None:
    specs = list(agent_env.TOOL_SPECS)
    changed = [replace(specs[0], description=specs[0].description + " "), *specs[1:]]
    assert tool_spec_id(changed) != tool_spec_id(specs)


def test_tool_spec_id_changes_when_parameters_change() -> None:
    specs = list(agent_env.TOOL_SPECS)
    parameters = {**specs[-1].parameters, "additionalProperties": True}
    changed = [*specs[:-1], replace(specs[-1], parameters=parameters)]
    assert tool_spec_id(changed) != tool_spec_id(specs)


def test_scripted_config_is_the_call_list() -> None:
    policy = ScriptedPolicy(["get_service_metrics", ("observe_only", {"service_id": "x"})], "A")
    assert policy.config == {
        "type": "scripted",
        "calls": [
            {"tool": "get_service_metrics", "args": None},
            {"tool": "observe_only", "args": {"service_id": "x"}},
        ],
    }
    assert config_id(policy) == content_id(policy.config)


def test_config_id_changes_with_calls_but_not_with_name() -> None:
    a = ScriptedPolicy(["get_service_metrics", "observe_only"], "A")
    b = ScriptedPolicy(["get_traffic_metrics", "observe_only"], "A")
    renamed = ScriptedPolicy(["get_service_metrics", "observe_only"], "other")
    assert config_id(a) != config_id(b)
    assert config_id(a) == config_id(renamed)
