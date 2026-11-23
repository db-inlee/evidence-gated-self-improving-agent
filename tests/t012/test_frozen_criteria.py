"""Ticket 012 §9의 2: S0-prime·S0-prime-r1과 진행 기준을 실행 전에 고정한다.

고정 파일은 `plans/ticket-012-criteria.json`(+ `.sha256`)과 `skills/S0-prime*/SKILL.md`다.
이 테스트는 고정 뒤 파일이 바뀌지 않았는지, 문안 원칙(§1.1)과 순서 지시 금지 규칙(§1.4)이
기록과 맞는지 본다.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evidencegate.skills import load_skill
from opsworld.agent_env import TOOL_SPECS

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "plans" / "ticket-012-criteria.json"
S0 = ROOT / "skills" / "S0" / "SKILL.md"
S0P = ROOT / "skills" / "S0-prime" / "SKILL.md"
S0P_R1 = ROOT / "skills" / "S0-prime-r1" / "SKILL.md"
OBSERVATION = [s.name for s in TOOL_SPECS if s.kind == "observation"]


@pytest.fixture(scope="module")
def criteria() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CRITERIA.read_text(encoding="utf-8"))
    return data


def test_criteria_file_matches_its_sha256() -> None:
    recorded = CRITERIA.with_name(CRITERIA.name + ".sha256").read_text().split()
    assert recorded == [hashlib.sha256(CRITERIA.read_bytes()).hexdigest(), CRITERIA.name]


def test_skill_ids_are_the_recorded_ones(criteria: dict[str, Any]) -> None:
    skills = criteria["skills"]
    assert skills["S0_prime"] == {
        "path": "skills/S0-prime/SKILL.md",
        "skill_id": load_skill(S0P).skill_id,
    }
    assert skills["S0_prime_r1"] == {
        "path": "skills/S0-prime-r1/SKILL.md",
        "skill_id": load_skill(S0P_R1).skill_id,
    }
    assert skills["S0"]["skill_id"] == load_skill(S0).skill_id == "bf65e569ae37"


def test_s0_prime_opens_the_tool_order_and_names_no_observation_tool() -> None:
    text = S0P.read_text(encoding="utf-8")
    for name in OBSERVATION:
        assert name not in text, name
    assert not re.search(r"\d", text)  # 숫자 임계값 없음
    for family in ("bad_deployment", "dependency_failure", "traffic_spike"):
        assert family not in text
    for action in ("rollback_deployment", "scale_service", "observe_only"):
        assert action in text  # 조치 도구 세 개의 뜻은 남긴다


def test_s0_prime_r1_only_restores_the_per_disposition_evidence_of_s0() -> None:
    base = S0P.read_text(encoding="utf-8")
    r1 = S0P_R1.read_text(encoding="utf-8")
    assert r1.startswith(base)  # S0-prime 본문은 그대로, 뒤에 한 절만 더한다
    added = r1[len(base) :]
    s0_lines = S0.read_text(encoding="utf-8").splitlines()
    rollback = next(line for line in s0_lines if line.startswith("4. ")).removeprefix("4. ")
    observe = next(line for line in s0_lines if line.startswith("5. ")).removeprefix("5. ")
    assert rollback in added and observe in added
    assert "inspect resource usage and traffic" not in added  # 무엇을 확인할지는 열어 둔다
    for name in OBSERVATION:
        assert name not in r1, name
    assert not re.search(r"\d", r1)


def test_progress_criteria_are_the_decided_numbers(criteria: dict[str, Any]) -> None:
    assert criteria["base_rate"] == {
        "cells_needed": 4,
        "cells": 6,
        "per_cell": 10,
        "cd_successes_min": 2,
        "cd_successes_max": 8,
    }
    assert criteria["k_rule"] == {
        "pairs": 120,
        "repeats_added": 2,
        "same_config_and_seed": True,
        "upper_bound": "clopper_pearson_two_sided_95",
        "floor": "25/600",
        "tail_below": "1/100",
        "episodes": 60,
        "k_max": 30,
    }
    assert criteria["probe"]["p0"] == {"max_tool_calls": 1, "min_changed": 50, "episodes": 60}
    assert criteria["probe"]["m"] == 3 and criteria["probe"]["min_fields"] == 2
    assert criteria["descriptive"] == ["first_call_is_disposition"]
    assert criteria["revision"] == {
        "allowed": 1,
        "only_if": "base_rate_fails",
        "change": "restore_per_disposition_evidence",
        "skill": "S0_prime_r1",
    }


def ban_pattern(criteria: dict[str, Any]) -> re.Pattern[str]:
    words = criteria["order_ban"]["words"]
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in words) + r")\b",
        re.IGNORECASE,
    )


def test_the_order_ban_list_is_the_decided_list(criteria: dict[str, Any]) -> None:
    ban = criteria["order_ban"]
    assert ban["words"] == [
        "first", "then", "next", "before", "after", "last", "start with", "begin with",
        "always call", "must call", "required", "prior to", "followed by",
        "afterward", "afterwards", "beforehand", "firstly", "subsequently", "initially",
    ]  # fmt: skip
    assert any("next-door" in limit for limit in ban["known_limits"])
    assert ban["match"] == "word_boundary_case_insensitive"
    assert ban["other_tool_names_banned"] is True
    assert ban["judgement"] == "automatic_only"
    assert len(ban["examples"]["reject"]) >= 5 and len(ban["examples"]["allow"]) >= 5


def test_the_examples_follow_the_ban_rule(criteria: dict[str, Any]) -> None:
    pattern = ban_pattern(criteria)
    for text in criteria["order_ban"]["examples"]["reject"]:
        assert pattern.search(text), text
    for text in criteria["order_ban"]["examples"]["allow"]:
        assert not pattern.search(text), text


def test_the_current_tool_descriptions_pass_the_ban(criteria: dict[str, Any]) -> None:
    pattern = ban_pattern(criteria)
    for spec in TOOL_SPECS:
        assert not pattern.search(spec.description), spec.name
