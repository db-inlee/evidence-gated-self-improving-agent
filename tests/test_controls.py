"""통제 후보 스킬(Ticket 003 §3): 정답 단서 금지 정규식 검사와 형식.

하위집합은 이름 앞부분으로 나눈다: pos(positive), neg(negative), mixed, neutral.
"""

import re
from pathlib import Path

import pytest

from evidencegate.skills import load_skill

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLS_DIR = REPO_ROOT / "skills" / "controls"
S0_PATH = REPO_ROOT / "skills" / "S0" / "SKILL.md"

CONTROLS = (
    "pos-traffic-check",
    "pos-timing-only",
    "neg-rollback-any",
    "neg-skip-metrics",
    "neg-timing-shortcut",
    "mixed-scale-on-traffic",
    "neutral-rephrase",
)

# family 이름과 L2 하위 유형 이름(밑줄·공백·하이픈 표기 모두)
FORBIDDEN_NAMES = (
    "bad_deployment",
    "dependency_failure",
    "traffic_spike",
    "cpu_saturated",
    "extra_deployment",
    "deploy_in_window",
    "traffic_elevated",
)
FORBIDDEN_PATTERN = re.compile(
    "|".join(name.replace("_", r"[ _-]") for name in FORBIDDEN_NAMES), re.IGNORECASE
)
STEP_NUMBER = re.compile(r"^\d+\. ", re.MULTILINE)


def control_text(name: str) -> str:
    return (CONTROLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


def test_exactly_the_ticket_controls_exist() -> None:
    assert sorted(p.name for p in CONTROLS_DIR.iterdir() if p.is_dir()) == sorted(CONTROLS)


@pytest.mark.parametrize("name", CONTROLS)
def test_control_has_no_numeric_thresholds(name: str) -> None:
    """단계 번호를 빼면 숫자가 없다(임계값·배수·퍼센트·분 단서 금지)."""
    body = STEP_NUMBER.sub("", control_text(name))
    assert not re.search(r"\d", body), body
    assert "%" not in body


@pytest.mark.parametrize("name", CONTROLS)
def test_control_has_no_family_or_subtype_names(name: str) -> None:
    match = FORBIDDEN_PATTERN.search(control_text(name))
    assert match is None, match


@pytest.mark.parametrize("name", CONTROLS)
def test_control_is_english_skill_derived_from_s0(name: str) -> None:
    text = control_text(name)
    assert text.isascii()
    assert text.startswith("# Incident Response Skill\n\n1. ")
    assert text != S0_PATH.read_text(encoding="utf-8")


def test_controls_have_distinct_skill_ids() -> None:
    ids = {load_skill(CONTROLS_DIR / name / "SKILL.md").skill_id for name in CONTROLS}
    ids.add(load_skill(S0_PATH).skill_id)
    assert len(ids) == len(CONTROLS) + 1


def test_forbidden_pattern_catches_spelling_variants() -> None:
    """검사 자체의 확인: 공백·하이픈·대소문자 변형도 잡는다."""
    for text in ("Traffic Spike", "dependency-failure", "deploy in window", "CPU_SATURATED"):
        assert FORBIDDEN_PATTERN.search(text)
    assert not FORBIDDEN_PATTERN.search("If traffic is far above its baseline and CPU is saturated")


def test_control_names_carry_their_subset() -> None:
    assert {name.split("-")[0] for name in CONTROLS} == {"pos", "neg", "mixed", "neutral"}
