"""텍스트 턴 진단 기록 (2026-11-03 amendment) 과 P5 표적 칸 규칙.

- 진단 필드를 뺀 기록과 LLM 요청 본문은 수정 전(golden)과 같아야 한다: 프롬프트·판정·흐름 불변.
- 텍스트 턴마다 assistant 내용(최대 500자)과 thinking 길이를 남긴다.
"""

import json
from pathlib import Path

import golden_scenario
import pytest

from alfworld_pilot.agent import DIAGNOSTIC_FIELDS, TEXT_CONTENT_LIMIT
from alfworld_pilot.p1 import SIX_TYPES
from alfworld_pilot.p4 import choose_target_cell

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "p2_golden.json").read_text())


@pytest.mark.parametrize("name", sorted(golden_scenario.SCENARIOS))
def test_record_without_diagnostics_and_prompts_match_the_pre_change_golden(name: str) -> None:
    now = golden_scenario.snapshot(name)
    record = {k: v for k, v in now["record"].items() if k not in DIAGNOSTIC_FIELDS}
    assert record == GOLDEN[name]["record"]
    assert now["bodies"] == GOLDEN[name]["bodies"]


def test_golden_covers_each_stop_rule() -> None:
    assert {GOLDEN[n]["record"]["stop_reason"] for n in GOLDEN} == {
        "won",
        "no_tool_call",
        "format_error",
    }


def test_text_turns_record_content_and_thinking_length() -> None:
    trace, _ = golden_scenario.run()
    assert DIAGNOSTIC_FIELDS == ("text_turn_diagnostics",)
    assert TEXT_CONTENT_LIMIT == 500
    diag = list(trace.as_dict()["text_turn_diagnostics"])
    assert diag == [
        {"llm_call": 0, "content": "Let me think.", "content_chars": 13, "thinking_chars": 30},
        {
            "llm_call": 3,
            "content": golden_scenario.LONG_TEXT[:500],
            "content_chars": 900,
            "thinking_chars": 0,
        },
    ]
    assert len(diag) == trace.text_turns


def test_diagnostics_default_to_empty_for_old_style_traces() -> None:
    from alfworld_pilot.agent import EpisodeTrace

    trace = EpisodeTrace(
        steps=(),
        won=False,
        stop_reason="max_steps",
        first_observation="",
        format_errors=0,
        text_turns=0,
        llm_calls=0,
        prompt_tokens=(),
        completion_tokens=(),
        truncated_calls=0,
        max_omitted=0,
        wall_s=0.0,
    )
    assert trace.text_turn_diagnostics == ()


def test_target_cell_is_the_highest_mean_success_with_six_types_order_on_ties() -> None:
    rates = {t: 0.1 for t in SIX_TYPES}
    rates["pick_cool_then_place_in_recep"] = 0.5
    assert choose_target_cell(rates) == "pick_cool_then_place_in_recep"
    tie = dict.fromkeys(SIX_TYPES, 0.3)
    assert choose_target_cell(tie) == SIX_TYPES[0]
    tie["pick_two_obj_and_place"] = 0.4
    tie["look_at_obj_in_light"] = 0.4
    assert choose_target_cell(tie) == "look_at_obj_in_light"
    with pytest.raises(ValueError):
        choose_target_cell({SIX_TYPES[0]: 0.2})  # 여섯 칸이 모두 있어야 한다
