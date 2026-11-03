"""P4 요약 (Ticket 008 §4 P4). 가짜 기록으로 시험한다.

G4·G3(d_c)는 P4-a 기준 "미리 보기"로만 낸다(2026-11-03 결정 2). 최종 판정은 P6.
"""

from typing import Any

import pytest

from alfworld_pilot.p1 import SIX_TYPES
from alfworld_pilot.p4 import (
    cell_summary,
    disagreement_rate,
    g3_preview,
    g4_preview,
    summarize_p4,
    text_turn_summary,
)

SEEN = "You arrive at shelf 1. On the shelf 1, you see a pen 1."
TAKE = "You pick up the pen 1 from the shelf 1."


def rec(game: str, task: str, repeat: int, won: bool, **kw: Any) -> dict[str, Any]:
    seen = kw.pop("seen", True)
    steps = [
        {"command": "go to shelf 1", "observation": SEEN if seen else "x", "admissible": True},
        {"command": "take pen 1 from shelf 1", "observation": TAKE, "admissible": True},
    ]
    r = {
        "key": f"train/{game}/t#{repeat}",
        "rel": f"train/{game}/t",
        "task_type": task,
        "repeat": repeat,
        "won": won,
        "stop_reason": "won" if won else kw.pop("stop", "max_steps"),
        "steps": steps,
        "first_observation": "room",
        "format_errors": 0,
        "text_turns": len(kw.get("diag", [])),
        "llm_calls": 3,
        "prompt_tokens": [100, 200, 300],
        "completion_tokens": [5, 5, 5],
        "truncated_calls": 0,
        "max_omitted": 0,
        "wall_s": 10.0,
        "started_at": 0.0,
        "finished_at": 10.0,
        "run": {"concurrency": 2, "server_num_parallel": 2},
        "text_turn_diagnostics": kw.pop("diag", []),
    }
    r.update(kw)
    return r


def test_disagreement_is_the_mean_over_the_three_repeat_pairs() -> None:
    # 게임 3개: g0 (1,1,1) / g1 (1,0,0) / g2 (0,0,1)
    outcomes = {"g0": [True, True, True], "g1": [True, False, False], "g2": [False, False, True]}
    # 쌍 (0,1): g1 다름 -> 1/3, (0,2): g1, g2 -> 2/3, (1,2): g2 -> 1/3. 평균 4/9
    assert disagreement_rate(outcomes) == pytest.approx(4 / 9)
    with pytest.raises(ValueError):
        disagreement_rate({"g0": [True, False]})  # 반복 3회가 아니면 거부


def test_cell_summary_rates_ci_and_repeat_means() -> None:
    records = [
        rec("a", "t", 0, True),
        rec("a", "t", 1, True),
        rec("a", "t", 2, False),
        rec("b", "t", 0, False),
        rec("b", "t", 1, False),
        rec("b", "t", 2, False),
    ]
    c = cell_summary(records)
    assert c["games"] == 2 and c["episodes"] == 6 and c["won"] == 2
    assert c["success_rate"] == pytest.approx(2 / 6)
    assert c["success_rate_by_repeat"] == [0.5, 0.5, 0.0]
    low, high = c["clopper_pearson_95"]
    assert 0 < low < 2 / 6 < high < 1
    # a (T, T, F), b (F, F, F): 쌍 (0,1) 0/2, (0,2) 1/2, (1,2) 1/2 -> 평균 1/3
    assert c["d_c"] == pytest.approx(1 / 3)
    assert c["stop_reasons"] == {"max_steps": 4, "won": 2}


def test_g4_preview_counts_cells_inside_the_range() -> None:
    rates = dict(zip(SIX_TYPES, [0.1, 0.15, 0.5, 0.85, 0.9, 0.3], strict=True))
    g = g4_preview(rates)
    assert g["inside"] == 4 and g["needed"] == 4 and g["passes_preview"] is True
    rates[SIX_TYPES[5]] = 0.0
    assert g4_preview(rates)["passes_preview"] is False


def test_g3_preview_uses_the_fixed_formula_and_floor() -> None:
    d = dict.fromkeys(SIX_TYPES, 0.0)  # 하한 0.10 적용
    available = dict.fromkeys(SIX_TYPES, 278)
    g = g3_preview(d, available)
    # n_O = ceil(1.96^2 * 0.1 / 0.1^2) = 39, n_B = ceil(39 * 35/50) = 28, n_A = 10
    assert g["n_O"][SIX_TYPES[0]] == 39 and g["n_B"][SIX_TYPES[0]] == 28
    assert g["n_min"] == 77 and g["passes_preview"] is True
    d[SIX_TYPES[2]] = 0.5  # n_O = ceil(3.8416 * 0.5 / 0.01) = 193 -> n_min = 10 + 136 + 193
    g = g3_preview(d, available)
    assert g["n_min"] == 339 and g["passes_preview"] is False


def test_text_turn_summary_by_type() -> None:
    d1 = {"llm_call": 0, "content": "Task completed.", "content_chars": 15, "thinking_chars": 40}
    d2 = {"llm_call": 3, "content": "I need to look.", "content_chars": 15, "thinking_chars": 0}
    records = [rec("a", "t", 0, False, diag=[d1, d1]), rec("b", "t", 0, False, diag=[d2])]
    s = text_turn_summary(records)
    assert s["t"]["text_turns"] == 3
    assert s["t"]["empty_content"] == 0
    assert s["t"]["thinking_chars"]["max"] == 40
    assert s["t"]["top_openings"][0] == ["Task completed.", 2]


def test_summarize_p4_picks_the_target_cell_and_marks_expert_failures() -> None:
    records = []
    for i, t in enumerate(SIX_TYPES):
        for g in range(2):
            for r in range(3):
                won = (i == 3) or (i == 1 and g == 0)
                records.append(rec(f"{t}-{g}", t, r, won, seen=not (i == 3 and g == 1)))
    s = summarize_p4(
        records,
        load_s=[13.0],
        expert_steps={f"train/{SIX_TYPES[0]}-0/t": 5},
        expert_failed={f"train/{SIX_TYPES[0]}-1/t"},
        available_train={t: 278 for t in SIX_TYPES},
    )
    assert s["target_cell"] == SIX_TYPES[3]
    assert s["cells"][SIX_TYPES[3]]["success_rate"] == 1.0
    assert s["expert_failed_games"] == [
        {"rel": f"train/{SIX_TYPES[0]}-1/t", "won": [False, False, False]}
    ]
    assert s["preview_only"] is True
    # 성공: 칸 3 전부(6) + 칸 1 게임 0(3) = 9. 칸 3 게임 1(3)은 내용을 보지 않고 꺼냄 -> E-a 불성립
    assert s["e_a"] == {
        "episodes": 36,
        "won": 9,
        "holds": 6,
        "base_rate": 6 / 36,
        "share_of_wins_failing": 3 / 9,
    }
    assert s["throughput"]["load_s"] == 13.0
