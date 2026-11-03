"""P6 판정 계산 (Ticket 008 §3, 고정 파일 da5ac108). 가짜 결과로 시험한다."""

from fractions import Fraction
from typing import Any

import pytest

from alfworld_pilot.p1 import SIX_TYPES
from alfworld_pilot.p6 import (
    available_after_folder_exclusion,
    diag_compare,
    g5_estimate,
    g6_cell,
    g6_judgement,
    game_level_cp,
    majority,
    stop_breakdown,
)


def test_majority_of_three() -> None:
    assert majority([True, False, True]) is True
    assert majority([False, False, True]) is False
    with pytest.raises(ValueError):
        majority([True, False])


def test_g6_cell_with_identical_repeats_has_zero_noise() -> None:
    s0 = {f"g{i}": [True, True, True] for i in range(4)}
    probe = {f"g{i}": False for i in range(4)}
    c = g6_cell(probe, s0)
    assert c["delta"] == Fraction(-1)
    assert c["q_lo"] == c["q_hi"] == Fraction(0)
    assert c["games"] == 4


def test_g6_cell_noise_comes_from_single_repeat_minus_majority() -> None:
    s0 = {"a": [True, False, False], "b": [True, True, True], "c": [False, False, False]}
    probe = {"a": False, "b": True, "c": False}
    c = g6_cell(probe, s0)
    assert c["delta"] == 0  # 다수결 (F, T, F)와 같다
    assert c["q_lo"] <= 0 <= c["q_hi"]
    assert c["q_hi"] > 0  # 반복 0을 고르면 a가 +1
    assert g6_cell(probe, s0) == c  # seed 0 고정


def test_g6_judgement_needs_a_target_drop_and_non_targets_inside_their_band() -> None:
    cells = {
        t: {"delta": Fraction(0), "q_lo": Fraction(-1, 10), "q_hi": Fraction(1, 10)}
        for t in SIX_TYPES
    }
    cells[SIX_TYPES[0]] = {
        "delta": Fraction(-1, 2),
        "q_lo": Fraction(-1, 10),
        "q_hi": Fraction(1, 10),
    }
    j = g6_judgement(cells, target=SIX_TYPES[0])
    assert j["target_drop"] and j["non_targets_inside"] and j["passes"]
    cells[SIX_TYPES[3]]["delta"] = Fraction(1, 5)
    j = g6_judgement(cells, target=SIX_TYPES[0])
    assert not j["non_targets_inside"] and not j["passes"]
    assert j["non_target_outside"] == [SIX_TYPES[3]]


def test_g5_estimate_uses_cell_sizes_candidates_and_s0_repeats() -> None:
    sizes = {t: {"n_A": 10, "n_B": 28, "n_O": 39} for t in SIX_TYPES}  # 칸당 77
    e = g5_estimate(sizes, s_bar=100.0, candidates=40, s0_repeats=3, budget_days=7)
    assert e["per_candidate"] == 6 * 77
    assert e["E_009"] == 40 * 462 + 3 * 462
    assert e["T_009_s"] == pytest.approx(e["E_009"] * 100.0 * 1.1)
    assert e["passes"] is (e["T_009_s"] <= 7 * 86400)


def test_game_level_cp_uses_the_majority_per_game() -> None:
    games = {"a": [True, True, False], "b": [False, False, True], "c": [True, True, True]}
    g = game_level_cp(games)
    assert g["games"] == 3 and g["won_majority"] == 2
    low, high = g["clopper_pearson_95"]
    assert low < 2 / 3 < high


def rec(task: str, stop: str, *, won: bool = False, fmt: int = 0) -> dict[str, Any]:
    return {"task_type": task, "stop_reason": stop, "won": won, "format_errors": fmt}


def test_stop_breakdown_per_cell() -> None:
    records = [
        rec("t", "won", won=True),
        rec("t", "max_steps", fmt=2),
        rec("t", "no_tool_call"),
        rec("t", "format_error", fmt=3),
        rec("t", "max_llm_calls"),
    ]
    b = stop_breakdown(records)
    assert b["t"] == {
        "episodes": 5,
        "won": 1,
        "max_steps": 1,
        "no_tool_call": 1,
        "format_error": 1,
        "max_llm_calls": 1,
        "episodes_with_format_errors": 2,
    }


def drec(key: str, task: str, won: bool, stop: str, steps: int, text: int) -> dict[str, Any]:
    return {
        "key": key,
        "task_type": task,
        "won": won,
        "stop_reason": stop,
        "steps": [{}] * steps,
        "text_turns": text,
    }


def test_diag_compare_pairs_games_and_counts_stop_reason_moves() -> None:
    base = [
        drec("g1#0", "t", False, "no_tool_call", 10, 2),
        drec("g2#0", "t", True, "won", 8, 0),
        drec("g3#0", "u", False, "max_steps", 50, 1),
    ]
    var = [
        drec("g1#0", "t", True, "won", 20, 6),
        drec("g2#0", "t", True, "won", 9, 0),
        drec("g3#0", "u", False, "max_steps", 50, 0),
    ]
    c = diag_compare(base, var)
    assert c["paired_games"] == 3
    assert c["cells"]["t"]["won"] == {"baseline": 1, "variant": 2}
    assert c["moves"]["no_tool_call -> won"] == 1
    assert c["moves"]["won -> won"] == 1
    assert c["cells"]["t"]["mean_steps"] == {"baseline": 9.0, "variant": 14.5}
    assert c["cells"]["t"]["text_turns"] == {"baseline": 2, "variant": 6}
    assert c["label"] == "exploratory diagnostic, not used for G1-G6"
    with pytest.raises(ValueError):
        diag_compare(base, var[:2])  # 짝이 맞지 않으면 거부


def test_folder_exclusion_removes_every_trial_of_a_pool_folder() -> None:
    from alfworld_pilot.p1 import GameInfo

    X, Y = SIX_TYPES[0], SIX_TYPES[1]  # playable은 여섯 유형만 인정한다

    def g(rel: str, t: str, playable: bool = True) -> GameInfo:
        return GameInfo(rel, t, t, playable, True, "a" * 64, "b" * 64)

    train = [
        g("train/A-1/t1", X),
        g("train/A-1/t2", X),
        g("train/B-2/t1", X),
        g("train/C-3/t1", Y),
        g("train/C-3/t2", Y, playable=False),
    ]
    pool = ["train/A-1/t1", "train/C-3/t1"]
    assert available_after_folder_exclusion(train, pool) == {**dict.fromkeys(SIX_TYPES, 0), X: 1}


def test_overall_verdict_follows_the_fixed_rules() -> None:
    from alfworld_pilot.p6 import overall_verdict

    ok = dict.fromkeys(("G1", "G2", "G3", "G4", "G5", "G6"), "pass")
    assert overall_verdict(ok, reduced_design_fits=False)["verdict"] == "Go"
    only_g5 = {**ok, "G5": "fail"}
    assert overall_verdict(only_g5, reduced_design_fits=True)["verdict"] == "Conditional Go"
    assert overall_verdict(only_g5, reduced_design_fits=False)["verdict"] == "No-Go"
    g3_g5 = {**ok, "G3": "fail", "G5": "fail"}
    assert overall_verdict(g3_g5, reduced_design_fits=True)["verdict"] == "Conditional Go"
    conda = {**ok, "G1": "conditional"}
    assert overall_verdict(conda, reduced_design_fits=False)["verdict"] == "Conditional Go"
    g4 = {**ok, "G4": "fail", "G5": "fail"}
    v = overall_verdict(g4, reduced_design_fits=True)
    assert v["verdict"] == "No-Go" and v["failed"] == ["G4", "G5"]


def test_reduced_designs_scale_candidates_cells_and_repeats() -> None:
    from alfworld_pilot.p6 import reduced_designs

    sizes = {t: {"n_A": 10, "n_B": 28, "n_O": 39} for t in SIX_TYPES}
    rows = reduced_designs(sizes, s_bar=100.0, budget_days=7)
    assert {"candidates", "cell_scale", "s0_repeats", "E_009", "T_009_days", "fits"} <= set(rows[0])
    full = next(
        r for r in rows if r["candidates"] == 40 and r["cell_scale"] == 1.0 and r["s0_repeats"] == 3
    )
    assert full["E_009"] == 43 * 462
    half = next(
        r for r in rows if r["candidates"] == 40 and r["cell_scale"] == 0.5 and r["s0_repeats"] == 3
    )
    assert half["E_009"] < full["E_009"]


def test_a_gate_that_did_not_run_counts_as_not_passed() -> None:
    from alfworld_pilot.p6 import overall_verdict

    ok = dict.fromkeys(("G1", "G2", "G3", "G4", "G5"), "pass")
    v = overall_verdict({**ok, "G6": "not_run"}, reduced_design_fits=True)
    assert v["verdict"] == "No-Go" and v["failed"] == ["G6"]


def test_folder_exclusion_counts_only_the_six_types() -> None:
    from alfworld_pilot.p1 import GameInfo

    movable = GameInfo(
        "train/M-1/t1", "pick_and_place_with_movable_recep", "p", False, False, None, "b" * 64
    )
    out = available_after_folder_exclusion([movable], [])
    assert set(out) == set(SIX_TYPES) and sum(out.values()) == 0


def test_a_target_delta_inside_the_band_is_not_a_drop() -> None:
    band = {"q_lo": Fraction(-1, 10), "q_hi": Fraction(1, 10)}
    cells = {t: {"delta": Fraction(0), **band} for t in SIX_TYPES}
    j = g6_judgement(cells, target=SIX_TYPES[0])
    assert not j["target_drop"] and not j["passes"]
    cells[SIX_TYPES[0]] = {"delta": Fraction(-1, 10), **band}  # 경계는 하락이 아니다(Δ < q_lo)
    assert not g6_judgement(cells, target=SIX_TYPES[0])["target_drop"]


def test_noise_draws_a_random_repeat_not_always_the_first() -> None:
    # 반복 0은 언제나 다수결과 같고 반복 1만 다르다: 반복 0만 쓰면 잡음이 0이 된다
    s0 = {f"g{i}": [False, True, False] for i in range(6)}
    c = g6_cell({g: False for g in s0}, s0)
    assert c["q_hi"] > 0
