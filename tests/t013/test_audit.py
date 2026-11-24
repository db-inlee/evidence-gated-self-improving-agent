"""Ticket 013 1단계 감사 요약(`evidencegate.t013.audit`). 가짜 기록만 쓴다(runs/ 없음)."""

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.t013 import audit as au

REPO_ROOT = Path(__file__).resolve().parents[2]
BAD1, BAD2 = ("bad_deployment", "L1"), ("bad_deployment", "L2")


def rec(
    family: str, level: str, seed: int, repeat: int, cd: bool, es: bool | None = None, *,
    skill: str = "s0", llm_seed: int = 0, action: str = "rollback_deployment",
) -> EpisodeRecord:  # fmt: skip
    return EpisodeRecord(
        key=EpisodeKey("p", "c", "t", skill, family, seed, repeat, level),
        policy_config={},
        llm_seed=llm_seed,
        correct_disposition=cd,
        evidence_safe=cd if es is None else es,
        action=action,
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-11-24T00:00:00+00:00",
        duration_s=1.0,
    )


# 셀별 S0 성공 수 -----------------------------------------------------------------------------


def test_cell_rates_count_correct_cases_per_cell_and_metric() -> None:
    table = au.outcome_table(
        [rec(*BAD1, 1, 0, True, False), rec(*BAD1, 2, 0, False), rec(*BAD2, 1, 0, True)]
    )
    rates = au.cell_rates(table[0], "cd")
    assert rates["bad_deployment/L1"] == {"correct": 1, "cases": 2}
    assert rates["bad_deployment/L2"] == {"correct": 1, "cases": 1}
    assert au.cell_rates(table[0], "es")["bad_deployment/L1"] == {"correct": 0, "cases": 2}


def test_majority_table_needs_an_odd_number_of_repeats() -> None:
    table = au.outcome_table([rec(*BAD1, 1, r, r < 3) for r in range(5)])  # 5회 중 3회
    assert au.majority_table(table)[(*BAD1, 1)] == {"cd": 1, "es": 1}
    with pytest.raises(ValueError, match="odd"):
        au.majority_table(au.outcome_table([rec(*BAD1, 1, r, True) for r in range(4)]))


# Oracle 비대칭(단일 실행 대 다수결) ------------------------------------------------------------


def test_majority_offsets_compare_single_runs_and_repeat_zero_with_the_majority() -> None:
    # 케이스 1: 반복 0~4 = 0,1,1,1,0 (다수결 1). 케이스 2: 모두 1(다수결 1)
    pattern = [0, 1, 1, 1, 0]
    records = [rec(*BAD1, 1, r, bool(v)) for r, v in enumerate(pattern)]
    records += [rec(*BAD1, 2, r, True) for r in range(5)]
    got = au.majority_offsets(au.outcome_table(records), "cd")["bad_deployment/L1"]
    # 단일 실행 평균: 케이스 1은 3/5 - 1, 케이스 2는 0 -> 칸 평균 -1/5
    assert got["single_minus_majority"] == Fraction(-1, 5)
    assert got["r0_minus_majority"] == Fraction(-1, 2)
    assert (got["mixed_cases"], got["cases"]) == (1, 2)


# 다른 seed 반복 사이 불일치 -----------------------------------------------------------------------


def test_repeat_discordance_counts_every_pair_of_repeats() -> None:
    records = [rec(*BAD1, 1, r, r == 0) for r in range(3)]  # 반복 0만 성공
    got = au.repeat_discordance(au.outcome_table(records), "cd")
    assert got["bad_deployment/L1"] == {"differ": 2, "pairs": 3}  # (0,1), (0,2) 다름, (1,2) 같음
    assert got["all"] == {"differ": 2, "pairs": 3}


# 칸 끝 구조 -------------------------------------------------------------------------------------


def test_edge_cells_use_the_oracle_majority_and_every_repeat() -> None:
    correct = {"bad_deployment/L1": 0, "bad_deployment/L2": 50, "traffic_spike/L1": 5,
               "traffic_spike/L2": 6}  # fmt: skip
    cells = {c: {"correct": k, "cases": 50} for c, k in correct.items()}
    got = au.edge_cells(cells)
    assert got == {"floor": 1, "ceiling": 1, "near_floor": 2, "near_ceiling": 1, "cells": 4}


# seed 공유 -------------------------------------------------------------------------------------


def test_seed_sharing_counts_family_seed_groups_with_both_levels() -> None:
    cases = [("bad_deployment", "L1", 1), ("bad_deployment", "L2", 1), ("traffic_spike", "L1", 2)]
    assert au.seed_sharing(cases) == {"cases": 3, "seed_groups": 2, "shared_by_l1_l2": 1}


def test_llm_seed_rule_ignores_the_level() -> None:
    assert au.level_free_seed_rule() is True


def test_same_seed_pairs_match_candidate_and_s0_repeat_zero() -> None:
    s0 = [rec(*BAD1, 1, 0, True, llm_seed=7), rec(*BAD1, 2, 0, True, llm_seed=8)]
    cand = [
        rec(*BAD1, 1, 0, False, skill="c", llm_seed=7),
        rec(*BAD1, 2, 0, False, skill="c", llm_seed=9),
    ]
    assert au.same_seed_pairs({"c": cand}, s0) == {"candidates": 1, "pairs": 2, "same_seed": 1}
    with pytest.raises(ValueError, match="missing"):
        au.same_seed_pairs({"c": cand[:1]}, s0)


# 커밋된 009b 결과의 라벨 칸 ---------------------------------------------------------------


def analysis_stub() -> dict[str, Any]:
    tau = {c: {"exact": "1/20", "float": 0.05} for c in ("bad_deployment/L2", "traffic_spike/L1")}

    def cand(severe: bool, local: bool, cells: dict[str, str]) -> dict[str, Any]:
        oracle = {k: {"exact": v, "float": float(Fraction(v))} for k, v in cells.items()}
        return {
            "correct_disposition": {"severe": severe, "local_harm": local, "oracle_cells": oracle}
        }

    return {
        "tau": {"correct_disposition": tau},
        "candidates": {
            "a": cand(True, True, {"bad_deployment/L2": "-7/50", "traffic_spike/L1": "0"}),
            "b": cand(False, True, {"bad_deployment/L2": "-3/50", "traffic_spike/L1": "0"}),
            "c": cand(False, True, {"bad_deployment/L2": "-3/50", "traffic_spike/L1": "-4/50"}),
            "d": cand(False, False, {"bad_deployment/L2": "0", "traffic_spike/L1": "1/10"}),
        },
    }


def test_label_cells_split_severe_and_local_harm_only_by_the_cells_below_tau() -> None:
    got = au.label_cells(analysis_stub())
    assert got["severe"]["candidates"] == 1
    assert got["severe"]["worst_cell"] == {
        "closest_to_zero": Fraction(-7, 50),
        "farthest": Fraction(-7, 50),
    }
    only = got["local_harm_only"]
    assert only["candidates"] == 2
    assert only["with_cell"] == {"bad_deployment/L2": 2, "traffic_spike/L1": 1}
    assert only["this_cell_only"] == {"bad_deployment/L2": 1, "traffic_spike/L1": 0}


# Ticket 012 0단계 쌍 구조와 S0-prime 칸 ---------------------------------------------------


def three_runs() -> list[list[EpisodeRecord]]:
    # 두 케이스(L1, L2, 같은 family·seed 번호)
    # 케이스 L1은 1회째만 다르고, 케이스 L2는 재실행 2만 다르다
    base = [("bad_deployment", "L1", 1, 5), ("bad_deployment", "L2", 1, 5)]
    actions = {0: ("a", "b"), 1: ("x", "b"), 2: ("x", "y")}
    return [
        [rec(f, lv, s, 0, act == "a", skill="sp", llm_seed=ls, action=act)
         for (f, lv, s, ls), act in zip(base, actions[run], strict=True)]
        for run in range(3)
    ]  # fmt: skip


def test_t012_pairs_count_clusters_seeds_and_the_unused_rerun_pair() -> None:
    got = au.t012_pairs(three_runs())
    assert got == {
        "cases": 2, "llm_seeds": 1, "pairs": 4, "differing": 3,
        "cases_both_pairs_differ": 1, "cases_one_pair_differs": 1, "cases_no_pair_differs": 0,
        "reruns_differing": 1,
    }  # fmt: skip


def test_t012_pairs_refuse_runs_whose_seeds_differ() -> None:
    runs = three_runs()
    runs[1][0] = rec("bad_deployment", "L1", 1, 0, True, skill="sp", llm_seed=99, action="x")
    with pytest.raises(ValueError, match="llm_seed"):
        au.t012_pairs(runs)


def test_s0_prime_cells_give_per_run_counts_and_the_two_direction_rule() -> None:
    got = au.s0_prime_cells(three_runs())
    cell = got["cd"]["bad_deployment/L1"]
    assert cell == {
        "runs": [1, 0, 0],
        "cases": 1,
        "pooled": 1,
        "pooled_of": 3,
        "both_directions": True,
    }
    assert got["cd"]["bad_deployment/L2"]["both_directions"] is False  # 세 번 모두 실패
    assert got["both_directions_cells"]["cd"] == 1


# 조립과 고정 입력 -------------------------------------------------------------------------------


def test_the_definition_file_matches_its_pinned_sha256() -> None:
    import hashlib

    data = (REPO_ROOT / au.DEFINITION_PATH).read_bytes()
    assert hashlib.sha256(data).hexdigest() == au.DEFINITION_SHA256


def test_record_pins_come_from_committed_files() -> None:
    pins = au.record_pins(REPO_ROOT)
    summary = json.loads((REPO_ROOT / "writeup/inputs/run_summary_009b.json").read_text())
    for path, digest in pins.items():
        if path.startswith(("runs/t009_dev_b/", "runs/t009_oracle/")):
            assert summary["provenance"]["inputs"][path] == digest
    assert set(au.T012_RECORD_PINS) <= set(pins)
    assert any(p.startswith("runs/t009_calibration/") for p in pins)


def test_audit_bytes_are_sorted_and_end_with_a_newline() -> None:
    out = au.audit_bytes({"b": Fraction(1, 2), "a": [1]})
    assert out.decode().endswith("\n")
    assert json.loads(out) == {"a": [1], "b": {"exact": "1/2", "float": 0.5}}


def test_committed_audit_matches_its_sidecar() -> None:
    import hashlib

    path = REPO_ROOT / au.DEFAULT_OUT
    sidecar = path.with_name(path.name + ".sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sidecar
    data = json.loads(path.read_text())
    assert set(data) == {"audit", "provenance", "notes"}
    assert data["provenance"]["definition_sha256"] == au.DEFINITION_SHA256
