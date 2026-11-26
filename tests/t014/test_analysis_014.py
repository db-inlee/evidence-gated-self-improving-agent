"""Ticket 014 분석(사전 등록 4~7절). 합성 기록만 쓴다. LLM 없음."""

import random
from fractions import Fraction as F

import pytest

from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.eval_data import CELLS, CaseId
from evidencegate.exact import clopper_pearson, null_diff_quantile_exact
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.t009.rq9 import cell_threshold_h, fisher_one_sided, rq9_1
from evidencegate.t013.sim import sage_accept, tango_lower
from evidencegate.t014 import analysis as an
from evidencegate.t014.settings import PREDICTED_LOSS


def cases(per_cell: int, first: int = 1) -> list[CaseId]:
    return [(f, lv, first + i) for f, lv in CELLS for i in range(per_cell)]


def record(case: CaseId, repeat: int, cd: bool, es: bool) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey(
            POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, S0_SKILL_ID, case[0], case[2], repeat, case[1]
        ),
        policy_config={},
        llm_seed=0,
        correct_disposition=cd,
        evidence_safe=es,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-11-27T00:00:00+00:00",
        duration_s=1.0,
    )


# 5절 보정 -----------------------------------------------------------------------------


def test_thresholds_call_the_009_algorithms_with_n_c_150_and_n_900() -> None:
    rng = random.Random(3)
    cal = cases(12)
    records = [
        record(c, r, cd := rng.random() < 0.6, cd and rng.random() < 0.5)
        for c in cal
        for r in range(5)
    ]
    got = an.thresholds(records, cal, n_boot=40)
    table = {
        r: {
            c: int(x.correct_disposition)
            for x in records
            if x.key.repeat == r
            for c in [(x.key.family, x.key.level, x.key.incident_seed)]
        }
        for r in range(5)
    }
    h, rate = cell_threshold_h(table, 150, n_boot=40)
    assert got["correct_disposition"]["h"] == h and got["correct_disposition"]["achieved"] == rate
    delta = null_diff_quantile_exact(records, "correct_disposition", 900, n_boot=40)
    assert got["correct_disposition"]["delta"] == delta
    assert set(got) == {"correct_disposition", "evidence_safe"}


def test_thresholds_refuse_missing_records() -> None:
    cal = cases(3)
    records = [record(c, r, True, True) for c in cal for r in range(5)][1:]
    with pytest.raises(ValueError, match="missing"):
        an.thresholds(records, cal, n_boot=10)


# 4절 게이트 ---------------------------------------------------------------------------


def paired(
    per_cell: int, cand_cell: dict[int, list[int]], s0_cell: dict[int, list[int]]
) -> tuple[dict[CaseId, int], dict[CaseId, int]]:
    cs = cases(per_cell)
    cand = {c: cand_cell[i // per_cell][i % per_cell] for i, c in enumerate(cs)}
    s0 = {c: s0_cell[i // per_cell][i % per_cell] for i, c in enumerate(cs)}
    return cand, s0


def test_a_uniformly_better_candidate_passes_c4_c5_c6_and_sage() -> None:
    n = 40
    s0 = {k: [1] * 20 + [0] * 20 for k in range(6)}
    cand = {k: [1] * 40 for k in range(6)}
    row = an.gate_row("x", "correct_disposition", *paired(n, cand, s0), h=F(1, 10), delta=F(1, 50))
    assert row.c4 and row.c5 and row.c6 and row.sage
    assert row.mean == F(1, 2) and row.w == 120 and row.loss == 0 and row.n_s == 120
    assert row.cell_deltas == (F(1, 2),) * 6
    assert row.tango_lowers == tuple(tango_lower(20, 0, 40, F(1, 20)) for _ in range(6))


def test_one_losing_cell_fails_c6_and_c5_but_not_c4() -> None:
    n = 40
    s0 = {k: [1] * 20 + [0] * 20 for k in range(6)}
    cand = {k: [1] * 40 for k in range(5)} | {5: [0] * 10 + [1] * 10 + [0] * 20}
    row = an.gate_row("x", "correct_disposition", *paired(n, cand, s0), h=F(1, 10), delta=F(1, 50))
    assert row.c4 and not row.c5 and not row.c6
    assert row.cell_deltas[5] == F(-1, 4)
    assert row.sage == sage_accept(row.w, row.loss, row.n_s, lam=1, tau=0.05, alpha=F(1, 20))


def test_c5_and_c6_need_c4() -> None:
    s0 = {k: [1] * 20 + [0] * 20 for k in range(6)}
    row = an.gate_row("x", "correct_disposition", *paired(40, s0, s0), h=F(1, 10), delta=F(1, 50))
    assert not row.c4 and not row.c5 and not row.c6 and not row.sage


def test_a_tango_lower_within_the_tie_band_of_minus_m_does_not_pass() -> None:
    assert not an.tango_pass(-0.05 + 1e-13)
    assert not an.tango_pass(-0.05)
    assert an.tango_pass(-0.05 + 1e-11)


def test_oracle_targets_are_c4_or_sage_accepted_in_cd() -> None:
    def row(name: str, c4: bool, sage: bool) -> an.GateRow:
        return an.GateRow(
            name,
            "correct_disposition",
            F(0),
            F(0),
            (F(0),) * 6,
            0,
            0,
            0,
            (0.0,) * 6,
            c4,
            c4,
            c4,
            sage,
        )

    rows = [row("a", True, False), row("b", False, True), row("c", False, False)]
    assert an.oracle_targets(rows) == ["a", "b"]


# 6절 Oracle ---------------------------------------------------------------------------


def reps(per_cell: int, by_cell: dict[int, list[list[int]]]) -> dict[int, dict[CaseId, int]]:
    """by_cell[칸][반복] = 칸 케이스 값 목록."""
    cs = cases(per_cell)
    return {
        r: {c: by_cell[i // per_cell][r][i % per_cell] for i, c in enumerate(cs)} for r in range(3)
    }


def test_symmetric_mean_label_marks_m_when_a_cell_upper_is_below_minus_m() -> None:
    n = 10
    s0 = reps(n, {k: [[1] * n] * 3 for k in range(6)})
    cand = reps(n, {0: [[0] * n] * 3} | {k: [[1] * n] * 3 for k in range(1, 6)})
    label = an.oracle_label(cand, s0, n_boot=50)
    assert label.cells[0] == (F(-1), F(-1), F(-1))
    assert label.m and label.m_point and not label.useful and not label.clean_useful
    assert label.mean == F(-1, 6)


def test_mean_aggregation_divides_case_sums_by_three() -> None:
    n = 6
    s0 = reps(n, {k: [[0] * n] * 3 for k in range(6)})
    one_of_three = [[1] * n, [0] * n, [0] * n]
    cand = reps(n, {k: one_of_three for k in range(6)})
    label = an.oracle_label(cand, s0, n_boot=20)
    assert label.mean == F(1, 3) and label.lower == F(1, 3) == label.upper
    assert label.useful and label.clean_useful and not label.m


def test_m_uncertain_when_a_cell_interval_crosses_minus_m() -> None:
    n = 20
    s0 = reps(n, {k: [[1] * n] * 3 for k in range(6)})
    # 칸 0: 20 케이스 중 1개만 세 번 모두 실패 -> 점추정 -1/20(경계), 구간은 -m을 가로지른다
    worse = [[0] + [1] * (n - 1)] * 3
    cand = reps(n, {0: worse} | {k: [[1] * n] * 3 for k in range(1, 6)})
    label = an.oracle_label(cand, s0, n_boot=200)
    lo, mid, hi = label.cells[0][1], label.cells[0][0], label.cells[0][2]
    assert mid == F(-1, 20) and lo < F(-1, 20) and hi >= F(-1, 20)
    assert not label.m and not label.m_point and label.m_uncertain


def test_useful_uncertain_when_the_overall_interval_crosses_eps() -> None:
    n = 20
    s0 = reps(n, {k: [[0] * n] * 3 for k in range(6)})
    some = [[1, 0, 0, 0] * 5] * 3
    cand = reps(n, {k: some for k in range(6)})
    label = an.oracle_label(cand, s0, n_boot=200)
    assert label.lower <= F(1, 20) < label.upper or label.useful
    assert label.useful_uncertain == (label.lower <= F(1, 20) < label.upper)


def test_oracle_label_refuses_other_repeat_sets() -> None:
    n = 2
    s0 = reps(n, {k: [[0] * n] * 3 for k in range(6)})
    with pytest.raises(ValueError, match="repeats"):
        an.oracle_label({0: s0[0]}, s0, n_boot=5)


# 7절 RQ -------------------------------------------------------------------------------


def g(
    name: str, *, c4: bool = True, c6: bool = True, c5: bool = True, sage: bool = False
) -> an.GateRow:
    return an.GateRow(
        name,
        "correct_disposition",
        F(0),
        F(0),
        (F(0),) * 6,
        0,
        0,
        0,
        (0.0,) * 6,
        c4,
        c5 and c4,
        c6 and c4,
        sage,
    )


def lab(m: bool = False, useful: bool = False, m_uncertain: bool = False) -> an.OracleLabel:
    return an.OracleLabel(
        F(0),
        F(0),
        F(0),
        ((F(0), F(0), F(0)),) * 6,
        m,
        m,
        useful,
        useful and not m,
        m_uncertain,
        False,
    )


def test_rq14_1_is_rq9_1_on_c6_rejection_and_m() -> None:
    rows = [g("a", c6=False), g("b", c6=False), g("c"), g("d"), g("e", c4=False)]
    labels = {
        "a": lab(m=True),
        "b": lab(m=True),
        "c": lab(),
        "d": lab(m_uncertain=True),
        "e": lab(),
    }
    got = an.rq14_1(rows, labels)
    main = rq9_1([(False, True), (False, True), (True, False), (True, False)])
    assert got["verdict"] == main["verdict"] and got["p"] == main["p"]
    assert (got["a"], got["f"], got["k"], got["n"]) == (2, 2, 2, 4)
    assert got["m_uncertain"] == 1
    moved = rq9_1([(False, True), (False, True), (True, False), (True, True)])
    assert got["sensitivity"]["uncertain_as_m"]["verdict"] == moved["verdict"]
    assert got["sensitivity"]["uncertain_as_not_m"]["verdict"] == main["verdict"]


def test_rq14_2_reports_the_observed_loss_against_the_prediction() -> None:
    rows = [g("a", c6=False), g("b"), g("c"), g("d")]
    labels = {k: lab(useful=True) for k in "abcd"}
    got = an.rq14_2(rows, labels)
    assert got["u_4"] == 4 and got["u"] == 1 and got["loss"] == F(1, 4)
    interval = clopper_pearson(1, 4)
    assert interval is not None
    lo, hi = interval
    assert got["interval"] == (lo, hi) and got["contains_prediction"] == (
        lo <= PREDICTED_LOSS <= hi
    )
    assert got["predicted_loss"] == F(20648, 80000)
    assert "verdict" not in got
    none = an.rq14_2([g("a")], {"a": lab(m=True, useful=True)})
    assert none["u_4"] == 0 and none["loss"] is None and none["contains_prediction"] == "undefined"


def test_rq14_3_tallies_each_gate_inside_the_oracle_targets() -> None:
    rows = [g("a", c6=False, c5=False), g("b"), g("c", c4=False, sage=True), g("d", c4=False)]
    labels = {"a": lab(m=True), "b": lab(useful=True), "c": lab(useful=True)}
    got = an.rq14_3(rows, labels)
    assert got["C4"] == {"accepted": 2, "m": 1, "m_point": 1, "clean_useful": 1}
    assert got["C5"] == {"accepted": 1, "m": 0, "m_point": 0, "clean_useful": 1}
    assert got["C6_IUT"] == got["C5"]
    assert got["SAGE_l1"] == {"accepted": 1, "m": 0, "m_point": 0, "clean_useful": 1}


def test_rq14_3_refuses_an_accepted_candidate_without_a_label() -> None:
    with pytest.raises(ValueError, match="no Oracle label"):
        an.rq14_3([g("a")], {})


def test_rq14_1_reports_the_smallest_achievable_p_for_n_f_k() -> None:
    rows = [g("a", c6=False), g("b", c6=False), g("c"), g("d")]
    labels = {"a": lab(), "b": lab(m=True), "c": lab(m=True), "d": lab()}
    got = an.rq14_1(rows, labels)
    assert (got["n"], got["f"], got["k"], got["a"]) == (4, 2, 2, 1)
    assert got["p_min"] == fisher_one_sided(a=2, f=2, k=2, n=4) == F(1, 6)
    assert got["p"] != got["p_min"]


def test_rq14_1_p_min_is_one_when_undecidable_by_f_or_k() -> None:
    got = an.rq14_1([g("a"), g("b")], {"a": lab(), "b": lab(m=True)})
    assert got["verdict"] == "undecidable" and got["p_min"] == 1


def test_no_c4_candidates_give_undecidable_undefined_and_zeros() -> None:
    rows = [g("a", c4=False), g("b", c4=False)]
    assert an.rq14_1(rows, {})["verdict"] == "undecidable"
    assert an.rq14_2(rows, {})["contains_prediction"] == "undefined"
    assert all(
        v == {"accepted": 0, "m": 0, "m_point": 0, "clean_useful": 0}
        for v in an.rq14_3(rows, {}).values()
    )
