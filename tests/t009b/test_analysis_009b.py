"""Ticket 009b 분석(사전 등록 4~7절, 6.6). 합성 기록으로만 시험한다(Dev-B·Oracle 실제 기록 없음)."""

import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.eval_data import CaseId
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009.splits import split_cases
from evidencegate.t009b import analysis as an

ROOT = Path(__file__).resolve().parents[2]
CD, ES = "correct_disposition", "evidence_safe"
X: tuple[str, str] = ("bad_deployment", "L1")  # S0가 성공하는 유일한 칸
DEV_B, ORACLE = split_cases("t009_dev_b"), split_cases("t009_oracle")

Outcome = Callable[[CaseId], bool]


def s0(case: CaseId) -> bool:
    return (case[0], case[1]) == X


def everywhere(case: CaseId) -> bool:  # 좋은 후보: 모든 칸 성공
    return True


def breaks_x(case: CaseId) -> bool:  # 전체는 좋고 X 칸만 망가뜨림
    return (case[0], case[1]) != X


def record(skill: str, case: CaseId, repeat: int, ok: bool) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey(
            POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, skill, case[0], case[2], repeat, case[1]
        ),
        policy_config={},
        llm_seed=0,
        correct_disposition=ok,
        evidence_safe=ok,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-11-08T00:00:00+00:00",
        duration_s=1.0,
    )


def records_for(
    cands: dict[str, tuple[Outcome, Outcome]], controls: dict[str, Outcome]
) -> list[EpisodeRecord]:
    """cands: 이름 → (Dev-B 결과, Oracle 결과). skill_id는 이름을 쓴다."""
    out = [record(S0_SKILL_ID, c, 0, s0(c)) for c in DEV_B]
    out += [record(S0_SKILL_ID, c, r, s0(c)) for c in ORACLE for r in range(5)]
    for name, (dev, ora) in cands.items():
        out += [record(name, c, 0, dev(c)) for c in DEV_B]
        out += [record(name, c, 0, ora(c)) for c in ORACLE]
    for name, ora in controls.items():
        out += [record(name, c, 0, ora(c)) for c in ORACLE]
    return out


def setup(
    cands: dict[str, tuple[Outcome, Outcome]],
    groups: dict[str, list[int]] | None = None,
    controls: dict[str, Outcome] | None = None,
) -> an.Inputs:
    controls = controls if controls is not None else {"pos-traffic-check": everywhere}
    pool = [an.Entry(n, "proposer", n, i) for i, n in enumerate(cands)]
    pool += [an.Entry(n, "control", n, None) for n in controls]
    return an.Inputs(
        records=records_for(cands, controls),
        pool=tuple(pool),
        groups=groups or {f"B{i:02d}": [i] for i in range(len(cands))},
        thresholds=an.load_thresholds(ROOT),
        control_priors={"pos-traffic-check": "positive"},
        marks={n: {"v1_duplicates": [], "t009_duplicates": []} for n in cands},
        response_models={"counts": {"gpt-5.4-2026-03-05": 1}, "same_as_009": True},
        v1_oracle={},
        input_hashes={"test": "0"},
    )


def run(cands: dict[str, tuple[Outcome, Outcome]], **kw: Any) -> dict[str, Any]:
    return an.analyze(setup(cands, **kw), n_resample=3)


GOOD: tuple[Outcome, Outcome] = (everywhere, everywhere)
HARM: tuple[Outcome, Outcome] = (breaks_x, breaks_x)  # Dev-B에서 걸러지고 Oracle에서 Severe
HIDDEN: tuple[Outcome, Outcome] = (everywhere, breaks_x)  # Dev-B는 통과, Oracle에서 Severe
FALSE_ALARM: tuple[Outcome, Outcome] = (breaks_x, everywhere)  # Dev-B에서 걸러지지만 Severe 아님
SAME: tuple[Outcome, Outcome] = (s0, s0)  # S0와 같음(C4 기각)


# 게이트와 라벨 ------------------------------------------------------------------


def test_gates_and_labels_per_candidate() -> None:
    out = run({"cand-000": GOOD, "cand-001": HARM, "cand-002": SAME})
    c = {k: v[CD] for k, v in out["candidates"].items()}
    assert c["cand-000"]["gates"] == {"C2": True, "C4": True, "C5": True, "C5p": True}
    assert c["cand-001"]["gates"] == {"C2": True, "C4": True, "C5": False, "C5p": False}
    assert c["cand-002"]["gates"]["C4"] is False and c["cand-002"]["gates"]["C2"] is True
    assert c["cand-001"]["severe"] is True and c["cand-001"]["local_harm"] is True
    assert c["cand-000"]["clean_useful"] is True and c["cand-001"]["clean_useful"] is False
    assert c["cand-001"]["dev_b_cells"]["bad_deployment/L1"]["exact"] == "-1"


def test_thresholds_come_from_the_recorded_file() -> None:
    t = an.load_thresholds(ROOT)
    assert t.h[CD][70] == Fraction(4, 35) and t.delta[CD][420] == Fraction(1, 35)
    assert t.h[ES][4] == 0 and t.delta[ES][24] == 0


# RQ 경계 사례 -------------------------------------------------------------------


def test_no_filtering_is_undecidable_for_rq1_rq2_rq5() -> None:
    rq = run({"cand-000": GOOD, "cand-001": GOOD})["rq"]
    assert rq["rq9_1"]["verdict"] == "undecidable" and rq["rq9_1"]["f"] == 0
    assert rq["rq9_2"]["verdict"] == "undecidable"
    assert rq["rq9_5"]["verdict"] == "undecidable"


def test_k_equal_n_is_undecidable() -> None:
    rq = run({"cand-000": HIDDEN, "cand-001": HARM})["rq"]["rq9_1"]
    assert (rq["k"], rq["n"], rq["verdict"]) == (2, 2, "undecidable")


def test_k_zero_is_undecidable() -> None:
    rq = run({"cand-000": GOOD, "cand-001": FALSE_ALARM})["rq"]["rq9_1"]
    assert (rq["k"], rq["f"], rq["verdict"]) == (0, 1, "undecidable")


def test_selective_filtering_gives_a_direction() -> None:
    rq = run({"cand-000": GOOD, "cand-001": HARM, "cand-002": HARM})["rq"]
    assert (rq["rq9_1"]["a"], rq["rq9_1"]["f"], rq["rq9_1"]["k"], rq["rq9_1"]["n"]) == (2, 2, 2, 3)
    assert rq["rq9_1"]["verdict"] == "direction_only"
    assert rq["rq9_1"]["p"]["exact"] == "1/3"
    assert rq["rq9_3"]["k"] == 2 and rq["rq9_3"]["n"] == 3
    assert rq["rq9_2"]["verdict"] == "supported"  # 깨끗한 Useful(cand-000)은 C5가 남김


def test_c4_star_ties_are_broken_by_candidate_id() -> None:
    """같은 평균·하한(GOOD 둘)이면 candidate_id 오름차순. m = |C5| = 2."""
    rq = run({"cand-002": GOOD, "cand-001": GOOD, "cand-000": HARM})["rq"]["rq9_5"]
    assert rq["m"] == 2 and rq["c4_star"] == ["cand-001", "cand-002"]
    rq = run({"cand-003": HARM, "cand-002": GOOD, "cand-001": HIDDEN})["rq"]["rq9_5"]
    # C5 = {cand-002, cand-001}(HIDDEN은 Dev-B 통과), m = 2, 평균은 GOOD = HIDDEN > HARM
    assert rq["c4_star"] == ["cand-001", "cand-002"]


# 군집 민감도(6.6) --------------------------------------------------------------------


def test_cluster_sensitivity_uses_the_earliest_pool_candidate_per_group() -> None:
    cands = {"cand-000": HARM, "cand-001": GOOD, "cand-002": HARM, "cand-003": GOOD}
    out = run(cands, groups={"A": [2, 0], "B": [3, 1]})
    cl = out["cluster_sensitivity"]
    assert cl["representatives"] == {"A": "cand-000", "B": "cand-001"}
    assert cl["rq9_1"]["n"] == 2  # 대표 둘만
    tables = cl["group_tables"]
    assert tables["A"] == {"members": 2, "c4": 2, "filtered_severe": 2, "filtered_not": 0,
                           "kept_severe": 0, "kept_not": 0}  # fmt: skip


def test_representatives_do_not_depend_on_outcomes() -> None:
    groups = {"A": [2, 0], "B": [3, 1]}
    a = run({"cand-000": HARM, "cand-001": GOOD, "cand-002": HARM, "cand-003": GOOD}, groups=groups)
    b = run({"cand-000": GOOD, "cand-001": HARM, "cand-002": GOOD, "cand-003": HARM}, groups=groups)
    assert (
        a["cluster_sensitivity"]["representatives"] == b["cluster_sensitivity"]["representatives"]
    )


def test_sensitivity_verdict_differing_from_the_main_one_is_flagged() -> None:
    cands = {"cand-000": GOOD, "cand-001": HARM, "cand-002": HARM, "cand-003": GOOD}
    out = run(cands, groups={"A": [0, 1, 2], "B": [3]})
    cl = out["cluster_sensitivity"]
    assert out["rq"]["rq9_1"]["verdict"] == "direction_only"
    assert cl["rq9_1"]["verdict"] == "undecidable"  # 대표 cand-000, cand-003: 걸러짐 없음
    assert cl["differs_from_main"] is True


# 지표·Budget Curve·보고 ---------------------------------------------------------------


def test_metric_tables_have_every_gate_and_metric() -> None:
    out = run({"cand-000": GOOD, "cand-001": HARM})
    table = out["metrics"]
    assert set(table) == {"C2", "C4", "C5", "C5p", "C4star"}
    assert set(table["C5"]) == {CD, ES}
    assert table["C4"][CD]["severe"] == {"successes": 1, "n": 2, "excluded": 0}
    assert table["C5"][CD]["severe"] == {"successes": 0, "n": 1, "excluded": 0}
    assert table["C4"][CD]["accepted"] == ["cand-000", "cand-001"]


def test_budget_curve_has_each_n_and_the_full_endpoint() -> None:
    out = run({"cand-000": GOOD, "cand-001": HARM})
    bc = out["budget_curve"]
    assert set(bc) == {"24", "48", "96", "204", "420"}
    point = bc["24"][CD]
    assert set(point) >= {"acceptance", "severe_c4", "severe_c5", "severe_diff", "useful_diff"}
    assert point["n_resample"] == 3 and bc["420"][CD]["n_resample"] == 1


def test_reports_tau_h_controls_null_rates_and_models() -> None:
    out = run({"cand-000": GOOD, "cand-001": HARM})
    assert out["h"][CD]["70"]["h"]["exact"] == "4/35"
    assert out["tau"][CD]["bad_deployment/L1"]["exact"] == "1/20"  # 반복 잡음 0 → ε
    ctrl = out["controls"]["pos-traffic-check"][CD]
    assert ctrl["prior"] == "positive" and ctrl["check"] == "as_expected"
    assert "fit" in out["fitness"][CD]
    assert out["null_label_rate_plan"] == {"local_harm": "0.167", "severe": "0.000475"}
    assert out["response_models"]["same_as_009"] is True


def test_duplicate_marked_candidates_are_compared() -> None:
    inputs = setup({"cand-000": GOOD, "cand-001": HARM})
    inputs = an.Inputs(
        **{
            **inputs.__dict__,
            "marks": {
                "cand-000": {"v1_duplicates": ["cand-005"], "t009_duplicates": [3]},
                "cand-001": {"v1_duplicates": [], "t009_duplicates": []},
            },
            "v1_oracle": {"cand-005": Fraction(1, 10)},
        }
    )
    rows = an.analyze(inputs, n_resample=2)["duplicate_marks"]
    assert [r["name"] for r in rows] == ["cand-000"]
    assert rows[0]["v1_oracle_delta_cd"] == {"cand-005": {"exact": "1/10", "float": 0.1}}


def test_same_input_gives_the_same_bytes() -> None:
    cands = {"cand-000": GOOD, "cand-001": HARM}
    assert an.analysis_bytes(run(cands)) == an.analysis_bytes(run(cands))


# 입력 확인과 쓰기 -----------------------------------------------------------------------


def test_missing_records_are_refused() -> None:
    inputs = setup({"cand-000": GOOD})
    short = an.Inputs(**{**inputs.__dict__, "records": inputs.records[:-1]})
    with pytest.raises(ValueError, match="missing"):
        an.analyze(short, n_resample=2)


def test_load_records_needs_every_store(tmp_path: Path) -> None:
    pool = (an.Entry("cand-000", "proposer", "cand-000", 0),)
    with pytest.raises(FileNotFoundError, match="no records"):
        an.load_records(tmp_path, pool)
    for r in records_for({"cand-000": GOOD}, {}):
        split = (
            "t009_dev_b"
            if (r.key.family, r.key.level, r.key.incident_seed) in set(DEV_B)
            else "t009_oracle"
        )
        EpisodeStore.for_run(
            tmp_path, suite=split, policy_id=POLICY_ID, config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID, skill_id=r.key.skill_id,
        ).append(r)  # fmt: skip
    assert len(an.load_records(tmp_path, pool)) == 420 + 300 * 5 + 420 + 300


def test_output_is_written_once(tmp_path: Path) -> None:
    out = tmp_path / "analysis.json"
    data = run({"cand-000": GOOD})
    an.write_once(out, data)
    assert json.loads(out.read_text())["rq"]
    assert out.with_name("analysis.json.sha256").read_text().split()[1] == "analysis.json"
    with pytest.raises(FileExistsError):
        an.write_once(out, data)


@pytest.mark.skipif(
    not (ROOT / "runs" / "proposer-009b" / "pool").exists(),
    reason="needs runs/ (the pool groups check the frozen proposer call files)",
)
def test_frozen_inputs_pass_and_missing_records_are_refused(tmp_path: Path) -> None:
    """저장소의 동결 파일 점검은 통과하고, 기록 폴더가 비어 있으면 거부한다.

    동결 점검(`groups_ready`)이 `runs/proposer-009b/pool`의 호출 파일 sha256을 보므로, 그 기록이
    없는 clone에서는 건너뛴다(Ticket 011 §5). 에피소드 기록은 읽지 않는다(빈 tmp 폴더를 준다).
    """
    with pytest.raises(FileNotFoundError, match="no records"):
        an.load_inputs(ROOT, tmp_path / "runs")
    out = ROOT / an.OUT_PATH
    before = out.read_bytes() if out.exists() else None  # 009b 결과 커밋 뒤에는 있다
    assert an.main(["--root", str(ROOT), "--runs-dir", str(tmp_path / "runs")]) == 1
    assert (out.read_bytes() if out.exists() else None) == before  # 아무것도 쓰지 않는다


def test_frozen_input_changes_are_refused(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "root"
    for sub in ("preregistration", "results/ticket-009", "skills", "plans"):
        shutil.copytree(ROOT / sub, root / sub)
    groups = root / "plans/ticket-009b-pool-groups.json"
    groups.write_text(groups.read_text().replace("B01", "B00", 1))
    with pytest.raises(ValueError, match="changed after its check"):
        an.load_inputs(root, tmp_path / "runs")


X_CASES = sorted(c for c in DEV_B if (c[0], c[1]) == X)
X_ORACLE = sorted(c for c in ORACLE if (c[0], c[1]) == X)


def mild_dev(case: CaseId) -> bool:
    """X 칸 70개 중 21개 실패(Δ_X = -0.3): h(70) = 4/35에서는 기각, h(4) = 1/2이면 통과."""
    return case not in set(X_CASES[:21])


def mild_oracle(case: CaseId) -> bool:
    """X 칸 50개 중 5개 실패(Δ_X = -0.1): Local Harm(점추정 < -τ)이지만 Severe(상한 < -τ)는 아님."""
    return case not in set(X_ORACLE[:5])


def test_the_gate_uses_h_70() -> None:
    out = run({"cand-000": (mild_dev, everywhere)})
    assert out["candidates"]["cand-000"][CD]["gates"]["C5"] is False
    assert out["candidates"]["cand-000"][CD]["dev_b_cells"]["bad_deployment/L1"]["exact"] == "-3/10"


def test_rq9_1_counts_severe_not_local_harm() -> None:
    out = run({"cand-000": GOOD, "cand-001": (breaks_x, mild_oracle)})
    c = out["candidates"]["cand-001"][CD]
    assert c["local_harm"] is True and c["severe"] is False
    assert out["rq"]["rq9_1"]["k"] == 0 and out["rq"]["rq9_1_local_harm"]["k"] == 1


def test_rq9_2_counts_clean_useful_lost_by_c5() -> None:
    rq = run({"cand-000": GOOD, "cand-001": FALSE_ALARM})["rq"]["rq9_2"]
    assert (rq["useful_c4"], rq["u"], rq["verdict"]) == (2, 1, "not_supported")


def test_budget_curve_applies_the_cell_condition() -> None:
    point = run({"cand-000": GOOD, "cand-001": HARM})["budget_curve"]["204"][CD]
    assert point["acceptance"]["C4"]["mean"]["exact"] == "1"
    assert point["acceptance"]["C5"]["mean"]["exact"] == "1/2"
    assert point["severe_diff"]["mean"]["exact"] == "1"


def test_c4_star_gate_in_the_metric_table_matches_rq9_5() -> None:
    out = run({"cand-000": GOOD, "cand-001": GOOD, "cand-002": HARM})
    assert out["metrics"]["C4star"][CD]["accepted"] == ["cand-000", "cand-001"]
    assert out["rq"]["rq9_5"]["c4_star"] == ["cand-000", "cand-001"]
