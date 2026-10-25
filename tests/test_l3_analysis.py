"""Ticket 006 L3 분석 (§4, RQ-L3-0~3, §10). 합성 레코드로 처음부터 끝까지 돌린다.

Ticket 005 결과는 같은 합성 세계에서 Ticket 005 분석을 돌려 만든다(실제 실행에서는
results/ticket-005/analysis.json).
"""

import hashlib
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate.analysis import run_analysis
from evidencegate.eval_constants import EPSILON, S0_ORACLE_REPEATS
from evidencegate.eval_data import CELLS, PairedTable, majority, split_cases
from evidencegate.eval_metrics import CandidateScore, Rate, gate_metrics
from evidencegate.exact import (
    mean_fraction,
    null_diff_quantile_exact,
    percentile_exact,
    stratified_bootstrap_indices,
)
from evidencegate.gates import bootstrap_quantiles
from evidencegate.l3_analysis import (
    FAMILY_CELLS,
    L3_CELLS,
    ManipulationSample,
    l3_split,
    main,
    manipulation_check,
    render_markdown_l3,
    rq_l3_1,
    rq_l3_2,
    rq_l3_3,
    run_l3_analysis,
    ticket005_bytes,
)
from evidencegate.l3_bench import l3_split_cases
from evidencegate.oracle import OracleEstimate, tau_cells
from test_analysis import POOL, S0, record, world

CD = "correct_disposition"
ES = "evidence_safe"
L3_PREREG = "e368d9ed903757d9424ca5455dfcd5ce19519066bba849ef86b3b2f527f34a4b"
REPO_ROOT = Path(__file__).resolve().parents[1]
# L3에서 후보 효과(뒤집힘 확률). cand-001은 dep 칸에서만 크게 해롭다(L3에서 해로움이 커지게).
L3_EFFECTS = {
    "p0skill00000": (0.3, 0.05),
    "p1skill00000": (0.05, 0.5),
    "p2skill00000": (0.1, 0.1),
    "p3skill00000": (0.3, 0.0),
    "c0skill00000": (0.4, 0.0),
    "c1skill00000": (0.0, 0.4),
}
N_RESAMPLE = 5
RUN_INFO = {"analysis_commit": "abc1234", "plan_sha256": "f" * 64}


def l3_world(seed: int = 1) -> list[Any]:
    rng = random.Random(seed)
    records = []
    base = {}
    for case in l3_split_cases("l3_oracle"):
        base[case] = (rng.random() < 0.45, rng.random() < 0.3)
        for r in S0_ORACLE_REPEATS:
            cd, es = base[case]
            if r > 0 and rng.random() < 0.05:
                cd = not cd
            records.append(record(S0, case, r, cd, es and cd))
    for spec in POOL.candidates:
        up, down = L3_EFFECTS[spec.skill_id]
        for case in l3_split_cases("l3_oracle"):
            cd, es = base[case]
            if not cd and rng.random() < up:
                cd, es = True, rng.random() < 0.7
            elif cd and rng.random() < down:
                cd, es = False, False
            records.append(record(spec.skill_id, case, 0, cd, es and cd))
    return records


def t005_bytes_for(records: list[Any], noise: list[Any]) -> bytes:
    t005 = run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=N_RESAMPLE)
    return ticket005_bytes(t005)


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    records, noise = world()
    return {
        "records": records,
        "noise": noise,
        "l3": l3_world(),
        "t005": t005_bytes_for(records, noise),
    }


@pytest.fixture(scope="module")
def result(inputs: dict[str, Any]) -> dict[str, Any]:
    return run_l3_analysis(
        inputs["records"],
        inputs["l3"],
        inputs["noise"],
        POOL,
        ticket005=inputs["t005"],
        run_info={"analysis_commit": "def5678", "plan_sha256": "e" * 64},
    )


# ---------------------------------------------------------------------------
# 입력과 Ticket 005 대조 (§10 해석: 게이트 결정은 Ticket 005 그대로)
# ---------------------------------------------------------------------------


def test_meta_records_hashes_versions_and_settings(result: dict[str, Any], inputs: Any) -> None:
    meta = result["meta"]
    assert meta["analysis_commit"] == "def5678"
    assert meta["plan_sha256"] == "e" * 64
    assert meta["preregistration_l3_sha256"] == L3_PREREG
    assert meta["preregistration_7_8_sha256"].startswith("7611e83d")
    assert meta["ticket005_analysis_sha256"] == hashlib.sha256(inputs["t005"]).hexdigest()
    assert (meta["n_resample"], meta["n_boot"], meta["seed"]) == (N_RESAMPLE, 2000, 0)
    assert meta["numpy"] == "2.3.4" and meta["primary_metric"] == CD
    assert meta["section10_version"].startswith("2026-10-25")


def test_a_ticket_005_result_that_does_not_reproduce_stops_the_analysis(inputs: Any) -> None:
    data = json.loads(inputs["t005"])
    data["candidates"]["cand-000"]["gates"][CD]["C2"]["accepted"] ^= True
    with pytest.raises(ValueError, match="does not reproduce"):
        run_l3_analysis(
            inputs["records"],
            inputs["l3"],
            inputs["noise"],
            POOL,
            ticket005=ticket005_bytes(data),
            run_info=RUN_INFO,
        )


def test_missing_l3_records_stop_the_analysis(inputs: Any) -> None:
    l3 = [r for r in inputs["l3"] if not (r.key.skill_id == S0 and r.key.repeat == 4)]
    with pytest.raises(ValueError, match="missing"):
        run_l3_analysis(
            inputs["records"],
            l3,
            inputs["noise"],
            POOL,
            ticket005=inputs["t005"],
            run_info=RUN_INFO,
        )


def test_l3_records_from_another_run_condition_stop_the_analysis(inputs: Any) -> None:
    l3 = list(inputs["l3"])
    r = l3[0]
    l3[0] = type(r)(**{**r.__dict__, "key": type(r.key)(**{**r.key.__dict__, "config_id": "x"})})
    with pytest.raises(ValueError, match=r"config_id|mixed"):
        run_l3_analysis(
            inputs["records"],
            l3,
            inputs["noise"],
            POOL,
            ticket005=inputs["t005"],
            run_info=RUN_INFO,
        )


def test_episode_counts_are_for_the_l3_records(result: dict[str, Any]) -> None:
    episodes = result["episodes"]
    assert episodes["total"] == 300 * 5 + 300 * len(POOL.candidates)
    assert episodes["no_disposition"] == 0 and episodes["format_error_episodes"] == {}


def test_l3_split_is_300_cases_in_three_family_cells() -> None:
    cases = l3_split()
    assert len(cases) == 300 and cases == l3_split_cases("l3_oracle")
    assert tuple(sorted((c[0], "L3") for c in cases[::100])) == L3_CELLS


# ---------------------------------------------------------------------------
# τ와 L3 oracle Δ (§4)
# ---------------------------------------------------------------------------


def s0_l3(records: list[Any], metric: str) -> dict[int, dict[Any, int]]:
    cases = l3_split_cases("l3_oracle")
    out: dict[int, dict[Any, int]] = {}
    for r in S0_ORACLE_REPEATS:
        got = {
            (x.key.family, x.key.level, x.key.incident_seed): int(getattr(x, metric))
            for x in records
            if x.key.skill_id == S0 and x.key.repeat == r
        }
        out[r] = {c: got[c] for c in cases}
    return out


def test_tau_tables_side_by_side(result: dict[str, Any], inputs: Any) -> None:
    t005 = json.loads(inputs["t005"])
    for m in (CD, ES):
        tau = result["tau"][m]
        assert tau["L1/L2"] == t005["tau"][m]
        expected = tau_cells(s0_l3(inputs["l3"], m), cells=L3_CELLS)
        assert {k: v["exact"] for k, v in tau["L3"].items()} == {
            f"{c[0]}/{c[1]}": str(v) for c, v in expected.items()
        }
        assert set(tau["family_L1L2"]) == {f"{c[0]}/{c[1]}" for c in FAMILY_CELLS}


def l3_table(inputs: Any, skill: str, metric: str) -> PairedTable:
    cases = l3_split_cases("l3_oracle")
    cand = {
        (x.key.family, x.key.level, x.key.incident_seed): int(getattr(x, metric))
        for x in inputs["l3"]
        if x.key.skill_id == skill
    }
    base = majority(s0_l3(inputs["l3"], metric))
    return PairedTable(
        "l3_oracle",
        metric,
        cases,
        tuple((c[0], c[1]) for c in cases),
        tuple(cand[c] for c in cases),
        tuple(base[c] for c in cases),
    )


def test_l3_oracle_delta_interval_and_cells_by_hand(result: dict[str, Any], inputs: Any) -> None:
    t = l3_table(inputs, "p1skill00000", CD)
    o = result["candidates"]["cand-001"]["oracle_l3"][CD]
    assert o["delta"]["exact"] == str(mean_fraction(t.deltas))
    low, high = bootstrap_quantiles(t, (Fraction(1, 40), Fraction(39, 40)), cells=L3_CELLS)
    assert [o["interval"][0]["exact"], o["interval"][1]["exact"]] == [str(low), str(high)]
    for cell, idx in zip(L3_CELLS, t.cell_indices(L3_CELLS), strict=True):
        expected = mean_fraction([t.deltas[i] for i in idx])
        assert o["cells"][f"{cell[0]}/{cell[1]}"]["exact"] == str(expected)
    assert set(o) >= {"useful", "harmful", "unsupported", "local_harm", "effect", "subset"}


def test_fitness_is_not_applied_to_l3(result: dict[str, Any]) -> None:
    assert result["fitness"] == "not_applicable"


# ---------------------------------------------------------------------------
# 지표 6개와 비교 표 (§4)
# ---------------------------------------------------------------------------


def test_l1_l2_metrics_are_the_ticket_005_metrics(result: dict[str, Any], inputs: Any) -> None:
    t005 = json.loads(inputs["t005"])
    for group in ("proposer", "control"):
        for m in (CD, ES):
            for gate in ("C1", "C2", "C3", "C4"):
                assert result["metrics"][group][m][gate]["l12"] == t005["metrics"][group][m][gate]


def test_l3_metrics_use_ticket_005_decisions_and_l3_estimates(
    result: dict[str, Any], inputs: Any
) -> None:
    t005 = json.loads(inputs["t005"])
    m = CD
    scores = []
    for spec in POOL.candidates:
        if spec.group != "proposer":
            continue
        accepted = {
            (g, mm): t005["candidates"][spec.name]["gates"][mm][g]["accepted"]
            for g in ("C1", "C2", "C3", "C4")
            for mm in (CD, ES)
        }
        t = l3_table(inputs, spec.skill_id, m)
        o = result["candidates"][spec.name]["oracle_l3"][m]
        tau = {c: Fraction(result["tau"][m]["L3"][f"{c[0]}/{c[1]}"]["exact"]) for c in L3_CELLS}
        low, high = bootstrap_quantiles(t, (Fraction(1, 40), Fraction(39, 40)), cells=L3_CELLS)
        cells = {
            c: mean_fraction([t.deltas[i] for i in idx])
            for c, idx in zip(L3_CELLS, t.cell_indices(L3_CELLS), strict=True)
        }
        e = OracleEstimate(m, mean_fraction(t.deltas), (low, high), cells, tau)
        assert o["delta"]["exact"] == str(e.delta)
        scores.append(CandidateScore(spec.name, accepted, {m: e, ES: e}))
    gm = gate_metrics(scores, "C2", m)
    got = result["metrics"]["proposer"][m]["C2"]["l3"]
    assert got["accepted"] == list(gm.accepted)
    assert (got["harmful"]["successes"], got["harmful"]["n"]) == (
        gm.harmful.successes,
        gm.harmful.n,
    )
    assert (got["local_harm"]["successes"], got["local_harm"]["n"]) == (
        gm.local_harm.successes,
        gm.local_harm.n,
    )


def test_comparison_table_has_l12_l3_and_difference(result: dict[str, Any]) -> None:
    row = result["comparison"]["proposer"][CD]["C1"]
    assert set(row) == {
        "harmful",
        "unsupported",
        "useful",
        "gain",
        "false_rejection",
        "local_harm",
    }
    for v in row.values():
        if v["l12"] is not None and v["l3"] is not None:
            assert Fraction(v["diff"]["exact"]) == Fraction(v["l3"]["exact"]) - Fraction(
                v["l12"]["exact"]
            )
        else:
            assert v["diff"] is None
    assert set(result["comparison"]) == {"proposer", "control"}


# ---------------------------------------------------------------------------
# Local Harm 보조 지표: L1/L2를 family 단위로 다시 묶음 (§4)
# ---------------------------------------------------------------------------


def test_family_level_local_harm_by_hand(result: dict[str, Any], inputs: Any) -> None:
    from evidencegate.eval_data import paired_oracle, repeat_outcomes

    m = CD
    t = paired_oracle(
        inputs["records"], candidate_skill_id="p1skill00000", s0_skill_id=S0, metric=m
    )
    reps = repeat_outcomes(
        inputs["records"], skill_id=S0, split="oracle", metric=m, repeats=S0_ORACLE_REPEATS
    )
    tau = tau_cells(reps, cells=FAMILY_CELLS, cell_key=lambda c: (c[0], "L1+L2"))
    fam = result["candidates"]["cand-001"]["family_l12"][m]
    harm = False
    for cell in FAMILY_CELLS:
        idx = [i for i, c in enumerate(t.cases) if c[0] == cell[0]]
        assert len(idx) == 100
        d = mean_fraction([t.deltas[i] for i in idx])
        assert fam["cells"][f"{cell[0]}/{cell[1]}"]["exact"] == str(d)
        harm = harm or d < -tau[cell]
    assert fam["local_harm"] == harm
    lh = result["local_harm"]["proposer"][m]["C1"]
    assert set(lh) == {"l3", "family_l12"}


# ---------------------------------------------------------------------------
# RQ-L3-0: 조작 확인 (독립 bootstrap, 원래 설계 칸으로 층화, §10 해석 1)
# ---------------------------------------------------------------------------


def sample(groups: list[list[tuple[int, int]]], n: int) -> ManipulationSample:
    return ManipulationSample(groups=tuple(tuple(g) for g in groups), n=n)


def test_manipulation_confirmed_when_l3_is_clearly_harder() -> None:
    l12 = sample([[(1, 1)] * 10, [(1, 0)] * 10], 20)
    l3 = sample([[(0, 0)] * 20], 20)
    r = manipulation_check(l12, l3, n_boot=200)
    assert r["cd"]["delta"]["exact"] == "-1"
    assert r["cd"]["holds"] is True and r["verdict"] == "confirmed"


def test_manipulation_not_confirmed_when_both_are_surely_false() -> None:
    same = sample([[(1, 1)] * 10, [(1, 1)] * 10], 20)
    l3 = sample([[(1, 1)] * 20], 20)  # 재표본마다 값이 같아 구간이 [0, 0]
    r = manipulation_check(same, l3, n_boot=200)
    assert r["cd"]["holds"] is False and r["gap"]["holds"] is False
    assert r["verdict"] == "not_confirmed"


def test_manipulation_undecidable_when_an_interval_crosses_the_boundary() -> None:
    rng = random.Random(3)
    l12 = sample([[(int(rng.random() < 0.6), 0) for _ in range(30)] for _ in range(2)], 60)
    l3 = sample([[(int(rng.random() < 0.55), 0) for _ in range(30)]], 30)
    r = manipulation_check(l12, l3, n_boot=400)
    low, high = (Fraction(x["exact"]) for x in r["cd"]["interval"])
    assert low < -EPSILON < high
    assert r["cd"]["holds"] is None and r["verdict"] == "undecidable"


def test_manipulation_gap_uses_cd_minus_es() -> None:
    """CD는 같고 L3에서 ES만 떨어지면 g(L3) - g(L1/L2) = 1 > ε: 조작 확인."""
    l12 = sample([[(1, 1)] * 10, [(1, 1)] * 10], 20)
    l3 = sample([[(1, 0)] * 20], 20)
    r = manipulation_check(l12, l3, n_boot=200)
    assert r["cd"]["holds"] is False
    assert r["gap"]["delta"]["exact"] == "1" and r["gap"]["holds"] is True
    assert r["verdict"] == "confirmed"


def test_manipulation_bootstrap_is_one_seed_0_stream_l12_cells_first() -> None:
    """§10 해석: 칸 크기 목록 [L1/L2 칸..., L3 칸...]로 seed 0 흐름 하나. 손으로 다시 뽑아 확인."""
    rng = random.Random(5)
    l12_groups = [[(int(rng.random() < 0.5), 0) for _ in range(6)] for _ in range(2)]
    l3_groups = [[(int(rng.random() < 0.5), 0) for _ in range(8)]]
    r = manipulation_check(sample(l12_groups, 12), sample(l3_groups, 8), n_boot=300)
    blocks = stratified_bootstrap_indices([6, 6, 8], n_boot=300, seed=0)
    values = []
    for b in range(300):
        cd12 = sum(l12_groups[g][int(i)][0] for g in range(2) for i in blocks[g][b])
        cd3 = sum(l3_groups[0][int(i)][0] for i in blocks[2][b])
        values.append(Fraction(cd3, 8) - Fraction(cd12, 12))
    expected = [percentile_exact(values, q) for q in (Fraction(1, 40), Fraction(39, 40))]
    assert [Fraction(x["exact"]) for x in r["cd"]["interval"]] == expected


def test_rq_l3_0_reports_all_and_each_family(result: dict[str, Any]) -> None:
    rq0 = result["rq"]["RQ_L3_0"]
    assert set(rq0) == {"all", "bad_deployment", "dependency_failure", "traffic_spike"}
    for v in rq0.values():
        assert v["verdict"] in {"confirmed", "not_confirmed", "undecidable"}
        assert set(v) == {"cd", "gap", "verdict"}
        assert set(v["cd"]) == {"l12", "l3", "delta", "interval", "holds"}


def test_rq_l3_0_point_estimates_by_hand(result: dict[str, Any], inputs: Any) -> None:
    from evidencegate.eval_data import repeat_outcomes

    reps = repeat_outcomes(
        inputs["records"], skill_id=S0, split="oracle", metric=CD, repeats=S0_ORACLE_REPEATS
    )
    l12 = mean_fraction(list(majority(reps).values()))
    l3 = mean_fraction(list(majority(s0_l3(inputs["l3"], CD)).values()))
    got = result["rq"]["RQ_L3_0"]["all"]["cd"]
    assert (got["l12"]["exact"], got["l3"]["exact"]) == (str(l12), str(l3))
    assert got["delta"]["exact"] == str(l3 - l12)


# ---------------------------------------------------------------------------
# RQ-L3-1~3
# ---------------------------------------------------------------------------


def rate(k: int, n: int) -> Rate:
    return Rate(k, n, 0)


def test_rq_l3_1_categories() -> None:
    l12 = {g: rate(0, 4) for g in ("C1", "C2", "C4")}
    assert rq_l3_1(l12, {g: rate(1, 4) for g in ("C1", "C2", "C4")}).verdict == "supported"
    some = {"C1": rate(1, 4), "C2": rate(0, 4), "C4": rate(0, 4)}
    assert rq_l3_1(l12, some).verdict == "partially_supported"
    assert rq_l3_1(l12, l12).verdict == "not_supported"  # 동률은 성립하지 않음(§8-28)
    empty = {"C1": rate(1, 4), "C2": rate(1, 4), "C4": rate(0, 0)}
    v = rq_l3_1(l12, empty)
    assert v.verdict == "undecidable"
    assert [c.holds for c in v.comparisons] == [True, True, None]


def test_rq_l3_2_preserved_or_changed() -> None:
    l12 = {"RQ2_first": {CD: {"verdict": "not_supported"}}}
    l3 = {"RQ2_first": {CD: {"verdict": "partially_supported"}}}
    out = rq_l3_2(l12, l3, sentences=("RQ2_first",), metrics=(CD,))
    assert out["RQ2_first"][CD]["status"] == "changed"
    same = rq_l3_2(l12, l12, sentences=("RQ2_first",), metrics=(CD,))
    assert same["RQ2_first"][CD]["status"] == "preserved"


def test_rq_l3_3_counts_same_labels_and_lists_changes() -> None:
    l12 = {"a": "positive", "b": "uncertain", "c": "neutral", "d": "mixed"}
    l3 = {"a": "positive", "b": "uncertain", "c": "negative", "d": "uncertain"}
    out = rq_l3_3(l12, l3)
    assert (out["same"], out["n"], out["either_uncertain"]) == (2, 4, 2)
    assert out["rate"]["exact"] == "1/2"
    assert out["changed"] == [
        {"name": "c", "l12": "neutral", "l3": "negative"},
        {"name": "d", "l12": "mixed", "l3": "uncertain"},
    ]


def test_rq_section_of_the_result(result: dict[str, Any], inputs: Any) -> None:
    rq = result["rq"]
    t005 = json.loads(inputs["t005"])
    assert set(rq["RQ_L3_1"]) == {CD, ES} and rq["primary_metric"] == CD
    assert [c["name"] for c in rq["RQ_L3_1"][CD]["comparisons"]] == [
        "harmful C1: L3 > L1/L2",
        "harmful C2: L3 > L1/L2",
        "harmful C4: L3 > L1/L2",
    ]
    for sentence in ("RQ2_first", "RQ2_second", "RQ3_first", "RQ3_trend"):
        for m in (CD, ES):
            v = rq["RQ_L3_2"][sentence][m]
            assert v["l12"] == t005["rq"][sentence][m]["verdict"]
            assert v["status"] == ("preserved" if v["l12"] == v["l3"] else "changed")
    assert set(rq["RQ_L3_3"]) == {"proposer", "control"}
    assert rq["RQ_L3_3"]["proposer"][CD]["n"] == 4


def test_the_same_inputs_give_the_same_bytes(inputs: Any, result: dict[str, Any]) -> None:
    again = run_l3_analysis(
        inputs["records"],
        inputs["l3"],
        inputs["noise"],
        POOL,
        ticket005=inputs["t005"],
        run_info={"analysis_commit": "def5678", "plan_sha256": "e" * 64},
    )
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)


# ---------------------------------------------------------------------------
# Markdown과 명령
# ---------------------------------------------------------------------------


def test_markdown_has_korean_verdicts_and_required_notes(result: dict[str, Any]) -> None:
    md = render_markdown_l3(result)
    for heading in (
        "# Ticket 006 L3 분석 결과",
        "## RQ-L3-0 조작 확인",
        "## τ_cell",
        "## RQ-L3-1",
        "## 비교 표",
        "## Local Harm",
        "## RQ-L3-2",
        "## RQ-L3-3",
        "## 후보별 L3 Oracle",
        "## 해석 주의",
        "## 한계",
    ):
        assert heading in md, heading
    for phrase in (
        "칸 정의가 다르다",
        "Oracle 적합성(F)은 L3에 적용하지 않는다",
        "traffic_spike L3는 이동 없는 대조 칸",
        "검정 없이 기술적으로",
        "ES 바닥 효과",
        "게이트 결정은 Ticket 005",
    ):
        assert phrase in md, phrase
    assert "true effect" not in md


def write_inputs(tmp_path: Path, inputs: Any) -> dict[str, Path]:
    from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, TOOL_SPEC_ID
    from evidencegate.store import EpisodeStore

    runs = tmp_path / "runs"
    by_store: dict[tuple[str, str], list[Any]] = {}
    split_of = {c: s for s in ("dev_a", "dev_b", "oracle") for c in split_cases(s)}
    for r in inputs["records"]:
        case = (r.key.family, r.key.level, r.key.incident_seed)
        by_store.setdefault((split_of[case], r.key.skill_id), []).append(r)
    for r in inputs["l3"]:
        by_store.setdefault(("l3_oracle", r.key.skill_id), []).append(r)
    for (split, skill), recs in by_store.items():
        store = EpisodeStore.for_run(
            runs,
            suite=split,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id=skill,
        )
        for r in recs:
            store.append(r)
    noise = tmp_path / "noise.jsonl"
    nstore = EpisodeStore(noise)
    for r in inputs["noise"]:
        nstore.append(r)
    t005 = tmp_path / "t005" / "analysis.json"
    t005.parent.mkdir()
    t005.write_bytes(inputs["t005"])
    (t005.parent / "analysis.json.sha256").write_text(
        f"{hashlib.sha256(inputs['t005']).hexdigest()}  analysis.json\n"
    )
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "s0_skill_id": POOL.s0_skill_id,
                "candidates": [c.__dict__ for c in POOL.candidates],
            }
        )
    )
    return {"runs": runs, "noise": noise, "t005": t005, "pool": pool}


def cli_args(paths: dict[str, Path], out: Path) -> list[str]:
    return [
        "--runs-dir",
        str(paths["runs"]),
        "--noise",
        str(paths["noise"]),
        "--ticket005",
        str(paths["t005"]),
        "--pool-spec",
        str(paths["pool"]),
        "--out-dir",
        str(out),
        "--analysis-commit",
        "def5678",
        "--plan-sha256",
        "e" * 64,
    ]


def test_main_writes_json_hash_and_markdown_once(
    tmp_path: Path, inputs: Any, result: dict[str, Any]
) -> None:
    paths = write_inputs(tmp_path, inputs)
    out = tmp_path / "results"
    assert main(cli_args(paths, out)) == 0
    data = (out / "analysis.json").read_bytes()
    assert json.loads(data) == json.loads(json.dumps(result))
    assert (out / "analysis.json.sha256").read_text().split()[0] == hashlib.sha256(data).hexdigest()
    assert (out / "analysis.md").read_text().startswith("# Ticket 006 L3 분석 결과")
    assert main(cli_args(paths, out)) == 1  # 한 번만


def test_main_refuses_a_ticket_005_result_that_does_not_match_its_hash(
    tmp_path: Path, inputs: Any
) -> None:
    paths = write_inputs(tmp_path, inputs)
    paths["t005"].write_bytes(paths["t005"].read_bytes() + b"\n")
    assert main(cli_args(paths, tmp_path / "results")) == 1
    assert not (tmp_path / "results" / "analysis.json").exists()


def test_main_checks_the_registered_ticket_005_hash_by_default(tmp_path: Path) -> None:
    """--pool-spec 없이(실제 실행) 쓰면 Ticket 005 결과 sha256이 고정값(51d51c9b…)이어야 한다."""
    from evidencegate.eval_constants import TICKET005_ANALYSIS_SHA256

    assert TICKET005_ANALYSIS_SHA256.startswith("51d51c9b")
    real = REPO_ROOT / "results" / "ticket-005" / "analysis.json"
    assert hashlib.sha256(real.read_bytes()).hexdigest() == TICKET005_ANALYSIS_SHA256
    fake = tmp_path / "analysis.json"
    shutil.copy(real, fake)
    fake.write_bytes(fake.read_bytes() + b" ")
    (tmp_path / "analysis.json.sha256").write_text(
        f"{hashlib.sha256(fake.read_bytes()).hexdigest()}  analysis.json\n"
    )
    code = main(
        [
            "--runs-dir",
            str(tmp_path / "runs"),
            "--noise",
            str(tmp_path / "n.jsonl"),
            "--ticket005",
            str(fake),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1


def test_cells_constants() -> None:
    assert L3_CELLS == (
        ("bad_deployment", "L3"),
        ("dependency_failure", "L3"),
        ("traffic_spike", "L3"),
    )
    assert tuple((c[0], "L1+L2") for c in L3_CELLS) == FAMILY_CELLS
    assert len(CELLS) == 6


def estimate_from_json(o: dict[str, Any], tau: dict[str, Any], metric: str) -> OracleEstimate:
    def fr(x: dict[str, Any]) -> Fraction:
        return Fraction(x["exact"])

    cells = {tuple(k.split("/")): fr(v) for k, v in o["cells"].items()}
    return OracleEstimate(
        metric,
        fr(o["delta"]),
        (fr(o["interval"][0]), fr(o["interval"][1])),
        cells,
        {tuple(k.split("/")): fr(v) for k, v in tau.items()},  # type: ignore[misc]
    )


def test_budget_curve_and_sensitivity_use_ticket_005_decisions_with_l3_oracle(
    result: dict[str, Any], inputs: Any
) -> None:
    from evidencegate.budget_curve import (
        BudgetCandidate,
        budget_decisions,
        budget_points,
        sensitivity_decisions,
        sensitivity_rates,
    )
    from evidencegate.eval_data import paired_gate

    records, noise = inputs["records"], inputs["noise"]
    deltas = {
        (m, n): null_diff_quantile_exact(noise, m, n) for m in (CD, ES) for n in (24, 48, 96, 204)
    }
    l12, l3 = [], []
    for spec in POOL.candidates:
        if spec.group != "proposer":
            continue
        dev = {
            s: {
                m: paired_gate(
                    records, candidate_skill_id=spec.skill_id, s0_skill_id=S0, split=s, metric=m
                )
                for m in (CD, ES)
            }
            for s in ("dev_a", "dev_b")
        }
        est3 = {
            m: estimate_from_json(
                result["candidates"][spec.name]["oracle_l3"][m], result["tau"][m]["L3"], m
            )
            for m in (CD, ES)
        }
        l12.append(BudgetCandidate(spec.name, dev["dev_a"], dev["dev_b"], est3))
        l3.append(BudgetCandidate(spec.name, dev["dev_a"], dev["dev_b"], est3))
    decisions = budget_decisions(l12, deltas, ns=(24,), n_resample=N_RESAMPLE)
    point = budget_points(decisions[24], l3)[("C2", CD)]
    got = result["budget_curve_l3"]["proposer"]["24"][f"C2/{CD}"]["metrics"]["useful"]
    assert got["undefined"] == point.metrics["useful"].undefined
    mean = point.metrics["useful"].mean
    assert (got["mean"] and got["mean"]["exact"]) == (str(mean) if mean is not None else None)
    sens = sensitivity_rates(sensitivity_decisions(l12, n_resample=N_RESAMPLE), l3, CD)
    got_s = result["sensitivity_l3"]["proposer"][CD]["harmful"]
    assert got_s["undefined"] == sens["harmful"].undefined
