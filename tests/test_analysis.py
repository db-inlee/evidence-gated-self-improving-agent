"""분석 전체 실행 (Ticket 005 §1, 순서 6). 합성 레코드로 처음부터 끝까지 돌린다."""

import hashlib
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate.analysis import (
    CandidateSpec,
    PoolSpec,
    check_preregistration,
    load_pool,
    load_records,
    main,
    render_markdown,
    run_analysis,
)
from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_ORACLE_REPEATS, TOOL_SPEC_ID
from evidencegate.eval_data import paired_gate, paired_oracle, split_cases
from evidencegate.exact import null_diff_quantile_exact
from evidencegate.gates import decide_gates
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
CD = "correct_disposition"
ES = "evidence_safe"
S0 = "s0skill00000"

POOL = PoolSpec(
    s0_skill_id=S0,
    candidates=(
        CandidateSpec("cand-000", "p0skill00000", "proposer", None, {"cd": "up", "es": "up"}),
        CandidateSpec("cand-001", "p1skill00000", "proposer", None, {"cd": "down", "es": "none"}),
        CandidateSpec("cand-002", "p2skill00000", "proposer", None, {"cd": "none", "es": "up"}),
        # CD는 올리고 ES는 떨어뜨린다(RQ1에서 CD 채택과 ES 채택이 갈리게)
        CandidateSpec("cand-003", "p3skill00000", "proposer", None, {"cd": "up", "es": "down"}),
        CandidateSpec("pos-x", "c0skill00000", "control", "positive", None),
        CandidateSpec("neg-x", "c1skill00000", "control", "negative", None),
    ),
)
# 후보마다 S0 대비 결과를 바꾸는 확률(뒤집힘): (CD를 1로, CD를 0으로)
EFFECTS = {
    "p0skill00000": (0.35, 0.05),
    "p1skill00000": (0.05, 0.35),
    "p2skill00000": (0.1, 0.1),
    "p3skill00000": (0.35, 0.0),
    "c0skill00000": (0.4, 0.0),
    "c1skill00000": (0.0, 0.4),
}


def record(
    skill: str, case: tuple[str, str, int], repeat: int, cd: bool, es: bool, **kw: Any
) -> EpisodeRecord:
    family, level, seed = case
    stop = kw.get("stop", StopReason.ACTION)
    return EpisodeRecord(
        key=EpisodeKey(POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, skill, family, seed, repeat, level),
        policy_config={},
        llm_seed=0,
        correct_disposition=cd,
        evidence_safe=es,
        action="observe_only" if stop is StopReason.ACTION else None,
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), kw.get("format_errors", 0), stop),
        started_at="2026-10-22T00:00:00+00:00",
        duration_s=1.0,
    )


def world(seed: int = 0) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    """S0 기준값, 후보 결과(효과 확률), 노이즈 기준(보정 120 케이스 x 반복 5)."""
    rng = random.Random(seed)
    records: list[EpisodeRecord] = []
    base: dict[tuple[str, str, int], tuple[bool, bool]] = {}
    for split in ("dev_a", "dev_b", "oracle"):
        for case in split_cases(split):
            base[case] = (rng.random() < 0.6, rng.random() < 0.35)
            reps = S0_ORACLE_REPEATS if split == "oracle" else (0,)
            for r in reps:
                cd, es = base[case]
                if r > 0 and rng.random() < 0.05:
                    cd = not cd
                if r > 0 and rng.random() < 0.12:  # ES 잡음: τ_ES가 τ_CD와 달라지게
                    es = not es
                records.append(record(S0, case, r, cd, es and cd))
    for spec in POOL.candidates:
        up, down = EFFECTS[spec.skill_id]
        for split in ("dev_a", "dev_b", "oracle"):
            for case in split_cases(split):
                cd, es = base[case]
                if not cd and rng.random() < up:
                    cd, es = True, rng.random() < 0.7
                elif cd and rng.random() < down:
                    cd, es = False, False
                if spec.skill_id == "p3skill00000":
                    es = False
                records.append(record(spec.skill_id, case, 0, cd, es and cd))
    noise = []
    families = ("bad_deployment", "dependency_failure", "traffic_spike")
    for i in range(120):
        case = (families[i % 3], ("L1", "L2")[(i // 3) % 2], 900_000 + i)
        cd0 = rng.random() < 0.6
        for r in range(5):
            noise.append(record(S0, case, r, cd0 != (rng.random() < 0.08), False))
    return records, noise


RUN_INFO = {"analysis_commit": "abc1234", "plan_sha256": None}


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    records, noise = world()
    return run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=5)


# ---------------------------------------------------------------------------
# 기록과 기본 값
# ---------------------------------------------------------------------------


def test_meta_records_versions_hashes_and_settings(result: dict[str, Any]) -> None:
    meta = result["meta"]
    assert meta["analysis_commit"] == "abc1234"
    assert meta["plan_sha256"] is None
    assert (
        meta["preregistration_sha256"]
        == "7611e83ddc2204eb12f457b2caa2c93fda5bdcb0c0dd42757cc203d64818a44d"
    )
    assert meta["section8_version"].startswith("2026-10-21")
    assert meta["python"] and meta["numpy"] == "2.3.4"
    assert (meta["n_resample"], meta["n_boot"], meta["seed"]) == (5, 2000, 0)
    assert meta["run_conditions"] == {
        "policy_id": POLICY_ID,
        "config_id": CONFIG_ID,
        "tool_spec_id": TOOL_SPEC_ID,
    }


def test_delta_is_exact_from_the_noise_records(result: dict[str, Any]) -> None:
    _, noise = world()
    for n in (210, 24, 48, 96, 204):
        expected = null_diff_quantile_exact(noise, CD, n)
        assert result["delta"][CD][str(n)]["exact"] == str(expected)
    assert result["delta"][ES]["210"]["exact"] == "0"


def test_episode_counts(result: dict[str, Any]) -> None:
    episodes = result["episodes"]
    assert episodes["total"] == 60 + 210 + 300 * 5 + 6 * 570
    assert episodes["no_disposition"] == 0
    assert episodes["format_error_episodes"] == {}


def test_format_errors_and_no_disposition_are_counted() -> None:
    records, noise = world()
    case = split_cases("dev_b")[0]
    records = [
        r
        for r in records
        if not (
            r.key.skill_id == "p0skill00000"
            and r.key.incident_seed == case[2]
            and r.key.family == case[0]
            and r.key.level == case[1]
        )
    ]
    records.append(
        record("p0skill00000", case, 0, False, False, format_errors=3, stop=StopReason.ERROR)
    )
    out = run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=2)
    assert out["episodes"]["format_error_episodes"] == {"cand-000": 1}
    assert out["episodes"]["no_disposition"] == 1


# ---------------------------------------------------------------------------
# 후보별 값이 모듈 함수와 같다
# ---------------------------------------------------------------------------


def test_candidate_gates_match_decide_gates(result: dict[str, Any]) -> None:
    records, noise = world()
    delta = null_diff_quantile_exact(noise, CD, 210)
    dev_a = paired_gate(
        records, candidate_skill_id="p0skill00000", s0_skill_id=S0, split="dev_a", metric=CD
    )
    dev_b = paired_gate(
        records, candidate_skill_id="p0skill00000", s0_skill_id=S0, split="dev_b", metric=CD
    )
    expected = {g.gate: g.accepted for g in decide_gates(dev_a, dev_b, delta)}
    got = {g: v["accepted"] for g, v in result["candidates"]["cand-000"]["gates"][CD].items()}
    assert got == expected


def test_candidate_oracle_delta_matches_the_paired_table(result: dict[str, Any]) -> None:
    records, _ = world()
    t = paired_oracle(records, candidate_skill_id="p0skill00000", s0_skill_id=S0, metric=CD)
    expected = Fraction(sum(t.deltas), len(t.deltas))
    assert result["candidates"]["cand-000"]["oracle"][CD]["delta"]["exact"] == str(expected)


def test_controls_have_a_label_check_and_proposers_an_effect_match(result: dict[str, Any]) -> None:
    pos = result["candidates"]["pos-x"]["oracle"][CD]
    assert pos["control_check"] in {"as_expected", "unexpected", "uncertain"}
    assert "effect_match" not in pos
    prop = result["candidates"]["cand-000"]["oracle"][CD]
    assert prop["effect_match"] in {True, False, None}
    assert "control_check" not in prop


# ---------------------------------------------------------------------------
# 표, 지표, RQ
# ---------------------------------------------------------------------------


def test_results_table_rows_follow_7_8_e(result: dict[str, Any]) -> None:
    rows = [r["row"] for r in result["results_table"]]
    assert rows[:5] == [
        "Positive controls",
        "Negative controls",
        "Mixed controls",
        "Neutral controls",
        "Proposer (all)",
    ]
    assert rows[5:] == [
        f"Proposer {label}" for label in ("positive", "negative", "mixed", "neutral", "uncertain")
    ]
    pos = result["results_table"][0]
    assert pos["candidates"] == 1
    assert set(pos["cells"]) == {f"{g}/{m}" for g in ("C1", "C2", "C3", "C4") for m in (CD, ES)}
    mixed = result["results_table"][2]
    assert mixed["candidates"] == 0 and mixed["cells"]["C2/" + CD] == "0 / 0"


def test_metrics_are_reported_per_group_and_never_pooled(result: dict[str, Any]) -> None:
    assert set(result["metrics"]) == {"proposer", "control"}
    c2 = result["metrics"]["proposer"][CD]["C2"]
    assert set(c2) == {
        "accepted",
        "harmful",
        "unsupported",
        "useful",
        "gain",
        "false_rejection",
        "local_harm",
    }
    assert set(c2["harmful"]) == {"successes", "n", "excluded", "value", "interval"}
    assert set(c2["gain"]) == {"n", "mean"}
    assert set(result["rq1_cross"]["proposer"]) == {"C1", "C2", "C3", "C4"}


def test_rq_verdicts_are_present_for_cd_and_es(result: dict[str, Any]) -> None:
    rq = result["rq"]
    labels = {"supported", "partially_supported", "not_supported", "undecidable"}
    assert rq["RQ1"]["verdict"] in labels
    for metric in (CD, ES):
        assert rq["RQ2_first"][metric]["verdict"] in labels
        assert rq["RQ2_second"][metric]["verdict"] in {
            "grows",
            "partly_grows",
            "does_not_grow",
            "undecidable",
        }
        assert rq["RQ3_first"][metric]["verdict"] in labels
        trend = rq["RQ3_trend"][metric]
        assert trend["verdict"] in {
            "trend_supported",
            "partial_trend_support",
            "trend_not_supported",
            "undecidable",
        }
        assert set(trend["non_increasing"]) == {"useful", "harmful"}
    assert rq["primary_metric"] == CD


def test_budget_curve_and_sensitivity_sections(result: dict[str, Any]) -> None:
    bc = result["budget_curve"]["proposer"]
    assert set(bc) == {"24", "48", "96", "204"}
    point = bc["24"]["C4/" + CD]
    assert set(point) == {"acceptance", "acceptance_rate", "metrics"}
    assert set(point["metrics"]["useful"]) == {"mean", "low", "high", "undefined", "n_resample"}
    assert set(result["sensitivity"]["control"][CD]) == {"harmful", "local_harm"}


def test_fitness_and_tau_sections(result: dict[str, Any]) -> None:
    for metric in (CD, ES):
        assert set(result["fitness"][metric]) == {
            "fit",
            "sign_met",
            "size_met",
            "sign_hits",
            "size_hits",
        }
        assert len(result["tau"][metric]) == 6


def test_analysis_is_deterministic(result: dict[str, Any]) -> None:
    records, noise = world()
    again = run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=5)
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)


def test_missing_episode_stops_the_analysis() -> None:
    records, noise = world()
    records = [
        r
        for r in records
        if not (r.key.skill_id == "c1skill00000" and r.key.incident_seed == 300_000)
    ]
    with pytest.raises(ValueError, match="missing"):
        run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=2)


def test_mixed_run_conditions_stop_the_analysis() -> None:
    records, noise = world()
    r = records[0]
    records[0] = EpisodeRecord(
        key=EpisodeKey(
            r.key.policy_id,
            "24bf38dea859",
            r.key.tool_spec_id,
            r.key.skill_id,
            r.key.family,
            r.key.incident_seed,
            r.key.repeat,
            r.key.level,
        ),
        policy_config={},
        llm_seed=0,
        correct_disposition=True,
        evidence_safe=True,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-10-22T00:00:00+00:00",
        duration_s=1.0,
    )
    with pytest.raises(ValueError, match="mixed run conditions"):
        run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=2)


# ---------------------------------------------------------------------------
# Markdown 표와 해석 주의
# ---------------------------------------------------------------------------


def test_markdown_has_korean_verdicts_and_required_notes(result: dict[str, Any]) -> None:
    md = render_markdown(result)
    assert "## 결과표" in md
    assert any(label in md for label in ("지지", "부분 지지", "지지 안 됨", "판정 불가"))
    assert "비증가(같은 값 허용)" in md
    assert "C2는 Δ = 0 후보도 채택" in md  # 7.8 G
    assert "C2·C4 쌍은 항상 같은 결과" in md  # §8-33
    assert "δ(N)은 칸 층화 없이" in md  # 7.8 A 한계
    assert "proposer 입력의 치우침" in md  # 7.8 C 한계
    assert "주요 결론은 proposer 후보로만" in md  # 7.8 G
    assert "의도된 조건" in md  # 7.8 G
    assert "주제 분포는 기술용" in md  # 7.8 G
    assert "## Oracle 적합성" in md and ("적합" in md or "미달" in md)  # 7.8 F·G


# ---------------------------------------------------------------------------
# 사전 등록 해시, 풀, 기록 읽기, 명령
# ---------------------------------------------------------------------------


def test_check_preregistration_passes_for_the_repo() -> None:
    assert (
        check_preregistration(REPO_ROOT)
        == "7611e83ddc2204eb12f457b2caa2c93fda5bdcb0c0dd42757cc203d64818a44d"
    )


def test_check_preregistration_fails_when_the_copy_changes(tmp_path: Path) -> None:
    shutil.copytree(REPO_ROOT / "preregistration", tmp_path / "preregistration")
    copy = tmp_path / "preregistration" / "ticket-004-section-7.8.txt"
    copy.write_text(copy.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match=r"7\.8"):
        check_preregistration(tmp_path)


def test_load_pool_reads_the_frozen_pool() -> None:
    pool = load_pool(REPO_ROOT)
    assert pool.s0_skill_id == "bf65e569ae37"
    proposers = [c for c in pool.candidates if c.group == "proposer"]
    controls = [c for c in pool.candidates if c.group == "control"]
    assert len(proposers) == 12 and len(controls) == 7
    assert proposers[0].name == "cand-000" and proposers[0].expected_effect == {
        "cd": "down",
        "es": "up",
    }
    assert {c.prior_label for c in controls} == {"positive", "negative", "mixed", "neutral"}
    assert all(c.prior_label is None for c in proposers)


def write_stores(root: Path, records: list[EpisodeRecord], suite_of: dict[int, str]) -> None:
    for r in records:
        suite = suite_of[r.key.incident_seed // 100_000]
        EpisodeStore.for_run(
            root,
            suite=suite,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id=r.key.skill_id,
        ).append(r)


def test_load_records_reads_every_split_store(tmp_path: Path) -> None:
    records, _ = world()
    write_stores(tmp_path, records, {1: "dev_a", 2: "dev_b", 3: "oracle"})
    loaded = load_records(tmp_path, POOL)
    assert len(loaded) == len(records)


def test_main_writes_json_markdown_and_hash(tmp_path: Path) -> None:
    records, noise = world()
    runs = tmp_path / "runs"
    write_stores(runs, records, {1: "dev_a", 2: "dev_b", 3: "oracle"})
    noise_path = tmp_path / "noise.jsonl"
    store = EpisodeStore(noise_path)
    for r in noise:
        store.append(r)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        json.dumps(
            {"s0_skill_id": POOL.s0_skill_id, "candidates": [c.__dict__ for c in POOL.candidates]}
        )
    )
    out = tmp_path / "out"
    code = main(
        [
            "--runs-dir",
            str(runs),
            "--noise",
            str(noise_path),
            "--out-dir",
            str(out),
            "--pool-spec",
            str(pool_path),
            "--analysis-commit",
            "abc1234",
            "--n-resample",
            "2",
        ]
    )
    assert code == 0
    data = (out / "analysis.json").read_bytes()
    assert (out / "analysis.json.sha256").read_text().split()[0] == hashlib.sha256(data).hexdigest()
    assert (out / "analysis.md").read_text(encoding="utf-8").startswith("# Ticket 005 분석 결과")
    assert json.loads(data)["meta"]["n_resample"] == 2


def test_main_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "analysis.json").write_text("{}")
    assert (
        main(
            [
                "--runs-dir",
                str(tmp_path),
                "--noise",
                str(tmp_path / "n.jsonl"),
                "--out-dir",
                str(out),
            ]
        )
        == 1
    )


def test_markdown_has_candidate_budget_and_sensitivity_tables(result: dict[str, Any]) -> None:
    """티켓 §6 결과 보고서 항목 3~5의 표."""
    md = render_markdown(result)
    assert "## 후보별 표" in md and "| cand-000 |" in md and "| pos-x |" in md
    assert "## Budget Curve (proposer)" in md and "| 24 |" in md and "| 204 |" in md
    assert "## RQ2 민감도" in md
    assert "## 통제 후보 라벨 검증" in md


# ---------------------------------------------------------------------------
# 연결: 각 부품을 맞는 입력으로 잇는다
# ---------------------------------------------------------------------------


def test_rq1_compares_cd_gated_with_es_gated_es_harm(result: dict[str, Any]) -> None:
    """A = CD로 채택한 후보의 ES 해로움(rq1_cross), B = ES로 채택한 후보의 ES 해로움."""
    for comparison in result["rq"]["RQ1"]["comparisons"]:
        gate = comparison["name"]
        assert comparison["left"] == result["rq1_cross"]["proposer"][gate]["harmful"]["value"]
        assert comparison["right"] == result["metrics"]["proposer"][ES][gate]["harmful"]["value"]


def test_oracle_estimates_use_each_metrics_own_tau(result: dict[str, Any]) -> None:
    from evidencegate.eval_data import repeat_outcomes
    from evidencegate.oracle import estimate, local_harm, subset_label, tau_cells

    records, _ = world()
    assert result["tau"][CD] != result["tau"][ES]
    for spec in POOL.candidates:
        for m in (CD, ES):
            tau = tau_cells(
                repeat_outcomes(
                    records, skill_id=S0, split="oracle", metric=m, repeats=S0_ORACLE_REPEATS
                )
            )
            e = estimate(
                paired_oracle(records, candidate_skill_id=spec.skill_id, s0_skill_id=S0, metric=m),
                tau,
            )
            got = result["candidates"][spec.name]["oracle"][m]
            assert got["local_harm"] == local_harm(e)
            assert got["subset"] == subset_label(e).label


def test_metrics_never_mix_proposer_and_control_candidates(result: dict[str, Any]) -> None:
    proposers = {c.name for c in POOL.candidates if c.group == "proposer"}
    for group, names in (("proposer", proposers), ("control", {"pos-x", "neg-x"})):
        for m in (CD, ES):
            for gate in ("C1", "C2", "C3", "C4"):
                assert set(result["metrics"][group][m][gate]["accepted"]) <= names
                total = result["metrics"][group][m][gate]["false_rejection"]
                assert total["n"] + total["excluded"] <= len(names)


def test_proposer_subset_rows_use_each_metrics_labels(result: dict[str, Any]) -> None:
    rows = {r["row"]: r for r in result["results_table"]}
    for label in ("positive", "negative", "mixed", "neutral", "uncertain"):
        for m in (CD, ES):
            members = [
                n
                for n, c in result["candidates"].items()
                if c["group"] == "proposer" and c["oracle"][m]["subset"] == label
            ]
            assert rows[f"Proposer {label}"]["cells"][f"C2/{m}"].endswith(f"/ {len(members)}")


def test_effect_match_uses_the_key_of_each_metric(result: dict[str, Any]) -> None:
    for spec in POOL.candidates:
        if spec.expected_effect is None:
            continue
        for m, key in ((CD, "cd"), (ES, "es")):
            o = result["candidates"][spec.name]["oracle"][m]
            expected = None if o["effect"] is None else o["effect"] == spec.expected_effect[key]
            assert o["effect_match"] == expected


def test_run_analysis_checks_the_preregistration(tmp_path: Path) -> None:
    shutil.copytree(REPO_ROOT / "preregistration", tmp_path / "preregistration")
    copy = tmp_path / "preregistration" / "ticket-004-section-7.8.txt"
    copy.write_text(copy.read_text(encoding="utf-8") + " ", encoding="utf-8")
    records, noise = world()
    with pytest.raises(ValueError, match=r"7\.8"):
        run_analysis(records, noise, POOL, run_info=RUN_INFO, n_resample=1, root=tmp_path)
