"""Ticket 012 §9의 4: 1단계 손잡이 탐침(`evidencegate.t012.stage1`). 가짜 정책만 쓴다.

판정 숫자는 기준 파일(m, 필드 수, P0)과 `results/ticket-012/stage0.json`(k)에서 읽는다. 비교 기준은
S0-prime 1회째 기록(split t012_stage0)이고 열기만 한다.
"""

import copy
import hashlib
import io
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from evidencegate.bench_runner import BenchStopped, run_blocks
from evidencegate.ids import content_id
from evidencegate.policy import PolicyTrace, StopReason, TurnRecord, call_tool
from evidencegate.run_plan import Block
from evidencegate.runner import EpisodeRecord
from evidencegate.skills import Skill
from evidencegate.store import EpisodeStore
from evidencegate.t009.splits import split_cases
from evidencegate.t010 import probe as p010
from evidencegate.t012 import stage1 as s1
from opsworld.agent_env import TOOL_SPECS, AgentEnv

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = json.loads((ROOT / "plans" / "ticket-012-criteria.json").read_text())
STAGE0 = json.loads((ROOT / "results" / "ticket-012" / "stage0.json").read_text())
DEV_A = split_cases("t009_dev_a")
S0P = CRITERIA["skills"]["S0_prime"]["skill_id"]
# tests/test_preflight.py의 REFERENCE와 같은 값(config 7c66ee5bf13c)
REFERENCE: dict[str, Any] = {
    "type": "ollama",
    "model": "gpt-oss:20b",
    "model_digest": "f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe",
    "quantization": "MXFP4",
    "ollama_version": "0.40.3",
    "instructions_sha256": "332b776ce67826fba8829790fc46090df243c2f9f38602e1c8c72faea4bfaed3",
    "options": {"temperature": 0.7, "num_ctx": 8192},
    "think": "low",
    "stop": {"max_tool_calls": 8, "max_consecutive_text_turns": 2},
    "format_error_policy": {"feedback": "error_as_tool_result", "max_consecutive_format_errors": 3},
    "seed_rule": "derive_llm_seed(family, incident_seed, repeat): sha256, first 8 bytes mod 2^31",
    "machine": {"hw_model": "Mac15,3", "chip": "Apple M3"},
    "parallel": {"client_concurrency": 2, "server_num_parallel": 2},
}


# 고정 값 ---------------------------------------------------------------------------------------


def test_inputs_are_the_frozen_files() -> None:
    assert Path("results/ticket-012/stage0.json") == s1.STAGE0_PATH
    assert s1.STAGE0_SHA256 == "9bbe3ba4ccbe572065452865ac952b40fc25b196b3da854cb6ae5036c6f34739"
    assert s1.PROBE_SPLIT == "t012_stage1"
    assert Path("results/ticket-012/stage1.json") == s1.RESULT_PATH
    assert s1.load_stage0(ROOT) == STAGE0


def test_the_rule_is_read_from_the_criteria_and_stage0() -> None:
    rule = s1.rule(CRITERIA, STAGE0)
    assert rule == {
        "k": 21,
        "m": CRITERIA["probe"]["m"],
        "min_fields": CRITERIA["probe"]["min_fields"],
        "p0_min": CRITERIA["probe"]["p0"]["min_changed"],
        "p0_max_tool_calls": CRITERIA["probe"]["p0"]["max_tool_calls"],
        "episodes": CRITERIA["probe"]["p0"]["episodes"],
    }
    other = copy.deepcopy(STAGE0)
    other["attempts"][0]["k"]["k"] = 15
    assert s1.rule(CRITERIA, other)["k"] == 15


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"verdict": "close"}, "pass"),
        ({"skill_for_stage1": "S0_prime_r1"}, "S0_prime"),
    ],
)
def test_the_rule_refuses_a_stage0_result_that_did_not_pass_with_s0_prime(
    change: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        s1.rule(CRITERIA, STAGE0 | change)


def test_a_changed_stage0_file_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / s1.STAGE0_PATH).parent.mkdir(parents=True)
    (root / s1.STAGE0_PATH).write_bytes((ROOT / s1.STAGE0_PATH).read_bytes() + b" ")
    with pytest.raises(ValueError, match="sha256"):
        s1.load_stage0(root)


# 변형 ------------------------------------------------------------------------------------------


def test_p1_to_p6_are_the_010_variants_and_p0_limits_to_the_criteria_value() -> None:
    variants = s1.variants(CRITERIA)
    assert [v.name for v in variants] == [f"P{i}" for i in range(7)]
    assert variants[1:] == p010.VARIANTS[1:]
    p0 = variants[0]
    assert (p0.field, p0.judged, p0.harness) == ("control", False, None)
    assert p0.max_tool_calls == CRITERIA["probe"]["p0"]["max_tool_calls"] == 1


def test_variant_configs_differ_only_in_their_own_field() -> None:
    check = s1.check_variants(REFERENCE, CRITERIA)
    assert check.ok, check.detail
    ids = {content_id(p010.offline_policy(v, REFERENCE).config) for v in s1.variants(CRITERIA)}
    assert len(ids) == 7 and "7c66ee5bf13c" not in ids


def test_a_variant_that_changes_another_field_is_refused() -> None:
    from dataclasses import replace

    from evidencegate.harness import HarnessText

    bad = [
        replace(s1.variants(CRITERIA)[0], harness=HarnessText(observation_renderer="json_compact"))
    ]
    check = s1.check_variants(REFERENCE, CRITERIA, bad)
    assert not check.ok and "P0 changes" in check.detail


def test_p3_and_p4_pass_the_012_description_check() -> None:
    for v in s1.variants(CRITERIA)[3:5]:
        assert v.harness is not None
        ((tool, text),) = v.harness.tool_descriptions
        assert s1.description_problems(tool, text, CRITERIA) == [], v.name


@pytest.mark.parametrize(
    ("tool", "text", "kind"),
    [
        ("get_traffic_metrics", "Call this first to see demand.", "order word"),
        ("get_traffic_metrics", "Useful afterwards, once latency is known.", "order word"),
        ("get_traffic_metrics", "Compare with get_service_metrics.", "other tool"),
        ("get_dependency_health", "Check this before get_recent_deployments.", "order word"),
    ],
)
def test_the_description_check_refuses_order_words_and_other_tool_names(
    tool: str, text: str, kind: str
) -> None:
    problems = s1.description_problems(tool, text, CRITERIA)
    assert any(kind in p for p in problems), problems


def test_the_current_tool_descriptions_pass_the_description_check() -> None:
    for spec in TOOL_SPECS:
        assert s1.description_problems(spec.name, spec.description, CRITERIA) == [], spec.name


def test_blocks_are_s0_prime_on_dev_a_in_the_stage1_split() -> None:
    blocks = s1.probe_blocks(CRITERIA)
    assert [b.name for b in blocks] == [f"P{i}" for i in range(7)]
    for b in blocks:
        assert (b.split, b.skill_id, b.repeat, list(b.cases)) == ("t012_stage1", S0P, 0, DEV_A)


# 판정 ------------------------------------------------------------------------------------------

RULE = s1.rule(CRITERIA, STAGE0)


def counts(p0: int, **others: int) -> dict[str, int]:
    return {"P0": p0, **{f"P{i}": 0 for i in range(1, 7)}, **others}


@pytest.mark.parametrize(
    ("different", "verdict", "moved"),
    [
        (counts(49, P1=60, P3=60, P5=60), "no_decision", []),
        (counts(50, P1=21, P3=21, P5=21), "proceed", ["P1", "P3", "P5"]),
        (counts(60, P1=21, P3=21, P5=20), "reconsider", ["P1", "P3"]),
        (counts(60, P1=21, P2=21, P3=21, P4=21, P5=21, P6=21), "proceed",
         [f"P{i}" for i in range(1, 7)]),
        (counts(60), "reconsider", []),
    ],
)  # fmt: skip
def test_judge_uses_k_from_stage0(
    different: dict[str, int], verdict: str, moved: list[str]
) -> None:
    decision = s1.judge(different, RULE, s1.variants(CRITERIA))
    assert decision.verdict == verdict and list(decision.moved) == moved


def test_judge_needs_the_field_count_from_the_criteria() -> None:
    strict = RULE | {"min_fields": 4}
    decision = s1.judge(counts(60, P1=21, P3=21, P5=21), strict, s1.variants(CRITERIA))
    assert decision.verdict == "reconsider"


# 실행(가짜 정책) -------------------------------------------------------------------------------


class Fake:
    policy_id = "ollama:gpt-oss:20b"

    def __init__(self, name: str, behave: Callable[[AgentEnv], PolicyTrace]) -> None:
        self.config: dict[str, Any] = {"fake": name}
        self.behave = behave

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        return self.behave(env)


def act(env: AgentEnv) -> PolicyTrace:
    _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
    return PolicyTrace((turn,), 0, StopReason.ACTION)


def stall(env: AgentEnv) -> PolicyTrace:
    _, turn = call_tool(env, "get_service_metrics", {"service_id": env.service_id})
    return PolicyTrace((turn,), 0, StopReason.MAX_STEPS)


def garbled(env: AgentEnv) -> PolicyTrace:
    bad = TurnRecord("<unparsed>", {}, "{}", True)
    _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
    return PolicyTrace((bad, turn), 1, StopReason.ACTION)


def fakes(**behaviour: Callable[[AgentEnv], PolicyTrace]) -> dict[str, Any]:
    return {f"P{i}": Fake(f"P{i}", behaviour.get(f"P{i}", act)) for i in range(1, 7)} | {
        "P0": Fake("P0", behaviour.get("P0", stall))
    }


def baseline_store(runs: Path) -> Path:
    """S0-prime 1회째 기록 자리(split t012_stage0)에 가짜 기록 60개."""
    block = Block(0, "S0_prime", "stage0", S0P, "skills/S0-prime/SKILL.md", "t012_stage0", 0,
                  tuple(DEV_A))  # fmt: skip
    run_blocks([block], Fake("base", act), root=ROOT, runs_dir=runs, out=io.StringIO())
    return next((runs / "t012_stage0").rglob("*.jsonl"))


def baseline(runs: Path) -> dict[p010.PairKey, EpisodeRecord]:
    path = baseline_store(runs)
    return {p010.pair_key(r): r for r in EpisodeStore(path, repair=False).records()}


def run_fakes(runs: Path, policies: Mapping[str, Any], **kw: Any) -> Any:
    base = baseline(runs)
    return s1.run_stage1(policies, root=ROOT, runs_dir=runs, out=io.StringIO(),
                         baseline=base, criteria=CRITERIA, rule=RULE, **kw)  # fmt: skip


def stores(runs: Path) -> dict[str, int]:
    return {
        p.parent.parent.name: len(p.read_text().splitlines())
        for p in sorted((runs / "t012_stage1").glob("*/*/*.jsonl"))
    }


def test_run_writes_one_stage1_store_per_variant(tmp_path: Path) -> None:
    result = run_fakes(tmp_path / "runs", fakes())
    assert result.blocks_run == 7 and result.stopped_by is None
    assert sorted(stores(tmp_path / "runs").values()) == [60] * 7


def test_run_stops_after_p0_when_p0_does_not_move(tmp_path: Path) -> None:
    with pytest.raises(BenchStopped, match="P0"):
        run_fakes(tmp_path / "runs", fakes(P0=act))
    assert len(stores(tmp_path / "runs")) == 1


def test_candidate_blocks_record_format_errors_and_stop_after_three(tmp_path: Path) -> None:
    result = run_fakes(tmp_path / "a", fakes(P1=garbled, P2=garbled))
    assert result.blocks_run == 7
    with pytest.raises(BenchStopped, match="format error"):
        run_fakes(tmp_path / "b", fakes(P1=garbled, P2=garbled, P3=garbled))


def test_stop_file_stops_at_a_variant_boundary(tmp_path: Path) -> None:
    stop = tmp_path / "STOP"
    stop.write_text("")
    result = run_fakes(tmp_path / "runs", fakes(), stop_file=stop)
    assert result.blocks_run == 0 and result.stopped_by == "stop_file"


def test_the_stage0_records_are_only_read(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    base = baseline(runs)
    path = next((runs / "t012_stage0").rglob("*.jsonl"))
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns)
    s1.run_stage1(fakes(), root=ROOT, runs_dir=runs, out=io.StringIO(), baseline=base,
                  criteria=CRITERIA, rule=RULE)  # fmt: skip
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns) == before
    assert sorted((runs / "t012_stage0").rglob("*.jsonl")) == [path]


# 보고 ------------------------------------------------------------------------------------------


def records_by_variant(runs: Path) -> dict[str, list[EpisodeRecord]]:
    out = {}
    for path in sorted((runs / "t012_stage1").glob("*/*/*.jsonl")):
        recs = EpisodeStore(path, repair=False).records()
        out[recs[0].policy_config["fake"]] = recs
    return out


def test_result_has_the_table_the_rule_and_the_decision(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    base = baseline(runs)
    s1.run_stage1(fakes(), root=ROOT, runs_dir=runs, out=io.StringIO(), baseline=base,
                  criteria=CRITERIA, rule=RULE)  # fmt: skip
    result = s1.build_result(base, records_by_variant(runs), CRITERIA, RULE)
    rows = {r["name"]: r for r in result["variants"]}
    assert rows["P0"]["different_disposition"] == 60
    assert all(rows[f"P{i}"]["different_disposition"] == 0 for i in range(1, 7))
    assert result["decision"]["verdict"] == "reconsider"
    assert result["rule"] == RULE
    assert result["stage0_sha256"] == s1.STAGE0_SHA256
    assert result["criteria_sha256"] == CRITERIA_SHA
    descriptions = result["tool_description_candidates"]
    assert [d["variant"] for d in descriptions] == ["P3", "P4"]
    for d in descriptions:
        assert d["problems"] == [] and d["text"]
    for key in ("different_tool_set", "traffic_episodes", "mean_tool_calls", "stop_reasons",
                "format_error_episodes"):  # fmt: skip
        assert key in rows["P1"]
    text = s1.result_bytes(result)
    assert s1.result_bytes(s1.build_result(base, records_by_variant(runs), CRITERIA, RULE)) == text


CRITERIA_SHA = "63a8f73ee8bac878c51007105036aa1bdeaed5d1f8a1678ce3893611efc5fe94"


def test_write_result_once_and_refuse_a_different_file(tmp_path: Path) -> None:
    result = {"a": 1}
    root = tmp_path / "root"
    assert s1.write_result(root, result) is True
    assert s1.write_result(root, result) is True
    (root / s1.RESULT_PATH).write_text("{}\n")
    assert s1.write_result(root, result) is False


def test_report_without_every_variant_gives_no_decision(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    base = baseline(runs)
    s1.run_stage1(fakes(), root=ROOT, runs_dir=runs, out=io.StringIO(), baseline=base,
                  criteria=CRITERIA, rule=RULE, max_blocks=3)  # fmt: skip
    recs = records_by_variant(runs)
    assert len(recs) == 3
    with pytest.raises(ValueError, match="incomplete"):
        s1.build_result(base, recs, CRITERIA, RULE)


# 실행 전 점검 ----------------------------------------------------------------------------------


def test_preflight_adds_the_stage0_result_variants_and_baseline(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    baseline_store(runs)
    checks = {
        c.name: c
        for c in s1.stage1_extra_checks(
            root=ROOT, runs_dir=runs, base_config={"fake": "base"}, policy_id=Fake.policy_id,
            criteria=CRITERIA,
        )
    }  # fmt: skip
    assert list(checks) == ["stage0_result", "variants", "baseline", "stage1_paths"]
    assert checks["stage0_result"].ok and checks["baseline"].ok and checks["stage1_paths"].ok
    assert not checks["variants"].ok  # 가짜 설정에는 엔진 키가 없다(진짜 설정에서 확인)
    real = s1.check_variants(REFERENCE, CRITERIA)
    assert real.ok


def test_preflight_refuses_a_baseline_with_missing_records(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    path = baseline_store(runs)
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    check = s1.check_baseline(runs, {"fake": "base"}, Fake.policy_id, CRITERIA)
    assert not check.ok and "59" in check.detail
