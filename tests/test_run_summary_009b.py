"""실행 요약 009b(Ticket 011 §6의 3) 테스트.

정의는 `writeup/inputs/run_summary_009b_definition.txt`다.

가짜 기록·가짜 저장소로 정의의 읽기·세기 규칙과 거부 조건을 확인한다. 실제 기록은 마지막 두
테스트에서만 읽는다(없으면 건너뜀).
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evidencegate import run_summary_009b as rs
from evidencegate.eval_constants import (
    CONFIG_ID,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
    TOOL_SPEC_ID,
)
from evidencegate.ids import content_id
from evidencegate.policy import PolicyTrace, StopReason, TurnRecord
from evidencegate.run_plan import Block, plan_bytes
from evidencegate.run_summary import idle_gaps, summary_bytes
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore, record_to_json
from evidencegate.t009.splits import CALIBRATION_V1

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "writeup" / "inputs" / "run_summary_009b.json"
HIDDEN = (
    "correct_disposition",
    "evidence_safe",
    "observe_only",
    "rollback_deployment",
    "scale_service",
)
SEQ_CONFIG: dict[str, Any] = {"type": "ollama", "model": "m", "options": {"temperature": 0.7}}
PAR_CONFIG: dict[str, Any] = SEQ_CONFIG | {
    "parallel": {"client_concurrency": 2, "server_num_parallel": 2}
}


def rec(
    case: tuple[str, str, int],
    *,
    repeat: int = 0,
    skill: str = "s0skill",
    config: dict[str, Any] | None = None,
    started: str = "2026-11-01T00:00:00+00:00",
    duration: float = 10.0,
    action: str | None = "observe_only",
    tools: Iterable[str] = ("get_service_metrics", "observe_only"),
    llm_seed: int = 7,
    format_errors: int = 0,
) -> EpisodeRecord:
    family, level, seed = case
    config = config if config is not None else PAR_CONFIG
    turns = tuple(TurnRecord(t, {}, "{}", False) for t in tools)
    stop = StopReason.ACTION if action else StopReason.MAX_STEPS
    return EpisodeRecord(
        key=EpisodeKey(POLICY_ID, content_id(config), TOOL_SPEC_ID, skill, family, seed, repeat,
                       level),
        policy_config=config, llm_seed=llm_seed, correct_disposition=True, evidence_safe=False,
        action=action, missing_predicates=(), failure_labels=(),
        trace=PolicyTrace(turns, format_errors, stop), started_at=started, duration_s=duration,
    )  # fmt: skip


def write_store(path: Path, records: Iterable[EpisodeRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(record_to_json(r), separators=(",", ":")) + "\n")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 정의 파일 ----------------------------------------------------------------------------------


def test_definition_file_matches_its_pinned_sha256() -> None:
    path = REPO_ROOT / rs.DEFINITION_PATH
    assert (
        sha(path)
        == rs.DEFINITION_SHA256
        == ("8b3462ab36855446c81fc6704bcdaf94ecd2eebafc6dc2269d7384bf5f6e18f5")
    )
    recorded = (REPO_ROOT / "writeup/inputs/run_summary_009b_definition.txt.sha256").read_text()
    assert recorded.split() == [rs.DEFINITION_SHA256, "run_summary_009b_definition.txt"]


def test_every_pinned_input_hash_is_the_one_in_the_definition() -> None:
    text = (REPO_ROOT / rs.DEFINITION_PATH).read_text(encoding="utf-8")
    for name, (path, digest) in rs.PINS.items():
        assert digest in text, name
        assert path.as_posix().split("/")[-1] in text, name
    assert set(rs.PINS) == {
        "t009_manifest",
        "t009b_plan",
        "t009b_sessions",
        "t009b_pool",
        "calibration_sequential",
        "calibration_concurrent",
    }


# A. 009 풀 ----------------------------------------------------------------------------------

MANIFEST: dict[str, Any] = {
    "status": "closed at the pool stage: shortfall (pool 14 / target 40); Dev-B not run",
    "preregistration": {"sha256": "ab", "tags": ["t009-prereg", "t009-prereg-r1"]},
    "summary": {
        "calls": 80,
        "pool": 14,
        "proposer_duplicates": 66,
        "cost_usd": 2.5,
        "ledger_lines": {"ok": 80, "server_error": 4},
    },
}


def test_pool_009_reads_the_manifest_values() -> None:
    run, prov = rs.pool_009(MANIFEST)
    assert run == {
        "calls": 80,
        "size": 14,
        "target": 40,
        "duplicates": 66,
        "cost_usd": 2.5,
        "server_errors": 4,
    }
    assert prov == {"status": MANIFEST["status"], "tags": ["t009-prereg", "t009-prereg-r1"]}


def test_pool_009_refuses_a_status_target_that_is_not_the_design_target() -> None:
    bad = MANIFEST | {"status": "closed: shortfall (pool 14 / target 30)"}
    with pytest.raises(ValueError, match="target"):
        rs.pool_009(bad)


# C. 009b proposer ---------------------------------------------------------------------------

POOL: dict[str, Any] = {
    "calls": 61,
    "pool_size": 40,
    "n_go": 25,
    "go": True,
    "proposer_duplicates": 21,
    "format_errors": 0,
    "invalid": 0,
    "cost_usd": 2.6,
    "proposer_config": {"budget_usd": 5.0},
    "stop_basis": "pool reached 40",
    "last_session_reason": "limit",
    "last_session_end": "2026-11-06T14:12:02+00:00",
    "response_models": {"counts": {"gpt-5.4-2026-03-05": 61}},
    "candidates": [
        {"name": "cand-000", "skill_id": "c0skill", "group": "proposer"},
        {"name": "neutral-rephrase", "skill_id": "ctlskill", "group": "control"},
    ],
}


def test_proposer_009b_reads_the_frozen_pool() -> None:
    run, prov = rs.proposer_009b(POOL)
    assert run == {
        "calls": 61,
        "pool_size": 40,
        "target": 40,
        "n_go": 25,
        "go": True,
        "duplicates": 21,
        "format_errors": 0,
        "invalid": 0,
        "cost_usd": 2.6,
        "budget_usd": 5.0,
    }
    assert prov == {
        "stop_basis": "pool reached 40",
        "last_session_reason": "limit",
        "last_session_end": "2026-11-06T14:12:02+00:00",
        "response_model": "gpt-5.4-2026-03-05",
    }


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"n_go": 30}, "n_go"),
        ({"response_models": {"counts": {"a": 1, "b": 2}}}, "response model"),
    ],
)
def test_proposer_009b_refuses(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rs.proposer_009b(POOL | change)


# D. 같은 seed 처분 일치 ---------------------------------------------------------------------

CASES = [("bad_deployment", "L1", 900_000), ("traffic_spike", "L2", 900_001)]
KEYS = [(c, r) for c in CASES for r in (0, 1)]


def pairs() -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    seq = [rec(c, repeat=r, config=SEQ_CONFIG) for c, r in KEYS]
    par = [rec(c, repeat=r, config=PAR_CONFIG) for c, r in KEYS]
    return seq, par


def test_same_seed_counts_dispositions_and_tool_sequences() -> None:
    seq, par = pairs()
    par[0] = replace(par[0], action=None)  # 처분 다름(처분 없음도 값 하나)
    turns = tuple(reversed(par[1].trace.turns))  # 같은 도구, 순서만 다름
    par[1] = replace(par[1], trace=replace(par[1].trace, turns=turns))
    extra = rec(("bad_deployment", "L1", 1), config=PAR_CONFIG)  # 케이스 밖
    run, prov = rs.same_seed(seq, [*par, extra], cases=CASES, repeats=(0, 1))
    assert run == {
        "pairs": 4,
        "same_disposition": 3,
        "same_tool_sequence": 3,
        "ignored": {"sequential": 0, "concurrent_2": 1},
    }
    assert prov == {
        "configs": {"sequential": content_id(SEQ_CONFIG), "concurrent_2": content_id(PAR_CONFIG)}
    }


def test_same_seed_refuses_a_missing_pair() -> None:
    seq, par = pairs()
    with pytest.raises(ValueError, match="missing"):
        rs.same_seed(seq, par[:-1], cases=CASES, repeats=(0, 1))


def test_same_seed_refuses_a_different_llm_seed() -> None:
    seq, par = pairs()
    par[2] = replace(par[2], llm_seed=99)
    with pytest.raises(ValueError, match="llm_seed"):
        rs.same_seed(seq, par, cases=CASES, repeats=(0, 1))


def test_same_seed_refuses_configs_that_differ_beyond_parallel() -> None:
    seq, _ = pairs()
    other = PAR_CONFIG | {"options": {"temperature": 0.5}}
    par = [rec(c, repeat=r, config=other) for c, r in KEYS]
    with pytest.raises(ValueError, match="parallel"):
        rs.same_seed(seq, par, cases=CASES, repeats=(0, 1))


def test_same_seed_refuses_a_store_with_two_configs() -> None:
    seq, par = pairs()
    # 첫 기록은 정상이라 parallel 점검은 통과한다. 마지막 기록만 다른 설정이다.
    seq[-1] = rec(CASES[1], repeat=1, config=SEQ_CONFIG | {"model": "other"})
    with pytest.raises(ValueError, match="different policy configs"):
        rs.same_seed(seq, par, cases=CASES, repeats=(0, 1))


def test_same_seed_default_case_set_is_v1_calibration_times_five_repeats() -> None:
    assert tuple(CALIBRATION_V1) == rs.SAME_SEED_CASES
    assert rs.SAME_SEED_REPEATS == S0_ORACLE_REPEATS == (0, 1, 2, 3, 4)
    assert len(CALIBRATION_V1) * len(S0_ORACLE_REPEATS) == 600
    assert {c[2] for c in CALIBRATION_V1} == set(range(900_000, 900_020))


# B. 009b 본 실행: 정지 구간 목록 ------------------------------------------------------------


def spaced(starts: list[str], duration: float = 10.0) -> list[EpisodeRecord]:
    return [
        rec(("bad_deployment", "L1", i), started=s, duration=duration) for i, s in enumerate(starts)
    ]


SESSIONS = [
    {"start": "2026-11-01T00:00:00+00:00", "end": "2026-11-01T00:10:00+00:00"},
    {"start": "2026-11-01T01:00:00+00:00", "end": "2026-11-01T01:30:00+00:00"},
]


def test_idle_spans_classify_in_and_between_sessions_and_agree_with_v1() -> None:
    records = spaced(
        [
            "2026-11-01T00:00:00+00:00",
            "2026-11-01T00:05:00+00:00",  # 앞 끝 00:00:10에서 290초: 세션 안
            "2026-11-01T01:00:30+00:00",  # 00:05:10에서: 세션 경계를 걸침 → 사이
            "2026-11-01T01:00:50+00:00",  # 10초 간격: 구간 아님
            "2026-11-01T01:20:00+00:00",  # 세션 안
        ]
    )
    out = rs.idle_spans(records, SESSIONS)
    assert [s["where"] for s in out["spans"]] == ["in_session", "between_sessions", "in_session"]
    assert out["spans"][0] == {
        "start": "2026-11-01T00:00:10+00:00",
        "end": "2026-11-01T00:05:00+00:00",
        "length_s": 290.0,
        "where": "in_session",
    }
    assert out["by_where"]["in_session"]["count"] == 2
    assert out["by_where"]["between_sessions"] == {"count": 1, "total_s": 3320.0}
    v1 = idle_gaps(records)
    assert len(out["spans"]) == v1["count"]
    assert sum(s["length_s"] for s in out["spans"]) == pytest.approx(v1["total_s"])
    assert max(s["length_s"] for s in out["spans"]) == v1["longest_s"]


def test_idle_spans_merge_overlapping_episodes() -> None:
    records = spaced(["2026-11-01T00:00:00+00:00", "2026-11-01T00:00:05+00:00"], duration=100.0)
    assert rs.idle_spans(records, SESSIONS)["spans"] == []


# B 전체와 build: 가짜 저장소 ----------------------------------------------------------------

PLAN_CASES = (("bad_deployment", "L1", 600_000), ("traffic_spike", "L2", 600_001))


def fake_root(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    root, runs = tmp_path / "root", tmp_path / "root" / "runs"
    (root / rs.MANIFEST_009.parent).mkdir(parents=True)
    (root / rs.MANIFEST_009).write_text(json.dumps(MANIFEST))
    (root / rs.POOL_009B.parent).mkdir(parents=True)
    (root / rs.POOL_009B).write_text(json.dumps(POOL))
    plan = [
        Block(0, "cand-000", "proposer", "c0skill", "x", "t009_dev_b", 0, PLAN_CASES),
        Block(1, "S0", "s0", S0_SKILL_ID, "x", "t009_oracle", 1, PLAN_CASES),
    ]
    (root / rs.PLAN_009B).parent.mkdir(parents=True)
    (root / rs.PLAN_009B).write_bytes(plan_bytes(plan))

    def store(split: str, skill: str) -> Path:
        return EpisodeStore.for_run(
            runs, suite=split, policy_id=POLICY_ID, config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID, skill_id=skill,
        ).path  # fmt: skip

    t = "2026-11-05T00:{:02d}:00+00:00"
    write_store(
        store("t009_dev_b", "c0skill"),
        [rec(c, skill="c0skill", started=t.format(i)) for i, c in enumerate(PLAN_CASES)],
    )
    write_store(
        store("t009_oracle", S0_SKILL_ID),
        [
            *(
                rec(c, skill=S0_SKILL_ID, repeat=1, started=t.format(10 + i))
                for i, c in enumerate(PLAN_CASES)
            ),
            rec(PLAN_CASES[0], skill=S0_SKILL_ID, repeat=3),  # 계획 밖(반복 3)
        ],
    )
    log = runs / rs.SESSIONS_009B
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {"event": "probe", "ratio": 1.3, "time": "2026-11-05T00:00:00+00:00"},
                {"event": "session_start", "time": "2026-11-05T00:00:00+00:00"},
                {"event": "session_end", "time": "2026-11-05T00:30:00+00:00",
                 "reason": "plan_done", "blocks_run": 2, "episodes_run": 4,
                 "episodes_failed": 0, "server_errors": 0},
            )
        )
        + "\n"
    )  # fmt: skip
    cal = {
        "calibration_sequential": (rs.SEQ_STORE, SEQ_CONFIG),
        "calibration_concurrent": (rs.PAR_STORE, PAR_CONFIG),
    }
    for path, config in cal.values():
        write_store(
            runs / path,
            [rec(c, repeat=r, config=config) for c in CALIBRATION_V1 for r in S0_ORACLE_REPEATS],
        )
    defn = root / rs.DEFINITION_PATH
    defn.parent.mkdir(parents=True, exist_ok=True)
    defn.write_text("fake definition\n")
    pins = {
        "t009_manifest": sha(root / rs.MANIFEST_009),
        "t009b_plan": sha(root / rs.PLAN_009B),
        "t009b_sessions": sha(log),
        "t009b_pool": sha(root / rs.POOL_009B),
        "calibration_sequential": sha(runs / rs.SEQ_STORE),
        "calibration_concurrent": sha(runs / rs.PAR_STORE),
        "definition": sha(defn),
    }
    return root, runs, pins


def build(root: Path, runs: Path, pins: dict[str, str]) -> dict[str, Any]:
    return rs.build_run_summary_009b(runs, root, pins=pins)


def test_build_assembles_run_and_provenance_sections(tmp_path: Path) -> None:
    root, runs, pins = fake_root(tmp_path)
    out = build(root, runs, pins)
    assert set(out) == {"run", "provenance", "notes"}
    run, prov = out["run"], out["provenance"]
    assert run["t009"]["pool"]["size"] == 14
    assert run["t009b"]["proposer"]["calls"] == 61
    assert run["same_seed"]["pairs"] == 600 and run["same_seed"]["same_disposition"] == 600
    main = run["t009b"]["main"]
    assert main["records"]["episodes"] == 4  # 계획 밖 1개는 뺀다
    assert main["records"]["episodes_by_skill"] == {"S0": 2, "cand-000": 2}
    assert main["sessions"] == [
        {
            "duration_s": 1800.0,
            "reason": "plan_done",
            "blocks_run": 2,
            "episodes_run": 4,
            "episodes_failed": 0,
            "server_errors": 0,
            "probe_ratio": 1.3,
        }
    ]
    assert prov["t009b"]["main"]["sessions"] == [
        {"start": "2026-11-05T00:00:00+00:00", "end": "2026-11-05T00:30:00+00:00"}
    ]
    assert main["server_error_events"] == 0
    assert main["idle"]["count"] == len(main["idle_spans"]["spans"])
    assert prov["definition"] == {
        "path": "writeup/inputs/run_summary_009b_definition.txt",
        "sha256": pins["definition"],
    }
    inputs = prov["inputs"]
    assert inputs["runs/ticket009b/main/sessions.jsonl"] == pins["t009b_sessions"]
    assert inputs["results/ticket-009/proposer-calls-manifest.json"] == pins["t009_manifest"]
    assert any(k.startswith("runs/t009_oracle/") for k in inputs)
    assert all(not k.startswith("/") for k in inputs)


def test_build_output_has_no_cd_es_or_action_kinds(tmp_path: Path) -> None:
    root, runs, pins = fake_root(tmp_path)
    text = summary_bytes(build(root, runs, pins)).decode()
    for word in HIDDEN:
        assert word not in text, word


def test_build_is_byte_deterministic(tmp_path: Path) -> None:
    root, runs, pins = fake_root(tmp_path)
    assert summary_bytes(build(root, runs, pins)) == summary_bytes(build(root, runs, pins))


@pytest.mark.parametrize(
    "name",
    [
        "t009_manifest",
        "t009b_plan",
        "t009b_sessions",
        "t009b_pool",
        "calibration_sequential",
        "calibration_concurrent",
        "definition",
    ],
)
def test_build_refuses_an_input_that_does_not_match_its_pin(tmp_path: Path, name: str) -> None:
    root, runs, pins = fake_root(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        build(root, runs, pins | {name: "0" * 64})


def test_build_refuses_a_plan_whose_records_are_incomplete(tmp_path: Path) -> None:
    root, runs, pins = fake_root(tmp_path)
    store = next((runs / "t009_dev_b").rglob("*.jsonl"))
    store.write_text(store.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="plan"):
        build(root, runs, pins)


def test_main_writes_once_with_a_hash_and_refuses_a_different_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runs, pins = fake_root(tmp_path)
    monkeypatch.setattr(rs, "DEFAULT_PINS", pins)
    out = tmp_path / "out" / "run_summary_009b.json"
    argv = ["--runs-dir", str(runs), "--root", str(root), "--out", str(out)]
    assert rs.main(argv) == 0
    data = out.read_bytes()
    assert (out.parent / "run_summary_009b.json.sha256").read_text() == (
        f"{hashlib.sha256(data).hexdigest()}  run_summary_009b.json\n"
    )
    assert rs.main(argv) == 0 and out.read_bytes() == data
    out.write_text("{}\n")
    assert rs.main(argv) == 1


def test_main_refuses_with_exit_1_on_a_pin_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runs, pins = fake_root(tmp_path)
    monkeypatch.setattr(rs, "DEFAULT_PINS", pins | {"t009b_pool": "0" * 64})
    out = tmp_path / "out.json"
    assert rs.main(["--runs-dir", str(runs), "--root", str(root), "--out", str(out)]) == 1
    assert not out.exists()


# 커밋된 파일 ---------------------------------------------------------------------------------


@pytest.mark.skipif(not COMMITTED.exists(), reason="run_summary_009b.json not generated yet")
def test_committed_run_summary_009b_matches_its_hash() -> None:
    recorded = (COMMITTED.parent / "run_summary_009b.json.sha256").read_text().split()
    assert recorded == [sha(COMMITTED), "run_summary_009b.json"]
    text = COMMITTED.read_text(encoding="utf-8")
    for word in HIDDEN:
        assert word not in text, word


@pytest.mark.skipif(
    not COMMITTED.exists() or not (REPO_ROOT / "runs" / rs.SESSIONS_009B).exists(),
    reason="needs runs/ and the generated file",
)
def test_committed_run_summary_009b_equals_a_rebuild_from_the_runs() -> None:
    built = summary_bytes(rs.build_run_summary_009b(REPO_ROOT / "runs", REPO_ROOT))
    assert COMMITTED.read_bytes() == built
