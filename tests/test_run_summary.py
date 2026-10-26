"""실행 요약(Ticket 007 §1 세 번째 출처)과 E1 정의 파일(§2) 테스트.

실행 요약은 실행 기록과 세션 기록을 읽기만 한다. 운영 지표만 담고 CD·ES·조치 종류는 담지 않는다.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evidencegate.policy import PolicyTrace, StopReason, TurnRecord
from evidencegate.run_summary import (
    build_run_summary,
    idle_gaps,
    main,
    record_summary,
    session_summary,
    summary_bytes,
    ticket004_summary,
)
from evidencegate.runner import EpisodeKey, EpisodeRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "writeup" / "inputs" / "run_summary.json"
E1 = REPO_ROOT / "writeup" / "inputs" / "e1_definition.txt"
# CD·ES 필드와 조치 이름. 통제 후보 이름(neg-rollback-any)과 겹치지 않게 조치 이름 전체로 본다.
HIDDEN = (
    "correct_disposition",
    "evidence_safe",
    "observe_only",
    "rollback_deployment",
    "scale_service",
)


def rec(
    seed: int,
    started: str,
    duration: float,
    *,
    skill: str = "s0",
    turns: int = 4,
    format_errors: int = 0,
    action: str | None = "observe_only",
) -> EpisodeRecord:
    turn = TurnRecord("get_service_metrics", {}, "{}", False)
    return EpisodeRecord(
        key=EpisodeKey("p", "c", "t", skill, "bad_deployment", seed, 0, "L3"),
        policy_config={},
        llm_seed=0,
        correct_disposition=True,
        evidence_safe=True,
        action=action,
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace(
            (turn,) * turns,
            format_errors,
            StopReason.ACTION if action else StopReason.MAX_STEPS,
        ),
        started_at=started,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# 기록 요약과 정지 구간
# ---------------------------------------------------------------------------


def test_record_summary_counts_operational_metrics_only() -> None:
    records = [
        rec(1, "2026-10-25T00:00:00+00:00", 10.0, turns=4, skill="a"),
        rec(2, "2026-10-25T00:00:00+00:00", 12.0, turns=6, format_errors=2, skill="b"),
        rec(3, "2026-10-25T00:00:10+00:00", 8.0, turns=2, action=None, skill="b"),
    ]
    s = record_summary(records, names={"a": "cand-a", "b": "cand-b"})
    assert s["episodes"] == 3
    assert s["episodes_by_skill"] == {"cand-a": 1, "cand-b": 2}
    assert s["format_error_episodes"] == {"cand-b": 1}
    assert s["format_errors_total"] == 2
    assert s["no_disposition"] == 1
    assert s["stop_reasons"] == {"action": 2, "max_steps": 1}
    assert s["tool_calls"] == {"mean": 4.0, "min": 2, "max": 6, "counts": {"2": 1, "4": 1, "6": 1}}
    assert s["policy_time_mean_s"] == 10.0
    assert s["first_start"] == "2026-10-25T00:00:00+00:00"
    assert s["last_end"] == "2026-10-25T00:00:18+00:00"
    assert s["wall_s"] == 18.0
    for word in HIDDEN:
        assert word not in json.dumps(s)


def test_idle_gaps_find_spans_with_nothing_running() -> None:
    records = [
        rec(1, "2026-10-25T00:00:00+00:00", 10.0),
        rec(2, "2026-10-25T00:00:05+00:00", 10.0),  # 겹침: 0~15초 실행 중
        rec(3, "2026-10-25T00:02:15+00:00", 10.0),  # 15초 ~ 135초 빈 구간(120초)
        rec(4, "2026-10-25T00:02:50+00:00", 10.0),  # 145초 ~ 170초 빈 구간(25초, 기준 미만)
    ]
    gaps = idle_gaps(records, min_gap_s=60.0)
    assert gaps["count"] == 1
    assert gaps["total_s"] == 120.0 and gaps["longest_s"] == 120.0
    assert gaps["first_start"] == "2026-10-25T00:00:15+00:00"
    assert gaps["last_end"] == "2026-10-25T00:02:15+00:00"
    assert gaps["active_s"] == 15.0 + 10.0 + 10.0
    assert gaps["min_gap_s"] == 60.0


def test_idle_gaps_without_a_gap() -> None:
    records = [rec(1, "2026-10-25T00:00:00+00:00", 10.0)]
    gaps = idle_gaps(records, min_gap_s=60.0)
    assert gaps == {
        "min_gap_s": 60.0,
        "count": 0,
        "total_s": 0.0,
        "longest_s": 0.0,
        "first_start": None,
        "last_end": None,
        "active_s": 10.0,
    }


# ---------------------------------------------------------------------------
# 세션 기록과 Ticket 004
# ---------------------------------------------------------------------------


def write_log(path: Path, events: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def test_session_summary_pairs_starts_and_ends(tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "s.jsonl",
        [
            {"event": "probe", "time": "2026-10-22T14:54:40+00:00", "ratio": 1.36},
            {"event": "session_start", "time": "2026-10-22T14:54:40+00:00", "blocks": 2},
            {
                "event": "session_end",
                "time": "2026-10-22T15:08:41+00:00",
                "reason": "plan_done",
                "blocks_run": 2,
                "episodes_run": 120,
                "episodes_failed": 0,
                "server_errors": 0,
            },
            {"event": "server_error", "time": "2026-10-22T15:00:00+00:00"},
        ],
    )
    s = session_summary(log)
    assert s["server_error_events"] == 1
    assert s["sessions"] == [
        {
            "start": "2026-10-22T14:54:40+00:00",
            "end": "2026-10-22T15:08:41+00:00",
            "duration_s": 841.0,
            "reason": "plan_done",
            "blocks_run": 2,
            "episodes_run": 120,
            "episodes_failed": 0,
            "server_errors": 0,
            "probe_ratio": 1.36,
        }
    ]


def test_session_summary_rejects_an_unfinished_session(tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "s.jsonl",
        [{"event": "session_start", "time": "2026-10-22T14:54:40+00:00", "blocks": 2}],
    )
    with pytest.raises(ValueError, match="without an end"):
        session_summary(log)


def test_ticket004_summary_from_the_frozen_pool() -> None:
    pool = json.loads((REPO_ROOT / "skills" / "candidates" / "POOL.json").read_text())
    s = ticket004_summary(pool)
    assert s["calls"] == 16 and s["format_errors"] == 0
    assert (s["target"], s["pool_size"], s["shortfall"]) == (12, 12, False)
    assert s["excluded"] == {"duplicate": 4}
    assert s["controls"] == 7
    assert s["candidates_overlapping_a_control"] == 1
    assert s["model"] == "gpt-5.4" and s["budget_usd"] == 1.5
    assert s["pool_cost_usd"] == pool["pool_cost_usd"]
    assert s["ledger_cumulative_usd"] == pool["ledger_cumulative_usd"]


# ---------------------------------------------------------------------------
# 전체 요약: 결정성, 입력 해시, 명령
# ---------------------------------------------------------------------------


def fake_runs(tmp_path: Path) -> Path:
    from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, TOOL_SPEC_ID
    from evidencegate.store import EpisodeStore

    runs = tmp_path / "runs"
    for split, start in (("dev_a", "2026-10-22T15:00:00+00:00"), ("l3_oracle", None)):
        store = EpisodeStore.for_run(
            runs,
            suite=split,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id="s0skill",
        )
        t = start or "2026-10-25T14:01:07+00:00"
        r = rec(1, t, 5.0, skill="s0skill")
        key = EpisodeKey(
            POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, "s0skill", "bad_deployment", 1, 0, "L3"
        )
        store.append(EpisodeRecord(**{**r.__dict__, "key": key}))
    for ticket in ("ticket005", "ticket006"):
        write_log(
            runs / ticket / "sessions.jsonl",
            [
                {"event": "probe", "time": "2026-10-22T14:54:40+00:00", "ratio": 1.3},
                {"event": "session_start", "time": "2026-10-22T14:54:40+00:00"},
                {
                    "event": "session_end",
                    "time": "2026-10-22T15:08:41+00:00",
                    "reason": "plan_done",
                    "blocks_run": 1,
                    "episodes_run": 1,
                    "episodes_failed": 0,
                    "server_errors": 0,
                },
            ],
        )
    return runs


def stores(runs: Path) -> dict[str, list[Path]]:
    return {
        "ticket005": sorted((runs / "dev_a").rglob("*.jsonl")),
        "ticket006": sorted((runs / "l3_oracle").rglob("*.jsonl")),
        "ticket006_check": [],
    }


def test_build_records_input_hashes_and_is_deterministic(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    a = build_run_summary(runs, REPO_ROOT, stores=stores(runs), names={"s0skill": "S0"})
    b = build_run_summary(runs, REPO_ROOT, stores=stores(runs), names={"s0skill": "S0"})
    assert summary_bytes(a) == summary_bytes(b)
    assert set(a) == {"inputs", "ticket004", "ticket005", "ticket006", "ticket006_check", "notes"}
    assert a["ticket005"]["records"]["episodes"] == 1
    assert a["ticket006"]["sessions"]["sessions"][0]["episodes_run"] == 1
    inputs = a["inputs"]
    pool = REPO_ROOT / "skills" / "candidates" / "POOL.json"
    assert inputs["skills/candidates/POOL.json"] == hashlib.sha256(pool.read_bytes()).hexdigest()
    for path in stores(runs)["ticket005"]:
        rel = path.relative_to(runs.parent).as_posix()
        assert inputs[rel] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "runs/ticket005/sessions.jsonl" in inputs
    assert not any("ledger" in k for k in inputs)  # proposer 장부는 읽지 않는다
    for word in HIDDEN:
        assert word not in summary_bytes(a).decode()


def test_main_writes_once_with_a_hash_and_refuses_a_different_file(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    out = tmp_path / "out" / "run_summary.json"
    args = ["--runs-dir", str(runs), "--out", str(out), "--root", str(REPO_ROOT)]
    assert main(args, stores=stores(runs), names={"s0skill": "S0"}) == 0
    data = out.read_bytes()
    sha = out.with_name("run_summary.json.sha256").read_text().split()[0]
    assert sha == hashlib.sha256(data).hexdigest()
    assert main(args, stores=stores(runs), names={"s0skill": "S0"}) == 0  # 같으면 그대로
    out.write_bytes(data + b" ")
    assert main(args, stores=stores(runs), names={"s0skill": "S0"}) == 1


# ---------------------------------------------------------------------------
# git에 올린 파일
# ---------------------------------------------------------------------------


def recorded_sha(path: Path) -> str:
    return path.with_name(path.name + ".sha256").read_text().split()[0]


def test_committed_run_summary_matches_its_hash() -> None:
    assert hashlib.sha256(SUMMARY.read_bytes()).hexdigest() == recorded_sha(SUMMARY)


@pytest.mark.skipif(not (REPO_ROOT / "runs" / "l3_oracle").exists(), reason="needs runs/")
def test_committed_run_summary_equals_a_rebuild_from_the_runs() -> None:
    assert summary_bytes(build_run_summary(REPO_ROOT / "runs", REPO_ROOT)) == SUMMARY.read_bytes()


def test_committed_run_summary_has_the_expected_totals() -> None:
    s = json.loads(SUMMARY.read_text())
    assert s["ticket005"]["records"]["episodes"] == 12_540
    assert s["ticket006"]["records"]["episodes"] == 7_200
    assert s["ticket006_check"]["records"]["episodes"] == 30
    assert s["ticket004"]["calls"] == 16
    for word in HIDDEN:
        assert word not in SUMMARY.read_text()


def test_e1_definition_matches_its_hash_and_states_the_fixed_definition() -> None:
    text = E1.read_text(encoding="utf-8")
    assert hashlib.sha256(E1.read_bytes()).hexdigest() == recorded_sha(E1)
    assert "E1 = (1/3)(Δ_bad,L2 + Δ_dep,L2 + Δ_traffic,L2)" in text
    assert "|Δ_L3 − E1| < |Δ_L3 − Δ_L1/L2|" in text  # noqa: RUF001 (파일의 빼기 기호 그대로)
    assert "E1 is a post-hoc, exploratory approximation." in text
    assert "results/ticket-005/analysis.json" in text
