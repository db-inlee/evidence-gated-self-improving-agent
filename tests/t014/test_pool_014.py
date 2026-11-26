"""Ticket 014 단계 0: Dev-A 내보내기, 풀 동결, 묶음 검사, 단계 0 결과(사전 등록 3·8절). API 없음."""

import hashlib
import json
import shutil
import string
from pathlib import Path
from typing import Any

import pytest

from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.eval_data import CaseId
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import record_to_json
from evidencegate.t014 import plans, pool, stages
from evidencegate.t014.settings import fixed_run_options
from evidencegate.t014.splits import split_cases

ROOT = Path(__file__).resolve().parents[2]
S0_TEXT = (ROOT / "skills/S0/SKILL.md").read_text()
CONFIG = {"model": "gpt-5.4", "prompt_sha256": "58ffd9f0"}


def episode(case: CaseId, cd: bool, es: bool, repeat: int = 0) -> EpisodeRecord:
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


def write_dev_a(runs: Path, records: list[EpisodeRecord]) -> None:
    path = stages.store_path(runs, "t014_dev_a", S0_SKILL_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record_to_json(r), ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in records
        )
    )


def dev_a(pattern: str = "mixed") -> list[EpisodeRecord]:
    out = []
    for i, c in enumerate(split_cases("t014_dev_a")):
        if pattern == "all_ok":
            out.append(episode(c, True, True))
        else:  # 셋 중 하나 성공, 하나 evidence-only, 하나 wrong
            out.append(episode(c, i % 3 != 2, i % 3 == 0))
    return out


# 내보내기 -----------------------------------------------------------------------------


def test_export_writes_the_failures_once_with_sha256(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_dev_a(runs, dev_a())
    out = tmp_path / "export.json"
    assert pool.export_main(runs, out) == 0
    cases = json.loads(out.read_text())
    assert len(cases) == 40  # 60 중 실패 40
    assert (
        out.with_name(out.name + ".sha256").read_text().split()[0]
        == hashlib.sha256(out.read_bytes()).hexdigest()
    )
    assert pool.export_main(runs, out) == 0  # 같은 바이트면 다시 써도 된다
    out.write_text("[]\n")
    assert pool.export_main(runs, out) == 1


def test_export_refuses_missing_or_foreign_records(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_dev_a(runs, dev_a()[1:])
    with pytest.raises(ValueError, match="missing"):
        pool.build_export(pool.dev_a_records(runs))
    foreign = [*dev_a()[1:], episode(("bad_deployment", "L1", 500_000), True, True)]
    with pytest.raises(ValueError, match="not a Ticket 014 Dev-A case"):
        pool.build_export(foreign)


def test_thin_strata_stop_before_any_call(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_dev_a(runs, dev_a("all_ok"))
    out = tmp_path / "export.json"
    assert pool.export_main(runs, out) == 1
    assert not out.exists()


# 풀 동결 ------------------------------------------------------------------------------


def word(i: int) -> str:
    letters = string.ascii_lowercase
    return "check" + letters[i // 26 % 26] + letters[i % 26]


def skill(i: int) -> str:
    return S0_TEXT + f"\n7. Before acting, {word(i)} the {word(i + 300)} signal.\n"


def call(i: int, kind: str) -> dict[str, Any]:
    """kind: pool / dup / excluded / invalid."""
    return {
        "call_index": i,
        "proposer_config": CONFIG,
        "cost_usd": 0.02,
        "format_error_kind": None,
        "invalid_reasons": ["x"] if kind == "invalid" else [],
        "duplicate_of": "cand-000" if kind == "dup" else None,
        "excluded_by": "cand-017" if kind == "excluded" else None,
        "control_duplicates": [] if kind == "pool" else None,
        "in_pool": kind == "pool",
        "candidate": {
            "skill_md": skill(i),
            "evidence": {"candidate_id": f"cand-{i:03d}", "hypothesis": f"H{i}"},
        },
    }


def make_root(tmp_path: Path, kinds: list[str], last_reason: str = "limit") -> Path:
    root = tmp_path / "root"
    for sub in ("skills/S0", "skills/controls", "skills/candidates-009b"):
        shutil.copytree(ROOT / sub, root / sub)
    run = root / "runs/proposer-014"
    (run / "pool").mkdir(parents=True)
    for i, kind in enumerate(kinds):
        (run / "pool" / f"call-{i:03d}.json").write_text(json.dumps(call(i, kind)))
    (run / "pool" / "run_options.json").write_text(json.dumps(fixed_run_options(root)))
    (run / "ledger.jsonl").write_text(
        "".join(json.dumps({"status": "ok", "cost_usd": 0.02}) + "\n" for _ in kinds)
    )
    (run / "response_models.jsonl").write_text(
        "".join(
            json.dumps({"call_index": i, "model": "gpt-5.4-2026-03-05"}) + "\n"
            for i in range(len(kinds))
        )
    )
    (run / "sessions.jsonl").write_text(
        json.dumps({"event": "session_end", "reason": last_reason, "time": "t"}) + "\n"
    )
    return root


def pool_json(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / "skills/candidates-014/POOL-014.json").read_text())
    return data


def test_freeze_writes_the_pool_and_lists_the_009b_exclusions(tmp_path: Path) -> None:
    kinds = ["pool"] * 30 + ["excluded", "dup"] + ["pool"] * 10
    root = make_root(tmp_path, kinds)
    assert pool.freeze_main(root) == 0
    data = pool_json(root)
    assert data["pool_size"] == 40 and data["go"] and data["stop_basis"] == "pool reached 40"
    assert data["excluded_009b"] == [
        {"call_index": 30, "candidate_id": "cand-030", "of": "cand-017"}
    ]
    assert data["proposer_duplicates"] == 1
    assert {c["group"] for c in data["candidates"]} == {"proposer"}
    first = data["candidates"][0]
    assert (root / first["skill_path"]).read_text() == skill(0)
    sha = (root / "skills/candidates-014/POOL-014.json.sha256").read_text().split()[0]
    assert (
        sha
        == hashlib.sha256((root / "skills/candidates-014/POOL-014.json").read_bytes()).hexdigest()
    )
    before = (root / "skills/candidates-014/POOL-014.json").read_bytes()
    assert pool.freeze_main(root) == 0  # 같은 입력이면 같은 바이트
    assert (root / "skills/candidates-014/POOL-014.json").read_bytes() == before


def test_a_pool_below_n_go_is_no_go_without_a_pool_file(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["pool"] * 20 + ["dup"] * 25)
    assert pool.freeze_main(root) == 3
    assert not (root / "skills/candidates-014/POOL-014.json").exists()


def test_unfinished_generation_and_other_run_options_are_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["pool"] * 10, last_reason="stop_file")
    assert pool.freeze_main(root) == 1
    root2 = make_root(tmp_path / "b", ["pool"] * 40)
    (root2 / "runs/proposer-014/pool/run_options.json").write_text(
        json.dumps({"previous": "valid"})
    )
    assert pool.freeze_main(root2) == 1


def groups(root: Path, calls: dict[str, list[int]]) -> None:
    data = json.loads((root / "skills/candidates-014/POOL-014.json").read_text())
    pool_sha = hashlib.sha256(
        (root / "skills/candidates-014/POOL-014.json").read_bytes()
    ).hexdigest()
    shas = {
        str(i): hashlib.sha256(
            (root / f"runs/proposer-014/pool/call-{i:03d}.json").read_bytes()
        ).hexdigest()
        for g in calls.values()
        for i in g
    }
    body = {
        "stage": pool.GROUPS_STAGE,
        "labeled_by": "Claude",
        "method": "read S0 diffs",
        "n_groups": len(calls),
        "n_ideas_read": len(data["candidates"]),
        "pool_sha256": pool_sha,
        "groups": {k: {"calls": v} for k, v in calls.items()},
        "call_sha256": shas,
    }
    (root / "plans").mkdir(exist_ok=True)
    (root / "plans/ticket-014-pool-groups.json").write_text(json.dumps(body))


def test_groups_check_needs_every_pool_call_in_one_group(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["pool"] * 40)
    pool.freeze_main(root)
    groups(root, {"a": list(range(20)), "b": list(range(20, 39))})
    assert any("call 39" in p for p in pool.check_pool_groups(root))
    groups(root, {"a": list(range(20)), "b": list(range(20, 40))})
    assert pool.check_pool_groups(root) == []
    assert pool.groups_ready(root)  # .sha256 전
    assert pool.check_groups_main(root) == 0
    assert pool.groups_ready(root) == []


def test_stage0_result_go_records_pool_groups_export_and_exclusions(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["pool"] * 30 + ["excluded"] + ["pool"] * 10)
    runs = root / "runs"
    (root / "plans").mkdir()
    plans.write_plan(root, 0, plans.build_plan(0, root))
    write_dev_a(runs, dev_a())
    assert pool.export_main(runs, root / "runs/proposer_inputs-014/x.json") == 0
    pool.freeze_main(root)
    groups(root, {"a": [i for i in range(41) if i != 30]})
    pool.check_groups_main(root)
    result = pool.finalize_stage0(root, runs, export=root / "runs/proposer_inputs-014/x.json")
    assert result["go"] is True and result["pool_size"] == 40
    assert result["excluded_009b"][0]["of"] == "cand-017"
    assert len(result["pool_sha256"]) == 64 and len(result["groups_sha256"]) == 64
    assert result["export"]["strata"]["wrong"] == 20
    assert result["dev_a"]["episodes"] == 60
    assert stages.check_previous(root, 1).ok


def test_stage0_result_no_go_when_the_pool_is_short(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["pool"] * 20 + ["dup"] * 25)
    runs = root / "runs"
    (root / "plans").mkdir()
    plans.write_plan(root, 0, plans.build_plan(0, root))
    write_dev_a(runs, dev_a())
    pool.export_main(runs, root / "runs/proposer_inputs-014/x.json")
    result = pool.finalize_stage0(root, runs, export=root / "runs/proposer_inputs-014/x.json")
    assert result["go"] is False and result["no_go_reason"] == "pool 20 < N_go 25"
    assert not stages.check_previous(root, 1).ok


def test_stage0_result_no_go_when_the_strata_are_thin(tmp_path: Path) -> None:
    root = make_root(tmp_path, [])
    runs = root / "runs"
    (root / "plans").mkdir()
    plans.write_plan(root, 0, plans.build_plan(0, root))
    write_dev_a(runs, dev_a("all_ok"))
    result = pool.finalize_stage0(root, runs, export=root / "runs/proposer_inputs-014/x.json")
    assert result["go"] is False and result["no_go_reason"].startswith("strata")
