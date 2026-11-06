"""Ticket 009b 풀 동결(사전 등록 2.1·6.6·8절, 티켓 §4의 4). 가짜 풀로만 시험한다(API 호출 없음)."""

import hashlib
import json
import shutil
import string
from pathlib import Path
from typing import Any

import pytest

from evidencegate.t009b import pool_freeze as pf
from evidencegate.t009b import run_plan as rp
from evidencegate.t009b.settings import FIXED_RUN_OPTIONS

ROOT = Path(__file__).resolve().parents[2]
S0_TEXT = (ROOT / "skills/S0/SKILL.md").read_text()
CONFIG = {"model": "gpt-5.4", "prompt_sha256": "58ffd9f0"}


def word(i: int) -> str:
    letters = string.ascii_lowercase
    return "check" + letters[i // 26 % 26] + letters[i % 26]


def skill(i: int) -> str:
    return S0_TEXT + f"\n7. Before acting, {word(i)} the {word(i + 300)} signal.\n"


def record(i: int, in_pool: bool, text: str | None = None) -> dict[str, Any]:
    return {
        "call_index": i,
        "proposer_config": CONFIG,
        "cost_usd": 0.02,
        "format_error": None,
        "format_error_kind": None,
        "invalid_reasons": [],
        "duplicate_of": None if in_pool else "cand-000",
        "control_duplicates": [] if in_pool else None,
        "in_pool": in_pool,
        "candidate": {
            "skill_md": text or skill(i),
            "evidence": {"candidate_id": f"cand-{i:03d}", "hypothesis": f"H{i}"},
        },
    }


def make_root(tmp_path: Path, pattern: list[bool], last_reason: str = "limit") -> Path:
    """pattern[i]가 참이면 call i가 풀 후보다."""
    root = tmp_path / "root"
    for sub in ("skills/S0", "skills/controls", "skills/candidates"):
        shutil.copytree(ROOT / sub, root / sub)
    run = root / "runs/proposer-009b"
    (run / "pool").mkdir(parents=True)
    for i, flag in enumerate(pattern):
        (run / "pool" / f"call-{i:03d}.json").write_text(json.dumps(record(i, flag)))
    (run / "pool" / "run_options.json").write_text(json.dumps(FIXED_RUN_OPTIONS))
    (run / "ledger.jsonl").write_text(
        "".join(json.dumps({"status": "ok", "cost_usd": 0.02}) + "\n" for _ in pattern)
    )
    (run / "response_models.jsonl").write_text(
        "".join(
            json.dumps({"call_index": i, "model": "gpt-5.4-2026-03-05"}) + "\n"
            for i in range(len(pattern))
        )
    )
    (run / "sessions.jsonl").write_text(
        json.dumps(
            {"event": "session_end", "reason": last_reason, "time": "2026-11-07T10:00:00+00:00"}
        )
        + "\n"
    )
    # 009 후보(가짜 2개)와 그 sha256 지도
    old = root / "runs/proposer-009/pool"
    old.mkdir(parents=True)
    sha = {}
    for i, text in enumerate([skill(900), S0_TEXT.replace("2. Check", "2. Verify")]):
        data = json.dumps({"candidate": {"skill_md": text}}).encode()
        (old / f"call-{i:03d}.json").write_bytes(data)
        sha[str(i)] = hashlib.sha256(data).hexdigest()
    (root / "plans").mkdir()
    (root / "plans/ticket-009b-dedupe-groups.json").write_text(json.dumps({"call_sha256": sha}))
    return root


def freeze(root: Path) -> int:
    return pf.main(["freeze", "--root", str(root)])


def pool_json(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / "skills/candidates-009b/POOL-009b.json").read_text())
    return data


# 멈춤 근거 ----------------------------------------------------------------------


def test_pool_40_is_a_preregistered_stop(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 40)
    assert freeze(root) == 0
    data = pool_json(root)
    assert data["stop_basis"] == "pool reached 40"
    assert data["pool_size"] == 40 and data["go"] is True


def test_25_idle_calls_is_a_preregistered_stop(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 25 + [False] * 25)
    assert freeze(root) == 0
    assert pool_json(root)["stop_basis"] == "no new pool candidate in the last 25 calls"


def test_120_calls_is_a_preregistered_stop(tmp_path: Path) -> None:
    root = make_root(tmp_path, [i % 4 == 0 for i in range(120)])
    assert freeze(root) == 0
    data = pool_json(root)
    assert data["stop_basis"] == "max total calls 120 reached" and data["pool_size"] == 30


@pytest.mark.parametrize("reason", ["stopped", "stop_file", "stop_at", "interrupted"])
def test_unfinished_generation_is_refused(tmp_path: Path, reason: str) -> None:
    """비용 상한·잔액 부족·형식 오류율(모두 stopped)·STOP.

    사전 등록 멈춤이 없으면 거부한다(종료 코드 1).
    """
    root = make_root(tmp_path, [True] * 10 + [False] * 3, last_reason=reason)
    assert freeze(root) == 1
    assert not (root / "skills/candidates-009b").exists()
    assert not (root / pf.SHORTFALL_PATH).exists()


def test_shortfall_writes_only_the_record_and_exits_3(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 10 + [False] * 25)
    assert freeze(root) == 3
    assert not (root / "skills/candidates-009b").exists()
    data = json.loads((root / pf.SHORTFALL_PATH).read_text())
    assert data["pool_size"] == 10 and data["n_go"] == 25 and data["calls"] == 35
    assert data["stop_basis"] == "no new pool candidate in the last 25 calls"
    assert data["last_session_reason"] == "limit" and data["go"] is False


def test_the_pool_count_is_proposer_candidates_only(tmp_path: Path) -> None:
    """통제 7개는 세지 않는다(009의 '풀 14 / 목표 40'과 같은 기준). 24개 + 통제 7개는 미달이다."""
    root = make_root(tmp_path, [True] * 24 + [False] * 25)
    assert freeze(root) == 3
    root = make_root(tmp_path / "b", [True] * 25 + [False] * 25)
    assert freeze(root) == 0
    groups = [c["group"] for c in pool_json(root)["candidates"]]
    assert groups.count("proposer") == 25 and groups.count("control") == 7
    assert pool_json(root)["pool_size"] == 25


# 무결성 ------------------------------------------------------------------------


def test_a_gap_in_call_indices_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_root(tmp_path, [True] * 41)
    (root / "runs/proposer-009b/pool/call-005.json").unlink()  # 남은 40개로도 풀 40이 성립한다
    assert freeze(root) == 1
    assert "call_index has gaps" in capsys.readouterr().err


def test_mixed_proposer_configs_are_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 40)
    path = root / "runs/proposer-009b/pool/call-003.json"
    data = json.loads(path.read_text())
    data["proposer_config"] = {**CONFIG, "model": "gpt-x"}
    path.write_text(json.dumps(data))
    assert freeze(root) == 1


def test_other_run_options_or_a_budget_overrun_are_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 40)
    (root / "runs/proposer-009b/pool/run_options.json").write_text(
        json.dumps({**FIXED_RUN_OPTIONS, "dedupe_threshold": 0.6})
    )
    assert freeze(root) == 1
    root = make_root(tmp_path / "b", [True] * 40)
    with (root / "runs/proposer-009b/ledger.jsonl").open("a") as f:
        f.write(json.dumps({"status": "ok", "cost_usd": 5.0}) + "\n")
    assert freeze(root) == 1


def test_a_different_response_model_is_recorded_not_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 40)
    path = root / "runs/proposer-009b/response_models.jsonl"
    path.write_text(path.read_text().replace("gpt-5.4-2026-03-05", "gpt-5.4-2026-09-01", 1))
    assert freeze(root) == 0
    summary = pool_json(root)["response_models"]
    assert summary["same_as_009"] is False
    assert summary["counts"] == {"gpt-5.4-2026-03-05": 39, "gpt-5.4-2026-09-01": 1}


# 동결 내용 ----------------------------------------------------------------------


def test_skill_files_and_duplicate_marks(tmp_path: Path) -> None:
    pattern = [True] * 40
    root = make_root(tmp_path, pattern)
    v1_text = (root / "skills/candidates/cand-003/SKILL.md").read_text()
    for i, text in ((1, v1_text), (2, skill(900))):  # v1 후보와 같음, 009 후보와 같음
        path = root / f"runs/proposer-009b/pool/call-{i:03d}.json"
        path.write_text(json.dumps(record(i, True, text)))
    assert freeze(root) == 0
    by_name = {c["name"]: c for c in pool_json(root)["candidates"]}
    assert by_name["cand-001"]["v1_duplicates"] == ["cand-003"]
    assert by_name["cand-002"]["t009_duplicates"] == [0]
    assert (
        by_name["cand-004"]["v1_duplicates"] == [] and by_name["cand-004"]["t009_duplicates"] == []
    )
    copied = root / "skills/candidates-009b/cand-001/SKILL.md"
    assert copied.read_bytes() == v1_text.encode()
    assert by_name["cand-001"]["skill_path"] == "skills/candidates-009b/cand-001/SKILL.md"
    assert (root / "skills/candidates-009b/POOL-009b.json.sha256").read_text().split()[
        1
    ] == "POOL-009b.json"


def test_009_candidate_files_must_match_their_recorded_hashes(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 40)
    (root / "runs/proposer-009/pool/call-000.json").write_text("{}")
    assert freeze(root) == 1


def test_an_existing_pool_must_be_identical(tmp_path: Path) -> None:
    root = make_root(tmp_path, [True] * 40)
    assert freeze(root) == 0
    assert freeze(root) == 0  # 같은 바이트면 그대로
    pool = root / "skills/candidates-009b/POOL-009b.json"
    pool.write_text(pool.read_text().replace('"pool_size": 40', '"pool_size": 41'))
    assert freeze(root) == 1


# 묶음 파일과 계획 동결 -------------------------------------------------------------------


def write_groups(root: Path, groups: dict[str, list[int]], *, pool_sha: str | None = None) -> Path:
    pool = root / "skills/candidates-009b/POOL-009b.json"
    calls = sorted(i for members in groups.values() for i in members)
    data = {
        "stage": pf.GROUPS_STAGE,
        "labeled_by": "Claude (Claude Code), by reading each candidate's S0 diff",
        "method": "one group per change intent, as in the 009 groups (d443e7e)",
        "n_groups": len(groups),
        "n_ideas_read": len(groups),
        "pool_sha256": pool_sha or hashlib.sha256(pool.read_bytes()).hexdigest(),
        "groups": {k: {"label": f"idea {k}", "calls": v} for k, v in groups.items()},
        "call_sha256": {
            str(i): hashlib.sha256(
                (root / f"runs/proposer-009b/pool/call-{i:03d}.json").read_bytes()
            ).hexdigest()
            for i in calls
        },
    }
    path = root / rp.GROUPS_PATH
    path.write_text(json.dumps(data, indent=1))
    return path


def frozen(tmp_path: Path) -> Path:
    root = make_root(tmp_path, [True] * 25 + [False] * 25)
    assert freeze(root) == 0
    return root


def test_groups_check_needs_every_pool_candidate_in_exactly_one_group(tmp_path: Path) -> None:
    root = frozen(tmp_path)
    write_groups(root, {"A": list(range(0, 10)), "B": list(range(10, 25))})
    report = pf.check_pool_groups(root)
    assert report.problems == []
    assert report.representatives == {"A": 0, "B": 10}  # 묶음 안 가장 먼저 나온 풀 후보(6.6)
    write_groups(root, {"A": list(range(0, 10)), "B": list(range(11, 25))})
    assert any("missing" in p for p in pf.check_pool_groups(root).problems)
    write_groups(root, {"A": list(range(0, 11)), "B": list(range(10, 25))})
    assert any("more than one" in p for p in pf.check_pool_groups(root).problems)
    write_groups(root, {"A": list(range(0, 10)), "B": [*range(10, 25), 30]})
    assert any("not a pool candidate" in p for p in pf.check_pool_groups(root).problems)


def test_groups_check_needs_the_metadata(tmp_path: Path) -> None:
    root = frozen(tmp_path)
    path = write_groups(root, {"A": list(range(25))})
    data = json.loads(path.read_text())
    del data["n_ideas_read"]
    data["stage"] = "whenever"
    path.write_text(json.dumps(data))
    problems = pf.check_pool_groups(root).problems
    assert any("n_ideas_read" in p for p in problems) and any("stage" in p for p in problems)


def test_groups_cli_records_the_sha256_only_when_valid(tmp_path: Path) -> None:
    root = frozen(tmp_path)
    write_groups(root, {"A": list(range(0, 10)), "B": list(range(10, 25))})
    assert pf.main(["check-groups", "--root", str(root)]) == 0
    sha = (root / "plans/ticket-009b-pool-groups.json.sha256").read_text().split()
    assert sha[0] == hashlib.sha256((root / rp.GROUPS_PATH).read_bytes()).hexdigest()
    write_groups(root, {"A": list(range(0, 10))})
    assert pf.main(["check-groups", "--root", str(root)]) == 1


def test_plan_freeze_needs_valid_groups_for_this_pool(tmp_path: Path) -> None:
    root = frozen(tmp_path)
    assert rp.main(["--root", str(root)]) == 1  # 묶음 파일 없음
    write_groups(root, {"A": list(range(25))}, pool_sha="0" * 64)
    pf.main(["check-groups", "--root", str(root)])
    assert rp.main(["--root", str(root)]) == 1  # 풀 sha256 불일치
    assert not (root / rp.PLAN_PATH).exists()
    write_groups(root, {"A": list(range(25))})
    assert pf.main(["check-groups", "--root", str(root)]) == 0
    assert rp.main(["--root", str(root)]) == 0
    assert (root / rp.PLAN_PATH).exists()


def test_plan_freeze_refuses_groups_changed_after_their_check(tmp_path: Path) -> None:
    root = frozen(tmp_path)
    path = write_groups(root, {"A": list(range(25))})
    assert pf.main(["check-groups", "--root", str(root)]) == 0
    path.write_text(path.read_text().replace("idea A", "idea A2"))
    assert rp.main(["--root", str(root)]) == 1
