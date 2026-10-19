"""후보 풀 동결 (Ticket 004 §6). 실제 풀 대신 임시 폴더의 가짜 결과로 확인한다."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evidencegate.pool_freeze import (
    PREREGISTRATION_COMMIT,
    freeze_pool,
    topic_of,
    verify_frozen_pool,
)
from evidencegate.skills import load_skill

REPO_ROOT = Path(__file__).resolve().parents[1]
S0 = (REPO_ROOT / "skills" / "S0" / "SKILL.md").read_text(encoding="utf-8")
FROZEN_AT = "2026-10-20T10:00:00+09:00"
CONFIG = {"model": "gpt-x", "prompt_sha256": "p" * 64, "reasoning_effort": "low"}


def skill_with(step: str) -> str:
    return S0.replace("6. Otherwise", f"6. {step}\n7. Otherwise")


def record(
    index: int,
    *,
    step: str | None = None,
    in_pool: bool = True,
    duplicate_of: str | None = None,
    control_duplicates: list[str] | None = None,
    invalid: list[str] | None = None,
    format_error_kind: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = None
    if format_error_kind is None:
        candidate = {
            "skill_md": skill_with(step or f"Check traffic case {'abcdefghijklmnop'[index]}."),
            "evidence": {
                "candidate_id": f"cand-{index:03d}",
                "failure_type": "PREMATURE_ACTION",
                "source_cases": [f"case-{i:012x}" for i in range(6)],
                "hypothesis": f"H{index}",
                "expected_effect": {"cd": "up", "es": "up"},
            },
        }
    return {
        "call_index": index,
        "proposer_config": config or CONFIG,
        "cost_usd": 0.01,
        "format_error": None if format_error_kind is None else "bad",
        "format_error_kind": format_error_kind,
        "invalid_reasons": None if candidate is None else (invalid or []),
        "duplicate_of": duplicate_of,
        "control_duplicates": control_duplicates if in_pool else None,
        "in_pool": in_pool and candidate is not None and not invalid and duplicate_of is None,
        "candidate": candidate,
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "skills" / "S0", tmp_path / "skills" / "S0")
    shutil.copytree(REPO_ROOT / "skills" / "controls", tmp_path / "skills" / "controls")
    return tmp_path


def write_pool(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    pool_dir = tmp_path / "runs" / "pool"
    pool_dir.mkdir(parents=True)
    for r in records:
        (pool_dir / f"call-{r['call_index']:03d}.json").write_text(json.dumps(r))
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"cost_usd": 0.03, "cumulative_usd": 0.03})
        + "\n"
        + json.dumps({"cost_usd": 0.02, "cumulative_usd": 0.05})
        + "\n"
    )
    return pool_dir


def standard_records() -> list[dict[str, Any]]:
    return [
        record(0, control_duplicates=[]),
        record(1, control_duplicates=["pos-timing-only"]),
        record(2, in_pool=False, duplicate_of="cand-001"),
        record(3, in_pool=False, invalid=["line_count"]),
        record(4, format_error_kind="parse_failure"),
        record(5, control_duplicates=[]),
    ]


def freeze(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    pool_dir = write_pool(root, records)
    return freeze_pool(root, pool_dir, root / "ledger.jsonl", frozen_at=FROZEN_AT)


def test_pool_candidates_are_written_exactly(root: Path) -> None:
    records = standard_records()
    freeze(root, records)
    written = sorted(p.name for p in (root / "skills" / "candidates").iterdir() if p.is_dir())
    assert written == ["cand-000", "cand-001", "cand-005"]
    for r in (records[0], records[1], records[5]):
        folder = root / "skills" / "candidates" / r["candidate"]["evidence"]["candidate_id"]
        assert (folder / "SKILL.md").read_bytes() == r["candidate"]["skill_md"].encode("utf-8")
        assert json.loads((folder / "evidence.json").read_text()) == r["candidate"]["evidence"]


def test_pool_json_records_the_ticket_fields(root: Path) -> None:
    pool = freeze(root, standard_records())
    assert pool == json.loads((root / "skills" / "candidates" / "POOL.json").read_text())
    assert pool["frozen_at"] == FROZEN_AT
    assert pool["preregistration_commit"] == PREREGISTRATION_COMMIT == "aa1c809"
    assert pool["proposer_config"] == CONFIG
    assert [c["candidate_id"] for c in pool["proposer_candidates"]] == [
        "cand-000",
        "cand-001",
        "cand-005",
    ]
    second = pool["proposer_candidates"][1]
    assert second["control_duplicates"] == ["pos-timing-only"]
    path = root / "skills" / "candidates" / "cand-001" / "SKILL.md"
    assert second["skill_id"] == load_skill(path).skill_id
    assert second["call_index"] == 1
    assert {c["name"]: c["subset"] for c in pool["controls"]} == {
        "mixed-scale-on-traffic": "mixed",
        "neg-rollback-any": "negative",
        "neg-skip-metrics": "negative",
        "neg-timing-shortcut": "negative",
        "neutral-rephrase": "neutral",
        "pos-timing-only": "positive",
        "pos-traffic-check": "positive",
    }
    control = root / "skills" / "controls" / "pos-timing-only" / "SKILL.md"
    assert {c["name"]: c["skill_id"] for c in pool["controls"]}["pos-timing-only"] == (
        load_skill(control).skill_id
    )


def test_excluded_calls_have_reasons(root: Path) -> None:
    pool = freeze(root, standard_records())
    assert pool["excluded"] == [
        {"call_index": 2, "candidate_id": "cand-002", "reason": "duplicate", "detail": "cand-001"},
        {"call_index": 3, "candidate_id": "cand-003", "reason": "invalid", "detail": "line_count"},
        {
            "call_index": 4,
            "candidate_id": None,
            "reason": "format_error",
            "detail": "parse_failure",
        },
    ]


def test_counts_cost_and_shortfall(root: Path) -> None:
    pool = freeze(root, standard_records())
    assert pool["calls"] == 6
    assert pool["format_errors"] == 1
    assert pool["pool_size"] == 3
    assert pool["target"] == 12
    assert pool["shortfall"] is True
    assert pool["pool_cost_usd"] == pytest.approx(0.06)
    assert pool["ledger_cumulative_usd"] == pytest.approx(0.05)


def test_twelve_candidates_is_no_shortfall(root: Path) -> None:
    pool = freeze(root, [record(i, control_duplicates=[]) for i in range(12)])
    assert (pool["pool_size"], pool["shortfall"]) == (12, False)


def test_file_hashes_cover_candidates_and_controls(root: Path) -> None:
    pool = freeze(root, standard_records())
    files = pool["files"]
    assert "skills/candidates/cand-000/SKILL.md" in files
    assert "skills/candidates/cand-000/evidence.json" in files
    assert "skills/controls/pos-timing-only/SKILL.md" in files
    assert "skills/candidates/POOL.json" not in files
    assert all(len(h) == 64 for h in files.values())


def test_freeze_refuses_to_overwrite(root: Path) -> None:
    freeze(root, standard_records())
    with pytest.raises(FileExistsError):
        freeze_pool(root, root / "runs" / "pool", root / "ledger.jsonl", frozen_at=FROZEN_AT)


def test_freeze_refuses_mixed_proposer_configs(root: Path) -> None:
    other = {**CONFIG, "model": "gpt-y"}
    with pytest.raises(ValueError, match="proposer_config"):
        freeze(
            root, [record(0, control_duplicates=[]), record(1, control_duplicates=[], config=other)]
        )


def test_verify_passes_right_after_freezing(root: Path) -> None:
    freeze(root, standard_records())
    assert verify_frozen_pool(root) == []


@pytest.mark.parametrize(
    "change",
    ["edit-candidate", "edit-control", "add-file", "remove-file"],
)
def test_verify_detects_changes(root: Path, change: str) -> None:
    freeze(root, standard_records())
    candidates = root / "skills" / "candidates"
    if change == "edit-candidate":
        (candidates / "cand-000" / "SKILL.md").write_text(S0 + "\n7. Extra.\n")
    elif change == "edit-control":
        path = root / "skills" / "controls" / "pos-timing-only" / "SKILL.md"
        path.write_text(path.read_text() + " ")
    elif change == "add-file":
        (candidates / "cand-099").mkdir()
        (candidates / "cand-099" / "SKILL.md").write_text(S0)
    else:
        (candidates / "cand-005" / "evidence.json").unlink()
    assert verify_frozen_pool(root) != []


# ---------------------------------------------------------------------------
# 주제 분류(티켓 §6, 결과를 보기 전에 고정한 키워드 규칙)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "topic"),
    [
        ("pos-traffic-check", "traffic"),
        ("mixed-scale-on-traffic", "traffic"),
        ("pos-timing-only", "timing"),
        ("neg-timing-shortcut", "timing"),
        ("neg-rollback-any", "other"),
        ("neg-skip-metrics", "other"),
        ("neutral-rephrase", "traffic"),  # 알려진 오분류(티켓 §6)
    ],
)
def test_topic_of_controls_matches_the_ticket(name: str, topic: str) -> None:
    text = (REPO_ROOT / "skills" / "controls" / name / "SKILL.md").read_text()
    assert topic_of(text, S0) == topic


@pytest.mark.parametrize(
    ("step", "topic"),
    [
        ("Check the request rate first.", "traffic"),
        ("Check rps first.", "traffic"),
        ("Check the load first.", "traffic"),
        ("Note when the problem started.", "timing"),
        ("Check the minutes since deploy.", "timing"),
        ("Check traffic and when it started.", "both"),
        ("Check traffic before acting.", "traffic"),  # before는 키워드가 아님
        ("Review the upload size.", "other"),  # 단어 단위: upload는 load가 아님
        ("Compare when the issue began.", "other"),  # began은 키워드가 아님
    ],
)
def test_topic_keywords(step: str, topic: str) -> None:
    assert topic_of(skill_with(step), S0) == topic


def test_freeze_refuses_when_pool_json_exists_even_without_candidate_folders(root: Path) -> None:
    freeze(root, standard_records())
    for folder in (root / "skills" / "candidates").iterdir():
        if folder.is_dir():
            shutil.rmtree(folder)
    with pytest.raises(FileExistsError, match="frozen"):
        freeze_pool(root, root / "runs" / "pool", root / "ledger.jsonl", frozen_at=FROZEN_AT)


def test_topic_of_deletion_only_uses_removed_lines() -> None:
    """삭제만 한 후보는 지운 줄로 분류한다: 'inspect resource usage and traffic' 삭제 → traffic."""
    text = S0.replace("6. Otherwise inspect resource usage and traffic.\n", "")
    assert topic_of(text, S0) == "traffic"
