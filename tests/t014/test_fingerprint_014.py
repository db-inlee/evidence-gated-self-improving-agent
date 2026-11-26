"""Ticket 014 환경 지문(사전 등록 1절, 티켓 §11.3). LLM 없음."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evidencegate.ids import content_id
from evidencegate.run_plan import Block
from evidencegate.t014 import fingerprint as fp

CONFIG: dict[str, Any] = {
    "ollama_version": "0.40.3",
    "model_digest": "f38aa0c5",
    "model": "gpt-oss:20b",
}


def git(clean: bool = True, tree: str = "t" * 40, head: str = "c" * 40) -> fp.Git:
    def run(argv: Sequence[str]) -> tuple[int, str]:
        if "status" in argv:
            return 0, "" if clean else " M src/evidencegate/x.py\n"
        if argv[-1] == "HEAD:src":
            return 0, tree + "\n"
        if argv[-1] == "HEAD":
            return 0, head + "\n"
        raise AssertionError(argv)

    return run


def plan() -> list[Block]:
    case = ("bad_deployment", "L1", 1)
    return [
        Block(0, "S0", "s0", "bf65e569ae37", "skills/S0/SKILL.md", "t014_dev_b", 0, (case,)),
        Block(1, "cand-1", "proposer", "aaaa", "x/SKILL.md", "t014_dev_b", 0, (case,)),
    ]


def collect(**kw: Any) -> dict[str, Any]:
    return fp.collect(
        stage=kw.pop("stage", 2),
        config=kw.pop("config", CONFIG),
        plan=kw.pop("plan", plan()),
        plan_sha256="p" * 64,
        prereg_sha256="r" * 64,
        git=kw.pop("git", git()),
        root=Path("."),
    )


def test_collect_records_the_preregistered_fields() -> None:
    got = collect()
    assert got["ollama_version"] == "0.40.3" and got["model_digest"] == "f38aa0c5"
    assert got["config_id"] == content_id(dict(CONFIG))
    assert len(got["tool_spec_id"]) == 12 and got["s0_skill_id"] == "bf65e569ae37"
    assert got["skill_ids"] == ["aaaa", "bf65e569ae37"]
    assert got["code_tree"] == "t" * 40 and got["code_commit"] == "c" * 40
    assert got["prereg_sha256"] == "r" * 64 and got["plan_sha256"] == "p" * 64
    assert got["stage"] == 2


def test_a_dirty_code_tree_is_refused() -> None:
    with pytest.raises(fp.FingerprintError, match="uncommitted"):
        collect(git=git(clean=False))


def test_first_run_writes_and_later_runs_compare(tmp_path: Path) -> None:
    path = tmp_path / "fingerprint.json"
    assert fp.ensure(path, collect()) == []
    first = path.read_bytes()
    assert fp.ensure(path, collect()) == []
    assert path.read_bytes() == first  # 고치지 않는다
    changed = collect(config={**CONFIG, "ollama_version": "0.41.0"})
    problems = fp.ensure(path, changed)
    assert any("ollama_version" in p for p in problems)
    assert path.read_bytes() == first


def test_code_commit_alone_may_change_but_the_code_tree_may_not(tmp_path: Path) -> None:
    path = tmp_path / "fingerprint.json"
    fp.ensure(path, collect())
    assert fp.ensure(path, collect(git=git(head="d" * 40))) == []
    problems = fp.ensure(path, collect(git=git(tree="u" * 40)))
    assert problems == [f"code_tree: recorded {'t' * 40}, now {'u' * 40}"]


def test_a_later_stage_must_share_the_environment_of_earlier_stages(tmp_path: Path) -> None:
    earlier = tmp_path / "stage0" / "fingerprint.json"
    earlier.parent.mkdir()
    earlier.write_text(json.dumps(collect(stage=0)))
    now = collect(stage=2, config={**CONFIG, "model_digest": "other"})
    problems = fp.compare_environment([earlier], now)
    # digest는 정책 설정에도 들어 있어 config_id도 함께 달라진다
    assert problems[0] == "model_digest: stage 0 recorded f38aa0c5, now other"
    assert [p.split(":")[0] for p in problems] == ["model_digest", "config_id"]
    assert fp.compare_environment([earlier], collect(stage=2)) == []
