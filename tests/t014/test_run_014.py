"""Ticket 014 실행 CLI(사전 등록 1·8절, 티켓 §11): --max-hours, 재개, 지문, status. LLM 없음."""

import io
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from evidencegate.policy import PolicyTrace, StopReason, call_tool
from evidencegate.preflight import Check
from evidencegate.run_plan import Block
from evidencegate.skills import Skill, load_skill
from evidencegate.t014 import run as cli
from evidencegate.t014.splits import cell_cases
from opsworld.agent_env import AgentEnv

ROOT = Path(__file__).resolve().parents[2]
S0 = load_skill(ROOT / "skills" / "S0" / "SKILL.md")


class FakePolicy:
    """observe_only로 끝내는 가짜 정책(LLM 없음)."""

    policy_id = "fake:policy"

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"ollama_version": "0.40.3", "model_digest": "f38a"}

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
        return PolicyTrace((turn,), 0, StopReason.ACTION)


class Probe:
    parallel = True
    ratio = 1.3
    ratios = (1.3,)
    max_ratio = 1.68


def small_plan() -> list[Block]:
    cells = [("bad_deployment", "L1"), ("bad_deployment", "L2")]
    return [
        Block(
            i,
            "S0",
            "s0",
            S0.skill_id,
            "skills/S0/SKILL.md",
            "t014_calibration",
            0,
            tuple(cell_cases("t014_calibration", c)[:2]),
        )
        for i, c in enumerate(cells)
    ]


def git(tree: str = "t" * 40, dirty: bool = False) -> Any:
    def run(argv: Sequence[str]) -> tuple[int, str]:
        if "status" in argv:
            return 0, " M src/x.py\n" if dirty else ""
        return 0, (tree if argv[-1] == "HEAD:src" else "c" * 40) + "\n"

    return run


def run_cli(tmp_path: Path, args: list[str], **kw: Any) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(
        [*args, "--runs-dir", str(tmp_path / "runs"), "--root", str(ROOT)],
        policy_factory=FakePolicy,
        probe=lambda: Probe(),
        plan=kw.pop("plan", small_plan()),
        plan_sha256="f" * 64,
        prereg_sha256="r" * 64,
        environment_checks=kw.pop("checks", lambda policy: [Check("engine", True, "fake")]),
        power_check=lambda: None,
        ollama_version=lambda: "0.40.3",
        git=kw.pop("git", git()),
        now=kw.pop("now", lambda: datetime.now().astimezone()),
        status_policy=FakePolicy(),
        out=out,
        err=out,
    )
    return code, out.getvalue()


def stage_dir(tmp_path: Path, stage: int = 1) -> Path:
    return tmp_path / "runs" / "ticket014" / f"stage{stage}"


def sessions(tmp_path: Path) -> list[dict[str, Any]]:
    path = stage_dir(tmp_path) / "sessions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_writes_the_fingerprint_and_logs_the_stage(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, ["run", "--stage", "1"])
    assert code == 0, out
    fp = json.loads((stage_dir(tmp_path) / "fingerprint.json").read_text())
    assert fp["stage"] == 1 and fp["code_tree"] == "t" * 40 and fp["plan_sha256"] == "f" * 64
    start = next(e for e in sessions(tmp_path) if e["event"] == "session_start")
    assert start["stage"] == 1 and start["code_commit"] == "c" * 40
    assert sessions(tmp_path)[-1]["reason"] == "plan_done"


def test_resume_skips_finished_blocks(tmp_path: Path) -> None:
    run_cli(tmp_path, ["run", "--stage", "1"])
    code, out = run_cli(tmp_path, ["run", "--stage", "1"])
    assert code == 0 and "0 blocks run, 2 skipped" in out


def test_a_fingerprint_mismatch_exits_2_and_runs_nothing(tmp_path: Path) -> None:
    run_cli(tmp_path, ["run", "--stage", "1"], plan=small_plan()[:1])
    code, out = run_cli(tmp_path, ["run", "--stage", "1"], git=git(tree="u" * 40))
    assert code == 2 and "fingerprint" in out and "code_tree" in out
    assert sessions(tmp_path)[-1]["event"] == "fingerprint_mismatch"
    assert "block 1" not in out


def test_an_earlier_stage_environment_must_match(tmp_path: Path) -> None:
    earlier = stage_dir(tmp_path, 0) / "fingerprint.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_text(json.dumps({"stage": 0, "model_digest": "other"}))
    code, out = run_cli(tmp_path, ["run", "--stage", "1"])
    assert code == 2 and "model_digest: stage 0 recorded other" in out
    assert not (stage_dir(tmp_path) / "fingerprint.json").exists()


def test_a_dirty_code_tree_fails_preflight_without_a_fingerprint(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, ["run", "--stage", "1"], git=git(dirty=True))
    assert code == 1 and "FAIL fingerprint" in out
    assert not (stage_dir(tmp_path) / "fingerprint.json").exists()


def test_failed_checks_write_no_fingerprint(tmp_path: Path) -> None:
    code, _ = run_cli(
        tmp_path, ["run", "--stage", "1"], checks=lambda policy: [Check("power", False, "lid")]
    )
    assert code == 1 and not (stage_dir(tmp_path) / "fingerprint.json").exists()


def test_preflight_compares_but_does_not_write_the_fingerprint(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, ["preflight", "--stage", "1"])
    assert code == 0, out
    assert not (stage_dir(tmp_path) / "fingerprint.json").exists()


def test_max_hours_zero_stops_at_the_first_block_boundary(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, ["run", "--stage", "1", "--max-hours", "0"])
    assert code == 0 and "stop_at" in out
    assert sessions(tmp_path)[-1]["reason"] == "stop_at"
    start = next(e for e in sessions(tmp_path) if e["event"] == "session_start")
    assert start["max_hours"] == 0.0


def test_max_hours_sets_stop_at_from_now(tmp_path: Path) -> None:
    now = datetime.now().astimezone()
    code, _ = run_cli(tmp_path, ["run", "--stage", "1", "--max-hours", "2.5"], now=lambda: now)
    assert code == 0 and sessions(tmp_path)[-1]["reason"] == "plan_done"
    start = next(e for e in sessions(tmp_path) if e["event"] == "session_start")
    assert start["stop_at"] == (now + timedelta(hours=2.5)).isoformat()


def test_status_counts_blocks_and_estimates_without_results(tmp_path: Path) -> None:
    run_cli(tmp_path, ["run", "--stage", "1", "--max-blocks", "1"])
    code, out = run_cli(tmp_path, ["status", "--stage", "1"])
    assert code == 0
    assert "stage 1: blocks 1/2, episodes 2/4" in out
    assert "remaining" in out and "last session: max_blocks" in out
    assert "fingerprint recorded" in out
    assert "correct" not in out.lower() and "evidence_safe" not in out


def test_run_refuses_a_stage_without_episodes(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, ["run", "--stage", "3"])
    assert code == 1 and "stage 3" in out


def test_run_stage_4_reports_the_skip_when_there_are_no_targets(tmp_path: Path) -> None:
    from evidencegate.t014 import stages

    root = tmp_path / "repo"
    for stage in range(3):
        stages.write_result(root, stage, {"go": True})
    stages.write_result(root, 3, {"oracle_targets": {"proposer": [], "controls": False}})
    out = io.StringIO()
    code = cli.main(
        ["run", "--stage", "4", "--root", str(root), "--runs-dir", str(tmp_path / "runs")],
        out=out,
        err=out,
    )
    assert code == 0 and "stage 4 is skipped: no Oracle targets" in out.getvalue()
