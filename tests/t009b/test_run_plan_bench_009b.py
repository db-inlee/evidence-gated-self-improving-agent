"""Ticket 009b 본 실행 계획과 실행기(§5의 4). 009의 계획 형식·실행기를 009b 풀·경로로 쓴다."""

import hashlib
import io
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evidencegate.machine import Machine
from evidencegate.policy import PolicyTrace, StopReason, call_tool
from evidencegate.preflight import Check
from evidencegate.run_plan import Block, plan_bytes
from evidencegate.skills import Skill, load_skill
from evidencegate.t009.run_plan import PoolEntry, build_main_plan
from evidencegate.t009.splits import split_cases
from evidencegate.t009b import bench as b009b
from evidencegate.t009b import pool_freeze
from evidencegate.t009b import run_plan as rp
from opsworld.agent_env import AgentEnv

ROOT = Path(__file__).resolve().parents[2]
S0 = load_skill(ROOT / "skills" / "S0" / "SKILL.md")


def pool_root(tmp_path: Path) -> Path:
    """S0·통제 후보 파일과 가짜 009b 풀(통제 파일을 proposer 후보 자리에 씀)."""
    root = tmp_path / "root"
    shutil.copytree(ROOT / "skills" / "S0", root / "skills" / "S0")
    shutil.copytree(ROOT / "skills" / "controls", root / "skills" / "controls")
    names = ["pos-timing-only", "neg-rollback-any", "neutral-rephrase"]
    entries = []
    for i, name in enumerate(names):
        path = f"skills/controls/{name}/SKILL.md"
        group = "proposer" if i < 2 else "control"
        entries.append(
            {
                "name": f"cand-{i:03d}" if group == "proposer" else name,
                "group": group,
                "skill_id": load_skill(root / path).skill_id,
                "skill_path": path,
                **({"call_index": i} if group == "proposer" else {}),
            }
        )
    pool = root / rp.POOL_PATH
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({"candidates": entries}))
    calls = root / "runs/proposer-009b/pool"
    calls.mkdir(parents=True)
    for i in (0, 1):
        (calls / f"call-{i:03d}.json").write_text(json.dumps({"call_index": i}))
    groups = {
        "stage": pool_freeze.GROUPS_STAGE,
        "labeled_by": "Claude (test)",
        "method": "one group per change intent",
        "n_groups": 2,
        "n_ideas_read": 2,
        "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "groups": {"A": {"label": "a", "calls": [0]}, "B": {"label": "b", "calls": [1]}},
        "call_sha256": {
            str(i): hashlib.sha256((calls / f"call-{i:03d}.json").read_bytes()).hexdigest()
            for i in (0, 1)
        },
    }
    (root / "plans").mkdir()
    (root / rp.GROUPS_PATH).write_text(json.dumps(groups))
    assert pool_freeze.main(["check-groups", "--root", str(root)]) == 0
    return root


def test_paths_are_the_009b_paths() -> None:
    assert Path("skills/candidates-009b/POOL-009b.json") == rp.POOL_PATH
    assert Path("plans/ticket-009b-main-plan.json") == rp.PLAN_PATH


def test_main_plan_uses_the_009_block_rules_with_the_009b_pool(tmp_path: Path) -> None:
    root = pool_root(tmp_path)
    plan = rp.build_main_plan_009b(root)
    pool = [
        PoolEntry(c["name"], c["group"], c["skill_id"], c["skill_path"])
        for c in json.loads((root / rp.POOL_PATH).read_text())["candidates"]
    ]
    assert plan == build_main_plan(pool, root)
    # proposer 2개 x (Dev-B, Oracle) + 통제 1개 x Oracle + S0 Dev-B 1 + S0 Oracle 5
    assert len(plan) == 2 * 2 + 1 + 1 + 5
    assert {b.split for b in plan} == {"t009_dev_b", "t009_oracle"}


def test_main_plan_needs_the_frozen_009b_pool(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="POOL-009b"):
        rp.build_main_plan_009b(tmp_path)


def test_freeze_writes_the_plan_and_refuses_a_different_one(tmp_path: Path) -> None:
    root = pool_root(tmp_path)
    assert rp.main(["--root", str(root)]) == 0
    plan_file = root / rp.PLAN_PATH
    assert plan_file.read_bytes() == plan_bytes(rp.build_main_plan_009b(root))
    assert (root / "plans/ticket-009b-main-plan.json.sha256").read_text().split()[
        1
    ] == plan_file.name
    assert rp.main(["--root", str(root)]) == 0
    plan_file.write_text("[]\n")
    assert rp.main(["--root", str(root)]) == 1


def test_plan_check_compares_with_a_rebuild(tmp_path: Path) -> None:
    root = pool_root(tmp_path)
    rp.main(["--root", str(root)])
    assert rp.check_plan_009b(root / rp.PLAN_PATH, root).ok
    sha = root / "plans/ticket-009b-main-plan.json.sha256"
    sha.write_text("0" * 64 + "  ticket-009b-main-plan.json\n")
    assert not rp.check_plan_009b(root / rp.PLAN_PATH, root).ok


def test_main_run_preflight_is_the_009_nine_with_the_009b_plan_and_three_more(
    tmp_path: Path,
) -> None:
    root = pool_root(tmp_path)
    shutil.copytree(ROOT / "preregistration", root / "preregistration")
    (root / "results/ticket-009").mkdir(parents=True)
    shutil.copy(ROOT / "results/ticket-009/thresholds.json", root / "results/ticket-009/")
    rp.main(["--root", str(root)])

    def power_run(argv: Sequence[str]) -> str:
        return "Now drawing from 'AC Power'"

    checks = b009b.main_run_preflight(
        root=root,
        runs_dir=tmp_path / "runs",
        current_config={"ollama_version": "0.40.3"},
        machine=Machine(hw_model="Mac15,3", chip="Apple M3"),
        power_run=power_run,
        git=lambda argv: 0,
        disk_usage=lambda path: shutil.disk_usage(tmp_path),
    )
    assert [c.name for c in checks] == [
        "engine",
        "config_id",
        "machine",
        "parallel",
        "preregistration_009",
        "plan_009b",
        "ollama_version",
        "power",
        "disk",
        "preregistration_009b",
        "prereg_tag",
        "thresholds",
    ]
    ok = {c.name: c.ok for c in checks}
    assert ok["plan_009b"] and ok["preregistration_009"] and ok["preregistration_009b"]
    assert ok["thresholds"] and ok["prereg_tag"]


class FakePolicy:
    policy_id = "fake:policy"

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"type": "fake"}

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
        return PolicyTrace((turn,), 0, StopReason.ACTION)


class Probe:
    parallel = True
    ratio = 1.3
    ratios = (1.3,)
    max_ratio = 1.68


def test_bench_logs_to_the_009b_session_folder(tmp_path: Path) -> None:
    cases = tuple(split_cases("t009_dev_b")[:2])
    plan = [Block(0, "S0", "s0", S0.skill_id, "skills/S0/SKILL.md", "t009_dev_b", 0, cases)]
    out = io.StringIO()
    code = b009b.main(
        ["run", "--runs-dir", str(tmp_path / "runs")],
        policy_factory=FakePolicy,
        preflight=lambda policy: [Check("engine", True, "fake")],
        probe=lambda: Probe(),
        plan=plan,
        plan_sha256="e" * 64,
        power_check=lambda: None,
        ollama_version=lambda: "0.40.3",
        out=out,
        err=out,
    )
    assert code == 0, out.getvalue()
    log = tmp_path / "runs" / "ticket009b" / "main" / "sessions.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    start = next(e for e in events if e["event"] == "session_start")
    assert start["phase"] == "main" and start["plan_sha256"] == "e" * 64
    assert start["stop_file"].endswith("runs/ticket009b/main/STOP")
    assert events[-1]["reason"] == "plan_done"
    assert not (tmp_path / "runs" / "ticket009").exists()
