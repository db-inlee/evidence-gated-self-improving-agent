"""Ticket 012 §9의 3: 0단계 실행기(`evidencegate.t012.stage0`).

가짜 정책만 쓴다(Ollama·API 호출 없음). 판정 숫자와 산식은 `plans/ticket-012-criteria.json`에서
읽어야 하므로, 기준 사전을 바꾸면 판정도 바뀌는지 함께 본다.
"""

import copy
import hashlib
import io
import json
import os
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evidencegate.bench_runner import run_blocks
from evidencegate.machine import Machine
from evidencegate.policy import PolicyTrace, StopReason, TurnRecord, call_tool
from evidencegate.preflight import Check
from evidencegate.run_plan import Block
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.skills import Skill
from evidencegate.t009.splits import split_cases
from evidencegate.t012 import stage0 as st
from opsworld.agent_env import AgentEnv

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = json.loads((ROOT / "plans" / "ticket-012-criteria.json").read_text())
DEV_A = split_cases("t009_dev_a")
S0P = CRITERIA["skills"]["S0_prime"]["skill_id"]
R1 = CRITERIA["skills"]["S0_prime_r1"]["skill_id"]


# 고정 값과 블록 ---------------------------------------------------------------------------------


def test_criteria_are_read_from_the_frozen_file() -> None:
    assert Path("plans/ticket-012-criteria.json") == st.CRITERIA_PATH
    assert st.CRITERIA_SHA256 == "63a8f73ee8bac878c51007105036aa1bdeaed5d1f8a1678ce3893611efc5fe94"
    assert st.CRITERIA_TAG == "t012-criteria"
    assert st.load_criteria(ROOT) == CRITERIA


def test_a_changed_criteria_file_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "plans").mkdir(parents=True)
    data = (ROOT / st.CRITERIA_PATH).read_bytes()
    (root / st.CRITERIA_PATH).write_bytes(data + b" ")
    (root / "plans" / "ticket-012-criteria.json.sha256").write_text(
        (ROOT / "plans" / "ticket-012-criteria.json.sha256").read_text()
    )
    with pytest.raises(ValueError, match="sha256"):
        st.load_criteria(root)


def test_three_runs_per_skill_use_their_own_split_and_the_same_cases() -> None:
    assert st.RUN_SPLITS == ("t012_stage0", "t012_stage0_rerun1", "t012_stage0_rerun2")
    for key, skill_id in (("S0_prime", S0P), ("S0_prime_r1", R1)):
        blocks = st.stage_blocks(CRITERIA, key)
        assert [b.split for b in blocks] == list(st.RUN_SPLITS)
        for b in blocks:
            assert (b.skill_id, b.repeat, list(b.cases)) == (skill_id, 0, DEV_A)
            assert b.skill_path == CRITERIA["skills"][key]["path"]


# 기준률 --------------------------------------------------------------------------------------


def rec(
    case: tuple[str, str, int],
    *,
    cd: bool = False,
    action: str | None = "observe_only",
    tools: tuple[str, ...] = ("get_service_metrics", "observe_only"),
    skill: str = S0P,
    llm_seed: int | None = None,
    config: dict[str, Any] | None = None,
    format_errors: int = 0,
) -> EpisodeRecord:
    family, level, seed = case
    turns = tuple(TurnRecord(t, {}, "{}", t == "<unparsed>") for t in tools)
    stop = StopReason.ACTION if action else StopReason.MAX_STEPS
    return EpisodeRecord(
        EpisodeKey("p", "c", "t", skill, family, seed, 0, level), config or {"x": 1},
        seed if llm_seed is None else llm_seed, cd, False, action, (), (),
        PolicyTrace(turns, format_errors, stop), "2026-11-23T00:00:00+00:00", 1.0,
    )  # fmt: skip


def run_with(successes: dict[tuple[str, str], int]) -> list[EpisodeRecord]:
    """셀마다 앞의 n개 케이스만 CD 참."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for c in DEV_A:
        cell = (c[0], c[1])
        i = seen.get(cell, 0)
        seen[cell] = i + 1
        out.append(rec(c, cd=i < successes.get(cell, 0)))
    return out


CELLS = sorted({(c[0], c[1]) for c in DEV_A})


@pytest.mark.parametrize(
    ("counts", "passed"),
    [
        ([2, 8, 5, 5, 0, 10], True),  # 범위 안 4셀
        ([1, 9, 5, 5, 0, 10], False),  # 경계 밖 1·9, 범위 안 2셀
        ([2, 2, 2, 0, 0, 0], False),  # 3셀
        ([3, 3, 3, 3, 3, 3], True),
    ],
)
def test_base_rate_counts_cells_in_the_fixed_range(counts: list[int], passed: bool) -> None:
    out = st.base_rate(run_with(dict(zip(CELLS, counts, strict=True))), CRITERIA)
    assert out["pass"] is passed
    assert out["cells"] == {f"{f}/{lv}": n for (f, lv), n in zip(CELLS, counts, strict=True)}


def test_base_rate_numbers_come_from_the_criteria() -> None:
    records = run_with(dict(zip(CELLS, [2, 8, 5, 5, 0, 10], strict=True)))
    stricter = copy.deepcopy(CRITERIA)
    stricter["base_rate"]["cells_needed"] = 5
    assert st.base_rate(records, CRITERIA)["pass"] is True
    assert st.base_rate(records, stricter)["pass"] is False
    narrow = copy.deepcopy(CRITERIA)
    narrow["base_rate"]["cd_successes_min"] = 3
    assert st.base_rate(records, narrow)["in_range"] == 3  # 8, 5, 5


def test_base_rate_refuses_records_that_are_not_the_dev_a_cases() -> None:
    with pytest.raises(ValueError, match="cases"):
        st.base_rate(run_with({})[:-1], CRITERIA)


# 같은 seed 불일치와 k ------------------------------------------------------------------------


def three_runs(differ_in_2: int = 0, differ_in_3: int = 0) -> list[list[EpisodeRecord]]:
    first = [rec(c) for c in DEV_A]
    second = [rec(c, action="rollback_deployment" if i < differ_in_2 else "observe_only")
              for i, c in enumerate(DEV_A)]  # fmt: skip
    third = [
        rec(c, action=None if i < differ_in_3 else "observe_only") for i, c in enumerate(DEV_A)
    ]
    return [first, second, third]


def test_disagreements_count_both_reruns_against_the_first_run() -> None:
    assert st.disagreements(three_runs(2, 3)) == {"pairs": 120, "differing": 5}
    assert st.disagreements(three_runs()) == {"pairs": 120, "differing": 0}


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda r: [r[0], r[1][:-1], r[2]], "Dev-A cases"),
        (lambda r: [r[0], [replace(r[1][0], llm_seed=99), *r[1][1:]], r[2]], "llm_seed"),
        (lambda r: [r[0], r[1], [replace(x, policy_config={"y": 2}) for x in r[2]]], "config"),
    ],
)
def test_disagreements_refuse_unpaired_runs(
    change: Callable[[list[list[EpisodeRecord]]], list[list[EpisodeRecord]]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        st.disagreements(change(three_runs()))


@pytest.mark.parametrize(
    ("differing", "k"), [(0, 8), (3, 10), (5, 12), (10, 17), (16, 21), (24, 26)]
)
def test_k_follows_the_frozen_rule(differing: int, k: int) -> None:
    out = st.k_rule({"pairs": 120, "differing": differing}, CRITERIA)
    assert out["k"] == k
    assert out["p_used"] >= 25 / 600


def test_k_floor_and_cap_come_from_the_criteria() -> None:
    no_floor = copy.deepcopy(CRITERIA)
    no_floor["k_rule"]["floor"] = "0"
    assert st.k_rule({"pairs": 120, "differing": 0}, no_floor)["k"] == 6
    out = st.k_rule({"pairs": 120, "differing": 40}, CRITERIA)
    assert out["k"] > 30 and out["over_cap"] is True
    other = copy.deepcopy(CRITERIA)
    other["k_rule"]["upper_bound"] = "wilson"
    with pytest.raises(ValueError, match="upper_bound"):
        st.k_rule({"pairs": 120, "differing": 0}, other)


# 시도의 판정 ---------------------------------------------------------------------------------

PASS_COUNTS = dict(zip(CELLS, [5, 5, 5, 5, 0, 10], strict=True))
FAIL_COUNTS = dict(zip(CELLS, [0, 0, 10, 10, 0, 0], strict=True))


def runs_from(counts: dict[tuple[str, str], int], differing: int = 0) -> list[list[EpisodeRecord]]:
    first = run_with(counts)
    reruns = [
        [replace(r, action=None) if i < differing and j == 0 else r for i, r in enumerate(first)]
        for j in range(2)
    ]
    return [first, *reruns]


def test_attempt_outcomes() -> None:
    assert st.attempt("S0_prime", runs_from(PASS_COUNTS), CRITERIA)["outcome"] == "pass"
    assert st.attempt("S0_prime", runs_from(FAIL_COUNTS), CRITERIA)["outcome"] == "revise"
    assert st.attempt("S0_prime_r1", runs_from(FAIL_COUNTS), CRITERIA)["outcome"] == "close"
    over = st.attempt("S0_prime", runs_from(PASS_COUNTS, differing=40), CRITERIA)
    assert over["outcome"] == "close" and over["k"]["over_cap"] is True


def test_descriptive_values() -> None:
    first = [
        rec(DEV_A[0], tools=("observe_only",)),  # 첫 호출이 조치
        rec(DEV_A[1], tools=("get_traffic_metrics", "scale_service"), action="scale_service"),
        rec(DEV_A[2], tools=("<unparsed>", "observe_only"), format_errors=1),
        *(rec(c) for c in DEV_A[3:]),
    ]
    out = st.describe_run(first)
    assert out["first_call_is_disposition"] == 1
    assert out["traffic_episodes"] == 1
    assert out["format_error_episodes"] == 1
    assert out["tool_call_counts"] == {"1": 1, "2": 59}
    assert out["stop_reasons"] == {"action": 60}
    assert sum(out["sequences"].values()) == 60


# 실행(가짜 정책) -------------------------------------------------------------------------------


class Fake:
    """스킬(skill_id)마다 정한 행동으로 조치 하나만 한다(예: seed 짝수면 rollback)."""

    policy_id = "ollama:gpt-oss:20b"

    def __init__(self, behave: dict[str, Callable[[AgentEnv], str]]) -> None:
        self.config: dict[str, Any] = {"fake": "stage0"}
        self.behave = behave

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        action = self.behave[skill.skill_id](env)
        _, turn = call_tool(env, action, {"service_id": env.service_id})
        return PolicyTrace((turn,), 0, StopReason.ACTION)


def parity(env: AgentEnv) -> str:
    seed = env._episode.incident.seed  # 테스트 전용: 시드는 숨은 값이 아니다
    return "rollback_deployment" if seed % 2 == 0 else "observe_only"


def always_observe(env: AgentEnv) -> str:
    return "observe_only"


def run_stage(tmp_path: Path, behave: dict[str, Callable[[AgentEnv], str]], **kw: Any) -> Any:
    return st.run_stage0(
        Fake(behave), root=ROOT, runs_dir=tmp_path / "runs", out=kw.pop("out", io.StringIO()),
        criteria=CRITERIA, **kw,
    )  # fmt: skip


def stored(tmp_path: Path) -> dict[str, int]:
    return {
        f"{p.parts[-4]}/{p.stem}": len(p.read_text().splitlines())
        for p in sorted((tmp_path / "runs").glob("t012_stage0*/*/*/*.jsonl"))
    }


def test_a_passing_s0_prime_runs_three_blocks_only(tmp_path: Path) -> None:
    out = io.StringIO()
    result = run_stage(tmp_path, {S0P: parity, R1: always_observe}, out=out)
    assert result.stopped_by is None and result.blocks_run == 3
    assert stored(tmp_path) == {f"{s}/{S0P}": 60 for s in st.RUN_SPLITS}
    assert not re.search(r"correct|evidence_safe|\bCD\b|\bES\b", out.getvalue())


def test_a_failing_s0_prime_runs_the_one_revision(tmp_path: Path) -> None:
    out = io.StringIO()
    result = run_stage(tmp_path, {S0P: always_observe, R1: parity}, out=out)
    assert result.blocks_run == 6
    assert stored(tmp_path) == {
        **{f"{s}/{S0P}": 60 for s in st.RUN_SPLITS},
        **{f"{s}/{R1}": 60 for s in st.RUN_SPLITS},
    }
    assert "revision" in out.getvalue()


def test_stop_file_stops_at_a_block_boundary(tmp_path: Path) -> None:
    stop = tmp_path / "STOP"
    stop.write_text("")
    result = run_stage(tmp_path, {S0P: parity, R1: parity}, stop_file=stop)
    assert result.blocks_run == 0 and result.stopped_by == "stop_file"


def test_the_009_dev_a_records_are_not_touched(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    block = Block(0, "S0", "s0", CRITERIA["skills"]["S0"]["skill_id"], "skills/S0/SKILL.md",
                  "t009_dev_a", 0, tuple(DEV_A))  # fmt: skip
    run_blocks([block], Fake({CRITERIA["skills"]["S0"]["skill_id"]: parity}), root=ROOT,
               runs_dir=runs, out=io.StringIO())  # fmt: skip
    old = sorted((runs / "t009_dev_a").rglob("*.jsonl"))
    before = [(hashlib.sha256(p.read_bytes()).hexdigest(), os.stat(p).st_mtime_ns) for p in old]
    run_stage(tmp_path, {S0P: always_observe, R1: parity})
    after = [(hashlib.sha256(p.read_bytes()).hexdigest(), os.stat(p).st_mtime_ns) for p in old]
    assert before == after and sorted((runs / "t009_dev_a").rglob("*.jsonl")) == old


# 보고와 결과 파일 -----------------------------------------------------------------------------


def report(tmp_path: Path) -> tuple[int, str]:
    out = io.StringIO()
    code = st.report(
        runs_dir=tmp_path / "runs", root=tmp_path / "root", criteria=CRITERIA,
        policy_id=Fake.policy_id, config={"fake": "stage0"}, out=out,
    )  # fmt: skip
    return code, out.getvalue()


def test_report_writes_the_verdict_once_and_deterministically(tmp_path: Path) -> None:
    run_stage(tmp_path, {S0P: always_observe, R1: parity})
    code, text = report(tmp_path)
    assert code == 0, text
    path = tmp_path / "root" / st.RESULT_PATH
    result = json.loads(path.read_text())
    assert result["verdict"] == "pass" and result["skill_for_stage1"] == "S0_prime_r1"
    assert [a["outcome"] for a in result["attempts"]] == ["revise", "pass"]
    assert result["attempts"][1]["k"]["k"] == 8  # 불일치 0 → 아래 한계 25/600
    assert result["criteria_sha256"] == st.CRITERIA_SHA256
    data = path.read_bytes()
    assert report(tmp_path)[0] == 0 and path.read_bytes() == data
    path.write_text("{}\n")
    assert report(tmp_path)[0] == 1


def test_report_closes_when_the_revision_also_fails(tmp_path: Path) -> None:
    run_stage(tmp_path, {S0P: always_observe, R1: always_observe})
    code, _ = report(tmp_path)
    assert code == 0
    result = json.loads((tmp_path / "root" / st.RESULT_PATH).read_text())
    assert result["verdict"] == "close" and result["skill_for_stage1"] is None
    assert [a["outcome"] for a in result["attempts"]] == ["revise", "close"]


def test_report_without_complete_runs_gives_no_verdict(tmp_path: Path) -> None:
    run_stage(tmp_path, {S0P: parity, R1: parity}, max_blocks=2)
    code, text = report(tmp_path)
    assert code == 1 and "incomplete" in text
    assert not (tmp_path / "root" / st.RESULT_PATH).exists()


# 실행 전 점검 ---------------------------------------------------------------------------------


GOOD_POWER = {
    "pmset -g batt": "Now drawing from 'AC Power'\n",
    "ioreg": '"AppleClamshellState" = No\n',
    "pmset -g assertions": (
        'pid 9(caffeinate): [0x1] 00:01:00 PreventUserIdleSystemSleep named: "c"\n'
    ),
}


def good_power(argv: Any) -> str:
    cmd = " ".join(argv)
    return next(v for k, v in GOOD_POWER.items() if cmd.startswith(k))


def preflight(tmp_path: Path, **kw: Any) -> dict[str, Check]:
    checks = st.stage0_preflight(
        root=kw.pop("root", ROOT),
        runs_dir=tmp_path / "runs",
        base_config=kw.pop("config", {"ollama_version": "0.40.3"}),
        machine=Machine("Mac15,3", "Apple M3"),
        git=kw.pop("git", lambda argv: 0),
        power_run=good_power,
        disk_usage=lambda path: type("U", (), {"free": 10**12})(),
    )
    return {c.name: c for c in checks}


def test_preflight_has_the_stage0_checks(tmp_path: Path) -> None:
    checks = preflight(tmp_path)
    for name in ("criteria", "criteria_tag", "skills", "stage0_paths", "ollama_version", "power"):
        assert checks[name].ok, (name, checks[name].detail)
    assert {"engine", "config_id", "machine", "parallel", "disk"} <= set(checks)
    assert not preflight(tmp_path, git=lambda argv: 1)["criteria_tag"].ok


def test_preflight_refuses_a_skill_file_that_differs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for rel in ("plans/ticket-012-criteria.json", "plans/ticket-012-criteria.json.sha256",
                "skills/S0-prime/SKILL.md", "skills/S0-prime-r1/SKILL.md"):  # fmt: skip
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    assert preflight(tmp_path, root=root)["skills"].ok
    (root / "skills/S0-prime/SKILL.md").write_text("changed\n")
    assert not preflight(tmp_path, root=root)["skills"].ok
