"""Ticket 009 CLI(티켓 §4.1).

--stop-at, 정지 파일, Ctrl-C 기록, 세션 시작 필드, 블록마다 전원·버전 점검.
"""

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from evidencegate.policy import PolicyTrace, StopReason, call_tool
from evidencegate.power import PowerCheckError
from evidencegate.preflight import Check
from evidencegate.run_plan import Block
from evidencegate.skills import Skill, load_skill
from evidencegate.t009.bench import block_check, main, next_stop_at
from evidencegate.t009.splits import split_cases
from opsworld.agent_env import AgentEnv

ROOT = Path(__file__).resolve().parents[2]
S0 = load_skill(ROOT / "skills" / "S0" / "SKILL.md")
SEOUL = ZoneInfo("Asia/Seoul")


class FakePolicy:
    """observe_only로 끝내는 가짜 정책(LLM 없음)."""

    policy_id = "fake:policy"

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"type": "fake"}

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
        return PolicyTrace((turn,), 0, StopReason.ACTION)


def test_stop_at_is_the_next_occurrence_of_that_local_time() -> None:
    now = datetime(2026, 11, 5, 22, 30, tzinfo=SEOUL)
    assert next_stop_at("23:15", now) == datetime(2026, 11, 5, 23, 15, tzinfo=SEOUL)
    assert next_stop_at("07:30", now) == datetime(2026, 11, 6, 7, 30, tzinfo=SEOUL)
    assert next_stop_at("22:30", now) == datetime(2026, 11, 6, 22, 30, tzinfo=SEOUL)
    for bad in ("7:3", "24:00", "ab:cd", "12:60"):
        with pytest.raises(ValueError):
            next_stop_at(bad, now)


def test_block_check_runs_power_then_the_ollama_version() -> None:
    calls: list[str] = []

    def power() -> None:
        calls.append("power")

    def version() -> str:
        calls.append("version")
        return "0.40.3"

    block_check(power, version)()
    assert calls == ["power", "version"]
    with pytest.raises(PowerCheckError, match=r"ollama version 0\.41\.0"):
        block_check(power, lambda: "0.41.0")()

    def bad_power() -> None:
        raise PowerCheckError("lid: closed")

    with pytest.raises(PowerCheckError, match="lid"):
        block_check(bad_power, lambda: "0.40.3")()


def small_plan() -> list[Block]:
    cases = tuple(split_cases("t009_calibration")[:2])
    return [
        Block(i, "S0", "s0", S0.skill_id, "skills/S0/SKILL.md", "t009_calibration", r, cases)
        for i, r in enumerate((3, 0))
    ]


class Probe:
    parallel = True
    ratio = 1.3
    ratios = (1.3,)
    max_ratio = 1.68


def run_cli(tmp_path: Path, args: list[str], **kw: Any) -> tuple[int, str]:
    out = io.StringIO()
    code = main(
        ["run", "--phase", "calibration", "--runs-dir", str(tmp_path / "runs"), *args],
        policy_factory=FakePolicy,
        preflight=lambda policy: [Check("engine", True, "fake")],
        probe=lambda: Probe(),
        plan=small_plan(),
        plan_sha256="f" * 64,
        power_check=kw.pop("power_check", lambda: None),
        ollama_version=kw.pop("ollama_version", lambda: "0.40.3"),
        now=kw.pop("now", lambda: datetime(2026, 11, 5, 21, 0, tzinfo=SEOUL)),
        out=out,
        err=out,
        **kw,
    )
    return code, out.getvalue()


def sessions(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "runs" / "ticket009" / "calibration" / "sessions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_logs_to_the_phase_session_file_with_start_fields(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, [])
    assert code == 0
    start = next(e for e in sessions(tmp_path) if e["event"] == "session_start")
    assert start["phase"] == "calibration" and start["plan_sha256"] == "f" * 64
    assert start["ollama_version"] == "0.40.3" and start["stop_at"] is None
    assert sessions(tmp_path)[-1]["reason"] == "plan_done"
    assert "remaining" in out  # 시작 전에 남은 블록과 예상 시간을 출력


def test_default_stop_file_stops_at_the_next_block(tmp_path: Path) -> None:
    stop = tmp_path / "runs" / "ticket009" / "calibration" / "STOP"
    stop.parent.mkdir(parents=True)
    stop.touch()
    code, out = run_cli(tmp_path, [])
    assert code == 0 and "stop_file" in out
    assert sessions(tmp_path)[-1]["reason"] == "stop_file"


def test_stop_at_flag_stops_after_the_time(tmp_path: Path) -> None:
    # 지금보다 1분 앞의 HH:MM은 다음 날 그 시각이다. 그 시각은 지나지 않았으므로 끝까지 돈다.
    # stop_at 계산은 주입한 now를 쓰지만, 블록 경계의 시각 비교는 v1 실행기의 실제 시계를 쓴다.
    # 그래서 now를 실제 현재 시각으로 둔다(고정 날짜면 그 날짜가 지난 뒤 실패한다).
    now = datetime.now(SEOUL).replace(second=0, microsecond=0)
    earlier = now - timedelta(minutes=1)
    code, _ = run_cli(tmp_path, ["--stop-at", earlier.strftime("%H:%M")], now=lambda: now)
    assert code == 0 and sessions(tmp_path)[-1]["reason"] == "plan_done"
    start = next(e for e in sessions(tmp_path) if e["event"] == "session_start")
    assert start["stop_at"].startswith((earlier + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M"))


def test_a_version_change_between_blocks_stops_the_run(tmp_path: Path) -> None:
    versions = iter(["0.40.3", "0.40.3", "0.41.0", "0.41.0"])
    code, out = run_cli(tmp_path, [], ollama_version=lambda: next(versions))
    assert code == 2 and "ollama version 0.41.0" in out
    assert sessions(tmp_path)[-1]["reason"].startswith("power check failed")


def test_ctrl_c_is_logged(tmp_path: Path) -> None:
    class Interrupting(FakePolicy):
        def run(self, *args: Any) -> Any:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        out = io.StringIO()
        main(
            ["run", "--phase", "calibration", "--runs-dir", str(tmp_path / "runs")],
            policy_factory=Interrupting,
            preflight=lambda policy: [Check("engine", True, "fake")],
            probe=lambda: Probe(),
            plan=small_plan(),
            plan_sha256="f" * 64,
            power_check=lambda: None,
            ollama_version=lambda: "0.40.3",
            now=lambda: datetime(2026, 11, 5, 21, 0, tzinfo=UTC),
            out=out,
            err=out,
        )
    assert sessions(tmp_path)[-1]["reason"] == "interrupted"
