"""실행기의 세션 단위 실행과 전원 점검 선택 인자(Ticket 009 원칙 9, §4.1).

기본값(인자 없음)이면 v1과 같다. v1 테스트(test_bench_runner.py)가 그대로 통과하고,
같은 가짜 정책에서 기록과 세션 기록이 인자 없이 돌린 것과 같다.
"""

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from evidencegate.bench_runner import BenchStopped, main, run_blocks
from evidencegate.power import PowerCheckError
from evidencegate.preflight import Check
from test_bench_runner import FAST, REPO_ROOT, FakePolicy, small_plan, stored


def go(tmp_path: Path, policy: FakePolicy, **kw: Any) -> Any:
    kw.setdefault("rules", FAST)
    return run_blocks(
        small_plan(n=2),
        policy,
        root=REPO_ROOT,
        runs_dir=tmp_path / "runs",
        out=io.StringIO(),
        **kw,
    )


def session_log(tmp_path: Path, name: str = "ticket005/sessions.jsonl") -> list[dict[str, Any]]:
    path = tmp_path / "runs" / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def without_time(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in e.items() if k != "time"} for e in entries]


def _strip(records: list[Any]) -> list[Any]:
    return [(r.key, r.correct_disposition, r.action) for r in records]


def test_defaults_match_v1_records_and_session_log(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    go(a, FakePolicy())
    go(b, FakePolicy(), check_power=None, stop_at=None, stop_file=None, log_interrupt=False)
    plan = small_plan(n=2)
    assert _strip(stored(a, plan, FakePolicy())) == _strip(stored(b, plan, FakePolicy()))
    assert without_time(session_log(a)) == without_time(session_log(b))


class Counter:
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self) -> None:
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise PowerCheckError("lid: closed")


def test_power_is_checked_at_start_and_after_every_block_that_ran(tmp_path: Path) -> None:
    check = Counter()
    summary = go(tmp_path, FakePolicy(), check_power=check)
    assert summary.blocks_run == 3 and check.calls == 1 + 3
    again = Counter()
    go(tmp_path, FakePolicy(), check_power=again)  # 할 일 없음: 시작 점검만
    assert again.calls == 1


def test_power_failure_at_start_runs_nothing(tmp_path: Path) -> None:
    with pytest.raises(BenchStopped, match="power check failed"):
        go(tmp_path, FakePolicy(), check_power=Counter(fail_at=1))
    assert stored(tmp_path, small_plan(n=2), FakePolicy()) == []
    assert session_log(tmp_path)[-1]["reason"].startswith("power check failed")


def test_power_failure_after_a_block_keeps_that_block_and_stops(tmp_path: Path) -> None:
    with pytest.raises(BenchStopped, match="lid"):
        go(tmp_path, FakePolicy(), check_power=Counter(fail_at=2))
    assert len(stored(tmp_path, small_plan(n=2), FakePolicy())) == 2  # 첫 블록만


def test_stop_file_stops_at_the_next_block_boundary(tmp_path: Path) -> None:
    stop = tmp_path / "STOP"
    policy = FakePolicy()
    original = policy.run

    def run_then_touch(*args: Any) -> Any:
        stop.touch()  # 첫 블록 도중에 정지 파일이 생김
        return original(*args)

    policy.run = run_then_touch  # type: ignore[method-assign, assignment]
    summary = go(tmp_path, policy, stop_file=stop)
    assert summary.blocks_run == 1 and summary.stopped_by == "stop_file"
    assert len(stored(tmp_path, small_plan(n=2), FakePolicy())) == 2  # 블록은 끝까지
    assert session_log(tmp_path)[-1] == {**session_log(tmp_path)[-1], "reason": "stop_file"}


def test_stop_at_stops_before_a_block_once_the_time_has_passed(tmp_path: Path) -> None:
    now = datetime(2026, 11, 5, 7, 59, tzinfo=UTC)
    ticks = iter([now, now + timedelta(minutes=2)] + [now + timedelta(hours=1)] * 10)
    summary = go(
        tmp_path, FakePolicy(), stop_at=now + timedelta(minutes=1), clock=lambda: next(ticks)
    )
    assert summary.blocks_run == 1 and summary.stopped_by == "stop_at"
    assert session_log(tmp_path)[-1]["reason"] == "stop_at"


def test_a_stop_on_an_already_finished_plan_reports_plan_done(tmp_path: Path) -> None:
    go(tmp_path, FakePolicy())
    stop = tmp_path / "STOP"
    stop.touch()
    summary = go(tmp_path, FakePolicy(), stop_file=stop)
    assert summary.stopped_by is None and session_log(tmp_path)[-1]["reason"] == "plan_done"


def test_ctrl_c_is_logged_only_when_asked(tmp_path: Path) -> None:
    def interrupt(*args: Any) -> Any:
        raise KeyboardInterrupt

    for log_interrupt, a in ((False, tmp_path / "x"), (True, tmp_path / "y")):
        policy = FakePolicy()
        policy.run = interrupt  # type: ignore[method-assign, assignment]
        with pytest.raises(KeyboardInterrupt):
            go(a, policy, log_interrupt=log_interrupt)
        last = session_log(a)[-1]
        if log_interrupt:
            assert last["event"] == "session_end" and last["reason"] == "interrupted"
        else:
            assert last["event"] == "session_start"  # v1: 끝 기록 없음


def test_session_start_can_carry_extra_fields(tmp_path: Path) -> None:
    go(tmp_path, FakePolicy(), session_fields={"phase": "calibration", "ollama_version": "0.40.3"})
    first = session_log(tmp_path)[0]
    assert first["event"] == "session_start" and first["phase"] == "calibration"


def test_main_passes_the_new_options_through(tmp_path: Path) -> None:
    out = io.StringIO()
    stop = tmp_path / "STOP"
    stop.touch()
    code = main(
        ["run", "--runs-dir", str(tmp_path / "runs")],
        policy_factory=FakePolicy,
        preflight=lambda policy: [Check("engine", True, "fake")],
        probe=lambda: type(
            "P", (), {"parallel": True, "ratio": 1.3, "ratios": (1.3,), "max_ratio": 1.68}
        )(),
        plan=small_plan(n=2),
        check_power=Counter(),
        stop_file=stop,
        out=out,
        err=out,
    )
    assert code == 0 and "stopped at a block boundary: stop_file" in out.getvalue()
