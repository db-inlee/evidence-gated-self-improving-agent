"""실행 환경 자동 점검(Ticket 009 원칙 9). 가짜 명령 출력으로 실패 경우마다 멈추는지 본다."""

import subprocess
from collections.abc import Sequence

import pytest

from evidencegate.power import PowerCheckError, check_power, power_problems

BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t71%; AC attached\n"
BATT_BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t71%; discharging\n"
LID_OPEN = '  |   "AppleClamshellCausesSleep" = Yes\n  |   "AppleClamshellState" = No\n'
LID_CLOSED = '  |   "AppleClamshellCausesSleep" = Yes\n  |   "AppleClamshellState" = Yes\n'
CAFFEINATE = (
    "Listed by owning process:\n"
    '   pid 99(caffeinate): [0x1] 00:10:00 PreventUserIdleSystemSleep named: "caffeinate"\n'
)
CAFFEINATE_SYSTEM = (
    "Listed by owning process:\n"
    '   pid 99(caffeinate): [0x1] 00:10:00 PreventSystemSleep named: "caffeinate"\n'
)
HANDOFF_ONLY = (
    "Assertion status system-wide:\n   PreventUserIdleSystemSleep    1\n"
    "Listed by owning process:\n"
    '   pid 667(sharingd): [0x2] 00:01:04 PreventUserIdleSystemSleep named: "Handoff"\n'
)
DISPLAY_ONLY = (
    "Listed by owning process:\n"
    '   pid 99(caffeinate): [0x1] 00:10:00 PreventUserIdleDisplaySleep named: "caffeinate"\n'
)


def fake(batt: str, lid: str, assertions: str) -> "FakeRun":
    return FakeRun({"pmset -g batt": batt, "ioreg": lid, "pmset -g assertions": assertions})


class FakeRun:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def __call__(self, argv: Sequence[str]) -> str:
        cmd = " ".join(argv)
        self.calls.append(cmd)
        for key, out in self.outputs.items():
            if cmd.startswith(key):
                return out
        raise AssertionError(cmd)


@pytest.mark.parametrize("assertions", [CAFFEINATE, CAFFEINATE_SYSTEM])
def test_all_good_passes(assertions: str) -> None:
    run = fake(BATT_AC, LID_OPEN, assertions)
    assert power_problems(run) == []
    check_power(run)
    assert run.calls[:3] == [
        "pmset -g batt",
        "ioreg -r -k AppleClamshellState",
        "pmset -g assertions",
    ]


@pytest.mark.parametrize(
    ("batt", "lid", "assertions", "problem"),
    [
        (BATT_BATTERY, LID_OPEN, CAFFEINATE, "power"),
        (BATT_AC, LID_CLOSED, CAFFEINATE, "lid"),
        (BATT_AC, LID_OPEN, HANDOFF_ONLY, "sleep"),
        (BATT_AC, LID_OPEN, DISPLAY_ONLY, "sleep"),
        ("", LID_OPEN, CAFFEINATE, "power"),  # 읽을 수 없는 출력
        (BATT_AC, "garbage", CAFFEINATE, "lid"),
        (BATT_AC, LID_OPEN, "", "sleep"),
    ],
)
def test_each_failure_is_reported_and_stops(
    batt: str, lid: str, assertions: str, problem: str
) -> None:
    run = fake(batt, lid, assertions)
    assert [p.split(":")[0] for p in power_problems(run)] == [problem]
    with pytest.raises(PowerCheckError, match=problem):
        check_power(run)


def test_all_problems_are_listed_together() -> None:
    problems = power_problems(fake(BATT_BATTERY, LID_CLOSED, HANDOFF_ONLY))
    assert [p.split(":")[0] for p in problems] == ["power", "lid", "sleep"]


@pytest.mark.parametrize(
    "error",
    [
        OSError("no pmset"),
        subprocess.CalledProcessError(1, "pmset"),
        subprocess.TimeoutExpired("pmset", 30),
    ],
)
def test_command_failure_stops(error: Exception) -> None:
    def broken(argv: Sequence[str]) -> str:
        raise error

    assert [p.split(":")[0] for p in power_problems(broken)] == ["command"]
    with pytest.raises(PowerCheckError, match="command"):
        check_power(broken)
