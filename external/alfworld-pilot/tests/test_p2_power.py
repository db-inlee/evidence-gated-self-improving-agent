"""원칙 7 실행 환경 자동 점검. 가짜 명령 출력으로 실패 경우마다 멈추는지 본다."""

from collections.abc import Sequence

import pytest

from alfworld_pilot.power import PowerCheckError, check_power, power_problems

BATT_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=1)\t71%; AC attached; not charging present: true\n"
)
BATT_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=1)\t71%; discharging; 5:00 remaining present: true\n"
)
LID_OPEN = '  |   "AppleClamshellCausesSleep" = Yes\n  |   "AppleClamshellState" = No\n'
LID_CLOSED = '  |   "AppleClamshellCausesSleep" = Yes\n  |   "AppleClamshellState" = Yes\n'
ASSERT_CAFFEINATE = (
    "Assertion status system-wide:\n   PreventUserIdleSystemSleep    1\n"
    "Listed by owning process:\n"
    '   pid 999(caffeinate): [0x1] 00:10:00 PreventUserIdleSystemSleep named: "caffeinate '
    'command-line tool"\n'
    '   pid 667(sharingd): [0x2] 00:01:04 PreventUserIdleSystemSleep named: "Handoff"\n'
)
ASSERT_HANDOFF_ONLY = (
    "Assertion status system-wide:\n   PreventUserIdleSystemSleep    1\n"
    "Listed by owning process:\n"
    '   pid 667(sharingd): [0x2] 00:01:04 PreventUserIdleSystemSleep named: "Handoff"\n'
    "\tTimeout will fire in 59 secs Action=TimeoutActionTurnOff\n"
)
ASSERT_DISPLAY_ONLY = (
    "Listed by owning process:\n"
    '   pid 999(caffeinate): [0x1] 00:10:00 PreventUserIdleDisplaySleep named: "caffeinate"\n'
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


def test_all_good_passes_and_runs_the_three_commands() -> None:
    run = fake(BATT_AC, LID_OPEN, ASSERT_CAFFEINATE)
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
        (BATT_BATTERY, LID_OPEN, ASSERT_CAFFEINATE, "power"),
        (BATT_AC, LID_CLOSED, ASSERT_CAFFEINATE, "lid"),
        (BATT_AC, LID_OPEN, ASSERT_HANDOFF_ONLY, "sleep"),
        (BATT_AC, LID_OPEN, ASSERT_DISPLAY_ONLY, "sleep"),
        ("", LID_OPEN, ASSERT_CAFFEINATE, "power"),  # 읽을 수 없는 출력도 멈춘다
        (BATT_AC, "", ASSERT_CAFFEINATE, "lid"),
    ],
)
def test_each_failure_stops(batt: str, lid: str, assertions: str, problem: str) -> None:
    run = fake(batt, lid, assertions)
    problems = power_problems(run)
    assert [p.split(":")[0] for p in problems] == [problem]
    with pytest.raises(PowerCheckError, match=problem):
        check_power(run)


def test_command_failure_stops() -> None:
    def broken(argv: Sequence[str]) -> str:
        raise OSError("no pmset")

    with pytest.raises(PowerCheckError):
        check_power(broken)
