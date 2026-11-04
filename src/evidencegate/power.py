"""실행 환경 자동 점검(Ticket 009 원칙 9, Ticket 006·008 교훈). 하나라도 어긋나면 멈춘다.

- 전원: `pmset -g batt`에 `Now drawing from 'AC Power'`
- 덮개: `ioreg -r -k AppleClamshellState`에 `"AppleClamshellState" = No`
- 잠자기 방지: `pmset -g assertions`의 프로세스 목록에서 `caffeinate`가 가진
  `PreventSystemSleep` 또는 `PreventUserIdleSystemSleep`. 시스템 합계는 다른 프로세스가 잠깐 잡은
  것일 수 있어 근거로 쓰지 않는다.
- 명령 실패(없음, 비정상 종료, 시간 초과)와 읽을 수 없는 출력도 어긋남이다.

`run`(명령 -> 표준 출력)을 바꿔 끼워 가짜 출력으로 시험한다.
"""

import re
import subprocess
from collections.abc import Callable, Sequence

Run = Callable[[Sequence[str]], str]

BATT = ("pmset", "-g", "batt")
LID = ("ioreg", "-r", "-k", "AppleClamshellState")
ASSERTIONS = ("pmset", "-g", "assertions")
COMMAND_TIMEOUT_S = 30
_CAFFEINATE = re.compile(
    r"pid \d+\(caffeinate\):.*\b(PreventSystemSleep|PreventUserIdleSystemSleep)\b"
)


class PowerCheckError(RuntimeError):
    pass


def system_run(argv: Sequence[str]) -> str:
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=True, timeout=COMMAND_TIMEOUT_S
    ).stdout


def power_problems(run: Run = system_run) -> list[str]:
    try:
        batt, lid, assertions = run(BATT), run(LID), run(ASSERTIONS)
    except (OSError, subprocess.SubprocessError) as error:
        return [f"command: {error}"]
    problems = []
    if "Now drawing from 'AC Power'" not in batt:
        problems.append("power: not on AC power")
    if '"AppleClamshellState" = No' not in lid:
        problems.append("lid: closed or unknown")
    if not _CAFFEINATE.search(assertions):
        problems.append("sleep: no caffeinate sleep-prevention assertion")
    return problems


def check_power(run: Run = system_run) -> None:
    problems = power_problems(run)
    if problems:
        raise PowerCheckError("; ".join(problems))
