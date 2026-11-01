"""원칙 7 실행 환경 자동 점검(Ticket 006 교훈). 하나라도 어긋나면 멈춘다.

- 전원: `pmset -g batt` 첫 줄이 `Now drawing from 'AC Power'`
- 덮개: `ioreg -r -k AppleClamshellState`에서 `"AppleClamshellState" = No`
- 잠자기 방지: `pmset -g assertions`의 프로세스 목록에 `caffeinate`가 가진
  `PreventSystemSleep` 또는 `PreventUserIdleSystemSleep`가 있다. 시스템 합계(예: Handoff가 잠깐
  잡는 assertion)는 실행 내내 유지된다는 보장이 없어 근거로 쓰지 않는다.
- 명령 실패나 읽을 수 없는 출력도 어긋남으로 본다.
"""

import re
import subprocess
from collections.abc import Callable, Sequence

Run = Callable[[Sequence[str]], str]

BATT = ("pmset", "-g", "batt")
LID = ("ioreg", "-r", "-k", "AppleClamshellState")
ASSERTIONS = ("pmset", "-g", "assertions")
_CAFFEINATE = re.compile(
    r"pid \d+\(caffeinate\):.*\b(PreventSystemSleep|PreventUserIdleSystemSleep)\b"
)


class PowerCheckError(RuntimeError):
    pass


def _system_run(argv: Sequence[str]) -> str:
    return subprocess.run(list(argv), capture_output=True, text=True, check=True, timeout=30).stdout


def power_problems(run: Run = _system_run) -> list[str]:
    problems = []
    try:
        batt = run(BATT)
        lid = run(LID)
        assertions = run(ASSERTIONS)
    except (OSError, subprocess.SubprocessError) as error:
        return [f"command: {error}"]
    if "Now drawing from 'AC Power'" not in batt:
        problems.append("power: not on AC power")
    if '"AppleClamshellState" = No' not in lid:
        problems.append("lid: closed or unknown")
    if not _CAFFEINATE.search(assertions):
        problems.append("sleep: no caffeinate sleep-prevention assertion")
    return problems


def check_power(run: Run = _system_run) -> None:
    problems = power_problems(run)
    if problems:
        raise PowerCheckError("; ".join(problems))
