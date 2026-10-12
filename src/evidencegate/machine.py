"""실행 장비 자동 감지.

같은 실험의 결과는 한 장비에서만 만든다. 장비 정보는 정책 설정(config)에 들어가므로 장비가
바뀌면 config_id가 바뀌어 결과 경로가 갈린다. 사람이 붙이는 이름 대신 `sysctl` 값을 쓴다.
"""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

CommandRunner = Callable[[list[str]], str]

_SYSCTL_KEYS = ("hw.model", "machdep.cpu.brand_string")


class MachineDetectionError(RuntimeError):
    """장비 정보를 읽지 못했다. 결과가 섞이지 않도록 추측값으로 대신하지 않는다."""


@dataclass(frozen=True)
class Machine:
    hw_model: str  # sysctl -n hw.model, 예: Mac15,3
    chip: str  # sysctl -n machdep.cpu.brand_string, 예: Apple M3

    def as_dict(self) -> dict[str, str]:
        return {"hw_model": self.hw_model, "chip": self.chip}


def detect_machine(run: CommandRunner | None = None) -> Machine:
    runner = run or _run_command
    values = []
    for key in _SYSCTL_KEYS:
        try:
            value = runner(["sysctl", "-n", key]).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise MachineDetectionError(f"cannot read {key}: {error}") from error
        if not value:
            raise MachineDetectionError(f"empty value for {key}")
        values.append(value)
    return Machine(hw_model=values[0], chip=values[1])


def _run_command(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout
