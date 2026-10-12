"""장비 자동 감지(sysctl)."""

import pytest

from evidencegate.machine import Machine, MachineDetectionError, detect_machine


def fake_sysctl(values: dict[str, str]) -> "FakeRun":
    return FakeRun(values)


class FakeRun:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.calls.append(command)
        return self.values[command[-1]]


def test_detects_hw_model_and_chip_with_sysctl() -> None:
    run = FakeRun({"hw.model": "Mac15,3\n", "machdep.cpu.brand_string": "Apple M3\n"})
    assert detect_machine(run) == Machine(hw_model="Mac15,3", chip="Apple M3")
    assert run.calls == [
        ["sysctl", "-n", "hw.model"],
        ["sysctl", "-n", "machdep.cpu.brand_string"],
    ]


def test_empty_output_is_an_error() -> None:
    run = FakeRun({"hw.model": "Mac15,3", "machdep.cpu.brand_string": "  \n"})
    with pytest.raises(MachineDetectionError, match=r"machdep\.cpu\.brand_string"):
        detect_machine(run)


def test_failing_command_is_an_error() -> None:
    def broken(command: list[str]) -> str:
        raise OSError("sysctl not found")

    with pytest.raises(MachineDetectionError, match="sysctl not found"):
        detect_machine(broken)


def test_machine_as_dict() -> None:
    assert Machine("Mac15,3", "Apple M3").as_dict() == {"hw_model": "Mac15,3", "chip": "Apple M3"}


def test_default_runner_works_on_this_mac() -> None:
    """실제 sysctl을 부른다. macOS가 아니면 건너뛴다."""
    import platform

    if platform.system() != "Darwin":
        pytest.skip("sysctl keys are macOS-specific")
    machine = detect_machine()
    assert machine.hw_model and machine.chip
