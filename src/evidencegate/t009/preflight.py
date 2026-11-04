"""Ticket 009 세션 시작 점검(티켓 원칙 2·9, §4.1).

v1 `preflight`는 고치지 않고 그 함수와 형식을 쓴다.

| 점검 | 기준 |
| engine | 지금 정책 설정 전체가 v1 S0 Dev-A 기록(Ticket 004)의 정책 설정과 같다 |
| config_id | 7c66ee5bf13c |
| machine | hw.model Mac15,3 |
| parallel | 클라이언트 동시 2, 서버 OLLAMA_NUM_PARALLEL 2(선언값, 실제 병렬은 실행 시 탐침) |
| preregistration_009 | `ticket-009.txt` 파일 전체의 sha256이 `ticket-009.sha256` 기록과 같음 |
| plan_009 | 그 단계의 계획 파일이 자기 sha256 파일과 같고, 지금 다시 만든 계획과 바이트까지 같음 |
| ollama_version | 정책 설정의 Ollama 버전이 0.40.3 |
| power | 전원 AC, 덮개 열림, caffeinate 잠자기 방지(`evidencegate.power`) |
| disk | 기록 폴더 여유 공간이 기준 이상 |
"""

import hashlib
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from evidencegate.machine import Machine
from evidencegate.power import Run, power_problems, system_run
from evidencegate.preflight import (
    Check,
    check_config_id,
    check_disk,
    check_engine,
    check_machine,
    check_parallel,
    reference_policy_config,
)
from evidencegate.run_plan import plan_bytes
from evidencegate.t009.run_plan import build_phase_plan

PREREG_FILE = Path("preregistration/ticket-009.txt")
PREREG_SHA_FILE = Path("preregistration/ticket-009.sha256")
EXPECTED_OLLAMA_VERSION = "0.40.3"  # 사전 등록 1절


def recorded_sha256(sha_file: Path) -> str | None:
    for line in sha_file.read_text().splitlines():
        if line.startswith("sha256: "):
            return line.removeprefix("sha256: ").strip()
    return None


def check_preregistration_009(root: Path) -> Check:
    text, sha_file = root / PREREG_FILE, root / PREREG_SHA_FILE
    if not text.exists() or not sha_file.exists():
        return Check("preregistration_009", False, f"missing {text} or {sha_file}")
    recorded = recorded_sha256(sha_file)
    actual = hashlib.sha256(text.read_bytes()).hexdigest()
    if actual != recorded:
        return Check("preregistration_009", False, f"009 sha256 {actual} does not match {recorded}")
    return Check("preregistration_009", True, f"009 sha256 {actual}")


def check_plan_009(plan_path: Path, phase: str, code_root: Path) -> Check:
    sha_path = plan_path.with_name(plan_path.name + ".sha256")
    if not plan_path.exists() or not sha_path.exists():
        return Check("plan_009", False, f"missing {plan_path} or {sha_path}")
    data = plan_path.read_bytes()
    recorded = sha_path.read_text().split()[0]
    actual = hashlib.sha256(data).hexdigest()
    if actual != recorded:
        return Check("plan_009", False, f"plan sha256 {actual} does not match recorded {recorded}")
    try:
        rebuilt = plan_bytes(build_phase_plan(phase, code_root))
    except (FileNotFoundError, ValueError) as error:
        return Check("plan_009", False, f"cannot rebuild the {phase} plan: {error}")
    if data != rebuilt:
        return Check("plan_009", False, f"{phase} plan file differs from a rebuild")
    return Check("plan_009", True, f"{phase} plan sha256 {actual}")


def check_ollama_version(current: Mapping[str, Any]) -> Check:
    actual = current.get("ollama_version")
    ok = actual == EXPECTED_OLLAMA_VERSION
    return Check("ollama_version", ok, f"ollama {actual} (expected {EXPECTED_OLLAMA_VERSION})")


def check_power_now(run: Run = system_run) -> Check:
    problems = power_problems(run)
    return Check("power", not problems, "; ".join(problems) or "AC power, lid open, caffeinate")


def run_preflight_009(
    *,
    root: Path,
    code_root: Path,
    runs_dir: Path,
    current_config: Mapping[str, Any],
    machine: Machine,
    phase: str,
    plan_path: Path,
    power_run: Run = system_run,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> list[Check]:
    """root는 사전 등록·계획 파일이 있는 곳, code_root는 계획을 다시 만들 때 읽는 스킬 파일이
    있는 곳이다."""
    try:
        engine = check_engine(current_config, reference_policy_config(runs_dir))
    except ValueError as error:
        engine = Check("engine", False, f"cannot read the S0 Dev-A reference: {error}")
    return [
        engine,
        check_config_id(current_config),
        check_machine(machine),
        check_parallel(current_config),
        check_preregistration_009(root),
        check_plan_009(plan_path, phase, code_root),
        check_ollama_version(current_config),
        check_power_now(power_run),
        check_disk(runs_dir if runs_dir.exists() else root, usage=disk_usage),
    ]
