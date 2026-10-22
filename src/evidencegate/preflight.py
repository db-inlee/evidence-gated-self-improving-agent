"""세션마다 하는 실행 전 점검 (Ticket 005 §2). 하나라도 실패하면 실행하지 않는다.

| 점검 | 기준 |
| engine | 지금 정책 설정 전체가 S0 Dev-A 기록(Ticket 004)의 정책 설정과 같다. 엔진(model digest,
|        | Ollama 버전, 양자화), 지침 해시, 옵션, 장비, 병렬 설정이 모두 들어 있다 |
| config_id | 지금 정책 설정의 content_id가 7c66ee5bf13c |
| machine | 장비 hw.model이 Mac15,3 |
| parallel | 클라이언트 동시 2, 서버 OLLAMA_NUM_PARALLEL 2(선언값, 실제 병렬은 실행 시 탐침) |
| frozen_pool | `verify_frozen_pool`에 문제 없음 |
| preregistration_7_8 | 7.8 원문 사본의 sha256이 기록과 같음 |
| preregistration_l3 | Ticket 006 사전 등록 사본의 sha256이 기록과 같음(본 실행 전 고정) |
| plan | 계획 파일이 자기 sha256 파일과 같고, 지금 풀로 다시 만든 계획과 바이트까지 같음 |
| disk | 결과 폴더의 여유 공간이 기준 이상 |

지금 정책 설정은 호출자가 넘긴다(실행기가 Ollama에 물어 만든 값). 이 모듈은 정책을 돌리지 않는다.
"""

import hashlib
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencegate.analysis import check_preregistration, load_pool
from evidencegate.eval_constants import (
    CONFIG_ID,
    EXPECTED_HW_MODEL,
    EXPECTED_PARALLEL,
    MIN_FREE_DISK_BYTES,
    POLICY_ID,
    S0_SKILL_ID,
    TOOL_SPEC_ID,
)
from evidencegate.ids import content_id
from evidencegate.machine import Machine
from evidencegate.pool_freeze import verify_frozen_pool
from evidencegate.run_plan import build_plan, plan_bytes
from evidencegate.store import EpisodeStore


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def reference_policy_config(runs_dir: Path) -> dict[str, Any]:
    """S0 Dev-A 기록(Ticket 004)의 정책 설정. 기록 전체가 한 설정이어야 한다."""
    store = EpisodeStore.for_run(
        runs_dir,
        suite="dev_a",
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=S0_SKILL_ID,
    )
    records = EpisodeStore(store.path, repair=False).records()
    if not records:
        raise ValueError(f"no S0 Dev-A records at {store.path}")
    configs = {content_id(r.policy_config) for r in records}
    if len(configs) != 1:
        raise ValueError(f"S0 Dev-A records carry {len(configs)} different policy_config values")
    return dict(records[0].policy_config)


def check_engine(current: Mapping[str, Any], reference: Mapping[str, Any]) -> Check:
    diffs = _diff(current, reference)
    if diffs:
        return Check("engine", False, "differs from S0 Dev-A: " + "; ".join(diffs))
    return Check("engine", True, "same policy_config as S0 Dev-A")


def check_config_id(current: Mapping[str, Any]) -> Check:
    actual = content_id(dict(current))
    return Check("config_id", actual == CONFIG_ID, f"config_id {actual} (expected {CONFIG_ID})")


def check_machine(machine: Machine) -> Check:
    ok = machine.hw_model == EXPECTED_HW_MODEL
    return Check("machine", ok, f"hw_model {machine.hw_model} (expected {EXPECTED_HW_MODEL})")


def check_parallel(current: Mapping[str, Any]) -> Check:
    actual = current.get("parallel")
    ok = actual == EXPECTED_PARALLEL
    return Check("parallel", ok, f"parallel {actual} (expected {EXPECTED_PARALLEL})")


def check_frozen_pool(root: Path) -> Check:
    problems = verify_frozen_pool(root)
    if problems:
        return Check("frozen_pool", False, "; ".join(problems))
    return Check("frozen_pool", True, "candidate and control files match POOL.json")


def check_preregistration_7_8(root: Path) -> Check:
    try:
        sha = check_preregistration(root)
    except ValueError as error:
        return Check("preregistration_7_8", False, str(error))
    return Check("preregistration_7_8", True, f"7.8 sha256 {sha}")


def check_preregistration_l3(root: Path) -> Check:
    hash_file = root / "preregistration" / "ticket-006-l3.sha256"
    copy = root / "preregistration" / "ticket-006-l3-sections.txt"
    if not hash_file.exists() or not copy.exists():
        return Check("preregistration_l3", False, f"missing {hash_file} or {copy}")
    recorded = next(
        (
            line.removeprefix("sha256: ").strip()
            for line in hash_file.read_text().splitlines()
            if line.startswith("sha256: ")
        ),
        None,
    )
    actual = hashlib.sha256(copy.read_bytes()).hexdigest()
    if actual != recorded:
        return Check("preregistration_l3", False, f"L3 copy {actual} does not match {recorded}")
    return Check("preregistration_l3", True, f"L3 sha256 {actual}")


def check_plan(plan_path: Path, root: Path) -> Check:
    sha_path = plan_path.with_name(plan_path.name + ".sha256")
    if not plan_path.exists() or not sha_path.exists():
        return Check("plan", False, f"missing {plan_path} or {sha_path}")
    data = plan_path.read_bytes()
    recorded = sha_path.read_text().split()[0]
    actual = hashlib.sha256(data).hexdigest()
    if actual != recorded:
        return Check("plan", False, f"plan sha256 {actual} does not match recorded {recorded}")
    if data != plan_bytes(build_plan(load_pool(root), root)):
        return Check("plan", False, "plan file differs from a rebuild from the frozen pool")
    return Check("plan", True, f"plan sha256 {actual}")


def check_disk(
    path: Path,
    minimum: int = MIN_FREE_DISK_BYTES,
    usage: Callable[[Path], Any] = shutil.disk_usage,
) -> Check:
    free = int(usage(path).free)
    return Check("disk", free >= minimum, f"{free} bytes free at {path} (need {minimum})")


def run_preflight(
    *,
    root: Path,
    runs_dir: Path,
    current_config: Mapping[str, Any],
    machine: Machine,
    plan_path: Path,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> list[Check]:
    try:
        engine = check_engine(current_config, reference_policy_config(runs_dir))
    except ValueError as error:
        engine = Check("engine", False, f"cannot read the S0 Dev-A reference: {error}")
    return [
        engine,
        check_config_id(current_config),
        check_machine(machine),
        check_parallel(current_config),
        check_frozen_pool(root),
        check_preregistration_7_8(root),
        check_preregistration_l3(root),
        check_plan(plan_path, root),
        check_disk(runs_dir if runs_dir.exists() else root, usage=disk_usage),
    ]


def preflight_ok(checks: Sequence[Check]) -> bool:
    return all(c.ok for c in checks)


def _diff(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = "") -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        name = f"{prefix}{key}"
        if key not in a or key not in b:
            out.append(f"{name} only in {'reference' if key not in a else 'current'}")
        elif isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
            out += _diff(a[key], b[key], f"{name}.")
        elif a[key] != b[key]:
            out.append(f"{name}: {a[key]!r} != {b[key]!r}")
    return out
