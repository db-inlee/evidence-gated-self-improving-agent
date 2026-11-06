"""Ticket 009b 본 실행 CLI(티켓 §5의 4).

009 실행기(`t009.bench.main`)에 009b 계획·기록 폴더·점검을 넘긴다.

    PYTHONUNBUFFERED=1 caffeinate -dims uv run python -m evidencegate.t009b.bench run \\
        [--stop-at HH:MM] [--stop-file PATH] [--max-blocks N]
    uv run python -m evidencegate.t009b.bench status
    uv run python -m evidencegate.t009b.bench preflight

- 단계는 main 하나다. 계획 `plans/ticket-009b-main-plan.json`, 세션 기록
  `runs/ticket009b/main/sessions.jsonl`, 정지 파일 `runs/ticket009b/main/STOP`.
- 실행 전 점검(`main_run_preflight`): 009의 9개(계획 점검은 009b 계획 `plan_009b`로 바꿈)에
  009b 사전 등록, `t009b-prereg` 태그, h·δ 파일 점검을 더한다.
  블록 점검(전원·Ollama 버전), 블록 경계 멈춤, Ctrl-C 기록은 009와 같다.
"""

import argparse
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidencegate.machine import Machine, detect_machine
from evidencegate.policy import Policy
from evidencegate.power import Run, system_run
from evidencegate.preflight import Check
from evidencegate.t009 import bench as b009
from evidencegate.t009.preflight import run_preflight_009
from evidencegate.t009b.preflight import Git, main_run_extra_checks, system_git
from evidencegate.t009b.run_plan import PLAN_PATH, check_plan_009b

LOG_DIR = "ticket009b"


def main_run_preflight(
    *,
    root: Path,
    runs_dir: Path,
    current_config: Mapping[str, Any],
    machine: Machine,
    power_run: Run = system_run,
    git: Git = system_git,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> list[Check]:
    checks = run_preflight_009(
        root=root,
        code_root=root,
        runs_dir=runs_dir,
        current_config=current_config,
        machine=machine,
        phase="main",
        plan_path=root / PLAN_PATH,
        power_run=power_run,
        disk_usage=disk_usage,
    )
    plan = check_plan_009b(root / PLAN_PATH, root)
    checks = [plan if c.name == "plan_009" else c for c in checks]
    return checks + main_run_extra_checks(root, git=git)


def _factory(args: argparse.Namespace, plan_file: Path) -> Callable[[Policy], list[Check]]:
    def preflight(policy: Policy) -> list[Check]:
        return main_run_preflight(
            root=args.root,
            runs_dir=args.runs_dir,
            current_config=policy.config,
            machine=detect_machine(),
        )

    return preflight


def main(argv: Sequence[str] | None = None, **kwargs: Any) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv:
        argv = ["--help"]
    command, rest = argv[0], argv[1:]
    return b009.main(
        [command, "--phase", "main", *rest],
        plan_file=PLAN_PATH,
        log_dir=LOG_DIR,
        preflight_factory=_factory,
        **kwargs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
