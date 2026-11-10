"""Ticket 009 실행기 CLI(티켓 §4.1). v1 `bench_runner.main`을 그대로 쓰고 009 설정만 넘긴다.

    PYTHONUNBUFFERED=1 caffeinate -dims uv run python -m evidencegate.t009.bench run \\
        --phase calibration [--stop-at HH:MM] [--stop-file PATH] [--max-blocks N]
    uv run python -m evidencegate.t009.bench status --phase calibration
    uv run python -m evidencegate.t009.bench preflight --phase calibration

- 계획: `plans/ticket-009-<단계>-plan.json`(동결된 파일).
- 세션 기록: `runs/ticket009/<단계>/sessions.jsonl`.
- 실행 전 점검은 `t009.preflight.run_preflight_009`(엔진, config, 장비, 병렬, 009 사전 등록, 그 단계
  계획, Ollama 버전, 전원, 디스크). 그다음 v1과 같은 병렬 탐침을 한다.
- 블록 점검(`run_blocks`의 `check_power` 자리): 시작할 때와 블록이 끝날 때마다 전원 점검과 Ollama
  버전 확인(0.40.3)을 함께 한다. 하나라도 어긋나면 종료 코드 2로 멈춘다.
- 블록 경계 멈춤: `--stop-at HH:MM`(이 맥의 현지 시각, 이미 지났으면 다음 날)이나 정지 파일(기본
  `runs/ticket009/<단계>/STOP`)이 있으면 다음 블록을 시작하지 않고 끝낸다. 진행 중인 블록은 끝까지
  돈다. Ctrl-C는 끝 기록 "interrupted"로 남는다. 다시 같은 명령을 내면 끝난 케이스는 건너뛴다.
- 시작 기록에 단계, 계획 sha256, Ollama 버전, stop_at, 정지 파일 경로를 남긴다.
"""

import argparse
import hashlib
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate import bench_runner
from evidencegate.analysis import REPO_ROOT
from evidencegate.machine import detect_machine
from evidencegate.policy import Policy
from evidencegate.power import PowerCheckError, check_power
from evidencegate.preflight import Check
from evidencegate.run_plan import Block, load_plan
from evidencegate.t009.preflight import EXPECTED_OLLAMA_VERSION, run_preflight_009
from evidencegate.t009.run_plan import PHASES, plan_path

_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


LOG_DIR = "ticket009"


def session_log(phase: str, log_dir: str = LOG_DIR) -> str:
    return f"{log_dir}/{phase}/sessions.jsonl"


def default_stop_file(runs_dir: Path, phase: str, log_dir: str = LOG_DIR) -> Path:
    return runs_dir / log_dir / phase / "STOP"


def next_stop_at(hhmm: str, now: datetime) -> datetime:
    """현지 시각 HH:MM의 다음 시점(now와 같거나 지났으면 다음 날)."""
    m = _HHMM.match(hhmm)
    if not m:
        raise ValueError(f"--stop-at needs HH:MM (00:00-23:59), got {hhmm!r}")
    candidate = now.replace(hour=int(m[1]), minute=int(m[2]), second=0, microsecond=0)
    return candidate if candidate > now else candidate + timedelta(days=1)


def block_check(power: Callable[[], None], version: Callable[[], str]) -> Callable[[], None]:
    """전원 점검 그다음 Ollama 버전 확인. 어긋나면 PowerCheckError."""

    def check() -> None:
        power()
        actual = version()
        if actual != EXPECTED_OLLAMA_VERSION:
            raise PowerCheckError(f"ollama version {actual} (expected {EXPECTED_OLLAMA_VERSION})")

    return check


def server_version(url: str) -> Callable[[], str]:
    def version() -> str:
        try:
            response = httpx.get(f"{url}/api/version", timeout=10)
            response.raise_for_status()
            return str(response.json()["version"])
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise PowerCheckError(f"ollama version check failed: {error}") from error

    return version


def main(
    argv: Sequence[str] | None = None,
    *,
    policy_factory: Callable[[], Policy] | None = None,
    preflight: Callable[[Policy], list[Check]] | None = None,
    probe: Callable[[], Any] | None = None,
    plan: Sequence[Block] | None = None,
    plan_sha256: str | None = None,
    power_check: Callable[[], None] | None = None,
    ollama_version: Callable[[], str] | None = None,
    now: Callable[[], datetime] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
    plan_file: Path | None = None,
    log_dir: str = LOG_DIR,
    preflight_factory: Callable[[argparse.Namespace, Path], Callable[[Policy], list[Check]]]
    | None = None,
) -> int:
    """plan_file(root 기준)·log_dir·preflight_factory는 Ticket 009b가 쓰는 선택 인자다.

    기본값이면 009와 같다(계획 `plans/ticket-009-<단계>-plan.json`, 기록 `runs/ticket009/<단계>/`,
    009 점검)."""
    parser = argparse.ArgumentParser(prog="evidencegate-t009-bench")
    parser.add_argument("command", choices=("run", "status", "preflight"))
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--stop-at", default=None, help="local time HH:MM; stop at a block boundary"
    )
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--ollama-url", default=bench_runner.DEFAULT_OLLAMA_URL)
    args = parser.parse_args(argv)

    plan_file = args.root / (plan_file or plan_path(args.phase))
    blocks = list(plan) if plan is not None else list(load_plan(plan_file))
    sha = plan_sha256 or hashlib.sha256(plan_file.read_bytes()).hexdigest()
    clock = now or (lambda: datetime.now().astimezone())
    try:
        stop_at = next_stop_at(args.stop_at, clock()) if args.stop_at else None
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    stop_file = args.stop_file or default_stop_file(args.runs_dir, args.phase, log_dir)
    version = ollama_version or server_version(args.ollama_url)
    log_name = session_log(args.phase, log_dir)

    status_policy = policy_factory() if policy_factory else bench_runner._StatusPolicy()
    print(
        f"phase {args.phase}, plan sha256 {sha[:12]}, done: "
        f"{bench_runner.status_line(blocks, args.runs_dir, status_policy, log_name)}",
        file=out,
        flush=True,
    )
    if args.command == "status":
        return 0

    def default_preflight(policy: Policy) -> list[Check]:
        return run_preflight_009(
            root=args.root,
            code_root=args.root,
            runs_dir=args.runs_dir,
            current_config=policy.config,
            machine=detect_machine(),
            phase=args.phase,
            plan_path=plan_file,
        )

    passthrough = ["--runs-dir", str(args.runs_dir), "--plan", str(plan_file)]
    passthrough += ["--root", str(args.root), "--ollama-url", args.ollama_url]
    if args.max_blocks is not None:
        passthrough += ["--max-blocks", str(args.max_blocks)]
    fields = {
        "phase": args.phase,
        "plan_sha256": sha,
        "stop_at": stop_at.isoformat() if stop_at else None,
        "stop_file": str(stop_file),
    }
    if args.command == "run":
        fields["ollama_version"] = version()
    return bench_runner.main(
        [args.command, *passthrough],
        policy_factory=policy_factory,
        preflight=preflight
        or (preflight_factory(args, plan_file) if preflight_factory else default_preflight),
        probe=probe,
        plan=blocks,
        session_log=log_name,
        out=out,
        err=err,
        check_power=block_check(power_check or check_power, version),
        stop_at=stop_at,
        stop_file=stop_file,
        log_interrupt=True,
        session_fields=fields,
    )


if __name__ == "__main__":
    raise SystemExit(main())
