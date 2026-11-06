"""Ticket 009b proposer 진입점(사전 등록 2.1, 티켓 §5의 4).

v1 `propose_cli`에 009b 고정 값을 넘긴다.

    caffeinate -dims uv run python -m evidencegate.t009b.propose preflight
    uv run python -m evidencegate.t009b.propose models
    caffeinate -dims uv run python -m evidencegate.t009b.propose propose --calls N \\
        [--stop-at HH:MM] [--stop-file PATH]
    uv run python -m evidencegate.t009b.propose status

- 고정 값(명령줄로 바꿀 수 없음): 009 진입점의 모델·단가·토큰에 더해 `--previous valid`,
  `--dedupe word --dedupe-threshold 0.55`, `--stop-when-pool 40`, `--stop-after-idle 25`,
  `--max-total-calls 120`, `--budget-usd 5`, `--quota-message "no credits remaining"`.
  경로는 009 내보내기, `runs/proposer-009b/`(원장·결과·response_models·sessions·STOP)다.
- 호출 전 점검: `t009b.preflight.proposer_preflight`. 하나라도 어긋나면 종료 코드 1이고 요청을
  보내지 않는다(세션 기록에 `preflight_failed`).
- 세션: 호출을 한 번씩(`--calls 1`) v1 `propose_cli.main`에 넘기고, 호출 사이마다 정지 파일(기본
  `runs/proposer-009b/STOP`)과 `--stop-at`(이 맥의 현지 시각, 같거나 지났으면 다음 날)을 본다.
  `--stop-at` 계산과 비교는 같은 시계를 쓴다.
  세션 기록 `runs/proposer-009b/sessions.jsonl`의 끝 이유:
  calls_done(요청한 수를 다 부름), limit(사전 등록 멈춤: 풀 40·포화 25·120회), stop_file, stop_at,
  stopped(v1 중단, 종료 코드 2: 예산·잔액·인증·서버·형식 오류율), error(종료 코드 1), interrupted.
- 응답 모델 이름은 009 진입점과 같은 응답 훅으로 `response_models.jsonl`에 남긴다.
- API 키는 v1과 같이 환경 변수 OPENAI_API_KEY에서만 읽는다(이 모듈은 키를 읽지 않는다).
"""

import argparse
import io
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate import propose_cli
from evidencegate.analysis import REPO_ROOT
from evidencegate.power import Run, system_run
from evidencegate.proposer import DEFAULT_BASE_URL
from evidencegate.t009 import propose as p009
from evidencegate.t009.bench import next_stop_at
from evidencegate.t009.propose import Paths
from evidencegate.t009.proposer_inputs import OUT_PATH
from evidencegate.t009b.preflight import Git, proposer_preflight, system_git
from evidencegate.t009b.settings import (
    BUDGET_USD,
    DEDUPE_THRESHOLD,
    EXPORT_SHA256,
    FIXED_RUN_OPTIONS,
    MAX_CALLS,
    POOL_TARGET,
    QUOTA_MESSAGE,
    RUN_DIR,
    STOP_AFTER_IDLE,
    STOP_FILE,
)

__all__ = ["FIXED_RUN_OPTIONS", "STOP_FILE", "Paths", "default_paths", "main", "next_stop_at"]


def default_paths(root: Path) -> Paths:
    return Paths(
        inputs=root / OUT_PATH,
        ledger=root / RUN_DIR / "ledger.jsonl",
        out_dir=root / RUN_DIR / "pool",
        response_models=root / RUN_DIR / "response_models.jsonl",
        skill=root / "skills/S0/SKILL.md",
        controls=root / "skills/controls",
    )


def v1_argv(paths: Paths, calls: int) -> list[str]:
    base = p009.v1_argv(paths, calls)
    fixed = dict(zip(base[1::2], base[2::2], strict=True))
    fixed |= {
        "--budget-usd": str(BUDGET_USD),
        "--max-total-calls": str(MAX_CALLS),
        "--stop-when-pool": str(POOL_TARGET),
        "--previous": FIXED_RUN_OPTIONS["previous"],
        "--dedupe": FIXED_RUN_OPTIONS["dedupe"],
        "--dedupe-threshold": str(DEDUPE_THRESHOLD),
        "--stop-after-idle": str(STOP_AFTER_IDLE),
        "--quota-message": QUOTA_MESSAGE,
    }
    return ["propose", *(x for pair in fixed.items() for x in pair)]


class _Tee(io.StringIO):
    """받은 출력을 그대로 넘기면서 줄을 모은다(v1의 멈춤 문구를 끝 이유에 적으려고)."""

    def __init__(self, target: TextIO) -> None:
        super().__init__()
        self.target = target
        self.lines: list[str] = []

    def write(self, text: str) -> int:
        self.lines += text.splitlines()
        return self.target.write(text)


def _log(paths: Paths, event: str, **fields: Any) -> None:
    path = paths.ledger.parent / "sessions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "time": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _calls(paths: Paths) -> list[dict[str, Any]]:
    if not paths.out_dir.exists():
        return []
    files = sorted(paths.out_dir.glob("call-*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _status(paths: Paths) -> str:
    calls = _calls(paths)
    idle = 0
    for c in calls:
        idle = 0 if c.get("in_pool") else idle + 1
    pool = sum(1 for c in calls if c.get("in_pool"))
    cost = sum(float(c.get("cost_usd") or 0.0) for c in calls)
    return f"calls {len(calls)}, pool {pool}, idle {idle}, cost ${cost:.4f}"


def _run_session(
    args: argparse.Namespace,
    paths: Paths,
    transport: httpx.BaseTransport | None,
    clock: Callable[[], datetime],
    out: TextIO,
    err: TextIO,
) -> int:
    stop_at = next_stop_at(args.stop_at, clock()) if args.stop_at else None
    stop_file: Path = args.stop_file
    _log(
        paths,
        "session_start",
        calls_requested=args.calls,
        stop_at=stop_at.isoformat() if stop_at else None,
        stop_file=str(stop_file),
        options=FIXED_RUN_OPTIONS,
        **{"status": _status(paths)},
    )
    made = 0
    reason, detail, code = "calls_done", "", 0
    try:
        with p009.recording_client(paths, transport) as client:
            for _ in range(args.calls):
                if stop_file.exists():
                    reason, detail = "stop_file", str(stop_file)
                    print(f"stop file {stop_file} found; ending the session", file=out)
                    break
                if stop_at is not None and clock() >= stop_at:
                    reason, detail = "stop_at", stop_at.isoformat()
                    print(f"stop time {stop_at.isoformat()} reached; ending the session", file=out)
                    break
                before = len(_calls(paths))
                tee = _Tee(out)
                code = propose_cli.main(v1_argv(paths, 1), client=client, out=tee, err=err)
                if code != 0:
                    reason = "stopped" if code == 2 else "error"
                    break
                if len(_calls(paths)) == before:
                    reason = "limit"
                    detail = next((x for x in tee.lines if "no more calls" in x), "")
                    break
                made += 1
    except KeyboardInterrupt:
        _log(paths, "session_end", reason="interrupted", calls_made=made, exit_code=130)
        raise
    _log(
        paths,
        "session_end",
        reason=reason,
        detail=detail,
        calls_made=made,
        exit_code=code,
        status=_status(paths),
    )
    first = p009._first_response_model(paths)
    if first is not None:
        print(f"first response model: {first} (requested {p009.MODEL})", file=out)
    print(f"session end: {reason}; {_status(paths)}", file=out)
    return code


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
    paths: Paths | None = None,
    transport: httpx.BaseTransport | None = None,
    power_run: Run = system_run,
    git: Git = system_git,
    expected_export_sha256: str = EXPORT_SHA256,
    now: Callable[[], datetime] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    paths = paths or default_paths(root)
    parser = argparse.ArgumentParser(prog="evidencegate-t009b-propose")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list model ids available to the API key")
    sub.add_parser("preflight", help="run the pre-call checks only")
    sub.add_parser("status", help="calls, pool, idle calls and cost so far")
    p = sub.add_parser("propose", help="call the proposer with the fixed Ticket 009b settings")
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--stop-at", default=None, help="local time HH:MM; stop between calls")
    p.add_argument("--stop-file", type=Path, default=paths.ledger.parent / STOP_FILE.name)
    args = parser.parse_args(argv)
    clock = now or (lambda: datetime.now().astimezone())

    if args.command == "models":
        client = httpx.Client(base_url=DEFAULT_BASE_URL, timeout=30, transport=transport)
        return propose_cli.main(["models"], client=client, out=out, err=err)
    if args.command == "status":
        print(_status(paths), file=out)
        return 0
    checks = proposer_preflight(
        root, paths, power_run=power_run, git=git, expected_export_sha256=expected_export_sha256
    )
    for c in checks:
        print(f"preflight {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}", file=out)
    failed = [c for c in checks if not c.ok]
    if args.command == "preflight":
        return 1 if failed else 0
    if failed:
        _log(paths, "preflight_failed", checks=[c.__dict__ for c in checks])
        for c in failed:
            print(f"error: {c.name}: {c.detail}", file=err)
        print("error: preflight failed; no request was sent", file=err)
        return 1
    try:
        return _run_session(args, paths, transport, clock, out, err)
    except ValueError as error:  # --stop-at 형식
        print(f"error: {error}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
