"""Ticket 014 proposer 진입점(사전 등록 3절). v1 `propose_cli`에 014 고정 값을 넘긴다.

    caffeinate -dims uv run python -m evidencegate.t014.propose preflight
    uv run python -m evidencegate.t014.propose models
    caffeinate -dims uv run python -m evidencegate.t014.propose propose --calls N \\
        [--stop-at HH:MM] [--stop-file PATH]
    uv run python -m evidencegate.t014.propose status

- 고정 값은 009b와 같다(`t009b.propose.v1_argv`). 다른 점은 다음뿐이다.
  - `--exclude-dir skills/candidates-009b`: 009b 후보와 W 기준 중복이면 풀에서 뺀다.
  - 경로: 014 Dev-A 내보내기, `runs/proposer-014/`(원장·결과·response_models·sessions·STOP).
- 호출 전 점검(하나라도 어긋나면 종료 코드 1, 요청 없음, 세션 기록 `preflight_failed`):

| 점검 | 기준 |
| preregistration_014 | `ticket-014.txt` 전체 sha256이 `ticket-014.sha256` 기록과 같음 |
| prereg_tag | 태그 `t014-prereg`가 HEAD의 조상 |
| export | 내보내기가 `.sha256`과 같고, 014 Dev-A 기록에서 다시 만든 바이트와 같음 |
| strata | 호출 0~119의 층 추출(`check_sampling`) |
| prompt | 프롬프트 sha256 `58ffd9f0…e2c6` |
| paths | 원장·출력 폴더가 v1·009·009b 경로가 아님 |
| outputs | 출력 폴더에 결과가 없거나, 있으면 같은 고정 설정(`run_options.json`)의 이어 부르기 |
| exclude_dir | 009b 후보 폴더에 SKILL.md 40개, `POOL-009b.json`이 `.sha256`과 같고 이름이 같음 |
| power | 전원 AC, 덮개 열림, caffeinate |

- 세션(호출을 하나씩, 호출 사이에 정지 파일과 `--stop-at` 확인), 끝 이유, 응답 모델 기록은 009b
  진입점과 같다.
- API 키는 v1과 같이 환경 변수 OPENAI_API_KEY에서만 읽는다(이 모듈은 키를 읽지 않는다).
"""

import argparse
import hashlib
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
from evidencegate.preflight import Check
from evidencegate.proposer import DEFAULT_BASE_URL
from evidencegate.proposer_inputs import load_proposer_cases
from evidencegate.t009 import propose as p009
from evidencegate.t009.bench import next_stop_at
from evidencegate.t009.propose import Paths
from evidencegate.t009.proposer_inputs import check_sampling
from evidencegate.t009b import propose as p009b
from evidencegate.t009b.preflight import Git, check_power_now, check_prompt, system_git
from evidencegate.t014 import pool
from evidencegate.t014.settings import (
    CONTROLS_DIR,
    EXCLUDE_COUNT,
    EXCLUDE_DIR,
    EXPORT_PATH,
    MAX_CALLS,
    PREREG_FILE,
    PREREG_SHA_FILE,
    PREREG_TAG,
    REJECTED_PATHS,
    RUN_DIR,
    RUN_OPTIONS_FILE,
    fixed_run_options,
)

__all__ = ["EXPORT_PATH", "Paths", "default_paths", "main", "v1_argv"]


def default_paths(root: Path) -> Paths:
    return Paths(
        inputs=root / EXPORT_PATH,
        ledger=root / RUN_DIR / "ledger.jsonl",
        out_dir=root / RUN_DIR / "pool",
        response_models=root / RUN_DIR / "response_models.jsonl",
        skill=root / "skills/S0/SKILL.md",
        controls=root / CONTROLS_DIR,
    )


def v1_argv(root: Path, paths: Paths, calls: int) -> list[str]:
    return [*p009b.v1_argv(paths, calls), "--exclude-dir", str(root / EXCLUDE_DIR)]


# 점검 ---------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_sha256(sha_file: Path) -> str | None:
    for line in sha_file.read_text().splitlines():
        if line.startswith("sha256: "):
            return line.removeprefix("sha256: ").strip()
    return None


def check_preregistration(root: Path) -> Check:
    text, sha_file = root / PREREG_FILE, root / PREREG_SHA_FILE
    if not text.exists() or not sha_file.exists():
        return Check("preregistration_014", False, f"missing {text} or {sha_file}")
    recorded, actual = recorded_sha256(sha_file), _sha256(text)
    if actual != recorded:
        return Check("preregistration_014", False, f"sha256 {actual} does not match {recorded}")
    return Check("preregistration_014", True, f"014 sha256 {actual}")


def check_prereg_tag(root: Path, git: Git = system_git) -> Check:
    code = git(["git", "-C", str(root), "merge-base", "--is-ancestor", PREREG_TAG, "HEAD"])
    if code == 0:
        return Check("prereg_tag", True, f"{PREREG_TAG} is an ancestor of HEAD")
    if code == 1:
        return Check("prereg_tag", False, f"{PREREG_TAG} is not an ancestor of HEAD")
    return Check("prereg_tag", False, f"cannot check {PREREG_TAG} (git exit {code})")


def check_export(root: Path, paths: Paths) -> Check:
    sha_file = paths.inputs.with_name(paths.inputs.name + ".sha256")
    if not paths.inputs.exists() or not sha_file.exists():
        return Check("export", False, f"missing {paths.inputs} or its .sha256")
    actual, recorded = _sha256(paths.inputs), sha_file.read_text().split()[0]
    if actual != recorded:
        return Check("export", False, f"export sha256 {actual} != recorded {recorded}")
    try:
        rebuilt = pool.export_bytes(pool.build_export(pool.dev_a_records(root / "runs")))
    except ValueError as error:
        return Check("export", False, f"cannot rebuild the export: {error}")
    if paths.inputs.read_bytes() != rebuilt:
        return Check("export", False, "export differs from a rebuild from the Dev-A records")
    return Check("export", True, f"export sha256 {actual}")


def check_strata(paths: Paths) -> Check:
    if not paths.inputs.exists():
        return Check("strata", False, f"missing {paths.inputs}")
    sampling = check_sampling(load_proposer_cases(paths.inputs), calls=MAX_CALLS)
    return Check("strata", sampling.ok, sampling.detail)


def check_paths(root: Path, paths: Paths) -> Check:
    rejected = {(root / p).resolve() for p in REJECTED_PATHS}
    bad = [str(p) for p in (paths.ledger, paths.out_dir) if p.resolve() in rejected]
    if bad:
        return Check("paths", False, f"earlier ticket path: {', '.join(bad)}")
    return Check("paths", True, f"ledger {paths.ledger}, out {paths.out_dir}")


def check_outputs(root: Path, paths: Paths) -> Check:
    calls = sorted(paths.out_dir.glob("call-*.json")) if paths.out_dir.exists() else []
    if not calls:
        return Check("outputs", True, f"no saved calls in {paths.out_dir} (new run)")
    options_path = paths.out_dir / RUN_OPTIONS_FILE
    if not options_path.exists():
        return Check("outputs", False, f"{len(calls)} calls without {RUN_OPTIONS_FILE}")
    recorded = json.loads(options_path.read_text(encoding="utf-8"))
    if recorded != fixed_run_options(root):
        return Check("outputs", False, f"{RUN_OPTIONS_FILE} {recorded} is not the 014 setting")
    return Check("outputs", True, f"resume: {len(calls)} calls saved with the 014 setting")


def check_exclude_dir(root: Path) -> Check:
    folder = root / EXCLUDE_DIR
    skills = sorted(p.parent.name for p in folder.glob("*/SKILL.md"))
    pool_file = folder / "POOL-009b.json"
    sha_file = pool_file.with_name(pool_file.name + ".sha256")
    if len(skills) != EXCLUDE_COUNT or not pool_file.exists() or not sha_file.exists():
        return Check("exclude_dir", False, f"{len(skills)} SKILL.md in {folder} (need 40 + POOL)")
    if _sha256(pool_file) != sha_file.read_text().split()[0]:
        return Check("exclude_dir", False, "POOL-009b.json does not match its .sha256")
    names = sorted(
        c["name"]
        for c in json.loads(pool_file.read_text(encoding="utf-8"))["candidates"]
        if c["group"] == "proposer"
    )
    if names != skills:
        return Check("exclude_dir", False, "009b candidate folders differ from POOL-009b.json")
    return Check(
        "exclude_dir", True, f"{len(skills)} Ticket 009b candidates for the duplicate rule"
    )


def proposer_preflight(
    root: Path, paths: Paths, *, power_run: Run = system_run, git: Git = system_git
) -> list[Check]:
    return [
        check_preregistration(root),
        check_prereg_tag(root, git),
        check_export(root, paths),
        check_strata(paths),
        check_prompt(),
        check_paths(root, paths),
        check_outputs(root, paths),
        check_exclude_dir(root),
        check_power_now(power_run),
    ]


# 세션 ---------------------------------------------------------------------------------


def _log(paths: Paths, event: str, **fields: Any) -> None:
    path = paths.ledger.parent / "sessions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "time": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _saved(paths: Paths) -> int:
    return len(list(paths.out_dir.glob("call-*.json"))) if paths.out_dir.exists() else 0


def _run_session(
    root: Path,
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
        options=fixed_run_options(root),
        status=p009b._status(paths),
    )
    made, reason, detail, code = 0, "calls_done", "", 0
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
                before = _saved(paths)
                tee = p009b._Tee(out)
                code = propose_cli.main(v1_argv(root, paths, 1), client=client, out=tee, err=err)
                if code != 0:
                    reason = "stopped" if code == 2 else "error"
                    break
                if _saved(paths) == before:
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
        status=p009b._status(paths),
    )
    first = p009._first_response_model(paths)
    if first is not None:
        print(f"first response model: {first} (requested {p009.MODEL})", file=out)
    print(f"session end: {reason}; {p009b._status(paths)}", file=out)
    return code


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
    transport: httpx.BaseTransport | None = None,
    power_run: Run = system_run,
    git: Git = system_git,
    now: Callable[[], datetime] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    paths = default_paths(root)
    parser = argparse.ArgumentParser(prog="evidencegate-t014-propose")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list model ids available to the API key")
    sub.add_parser("preflight", help="run the pre-call checks only")
    sub.add_parser("status", help="calls, pool, idle calls and cost so far")
    p = sub.add_parser("propose", help="call the proposer with the fixed Ticket 014 settings")
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--stop-at", default=None, help="local time HH:MM; stop between calls")
    p.add_argument("--stop-file", type=Path, default=root / RUN_DIR / "STOP")
    args = parser.parse_args(argv)
    clock = now or (lambda: datetime.now().astimezone())

    if args.command == "models":
        client = httpx.Client(base_url=DEFAULT_BASE_URL, timeout=30, transport=transport)
        return propose_cli.main(["models"], client=client, out=out, err=err)
    if args.command == "status":
        print(p009b._status(paths), file=out)
        return 0
    checks = proposer_preflight(root, paths, power_run=power_run, git=git)
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
        return _run_session(root, args, paths, transport, clock, out, err)
    except ValueError as error:  # --stop-at 형식
        print(f"error: {error}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
