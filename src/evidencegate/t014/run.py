"""Ticket 014 실행 CLI(사전 등록 1·8절, 티켓 §11).

    PYTHONUNBUFFERED=1 caffeinate -dims uv run python -m evidencegate.t014.run run --stage N \\
        [--max-hours H] [--max-blocks K] [--stop-file PATH]
    uv run python -m evidencegate.t014.run preflight --stage N
    uv run python -m evidencegate.t014.run status [--stage N]
    uv run python -m evidencegate.t014.run finalize --stage N

- 에피소드 단계는 0(Dev-A), 1(보정), 2(Dev-B), 4(Oracle)다. 계획은
  `plans/ticket-014-stage<N>-plan.json`(동결 파일). 블록 실행·재개·서버 오류·형식 오류 규칙은 v1
  `bench_runner.main` 그대로다. 블록마다 전원과 Ollama 버전을 확인한다(`t009.bench.block_check`).
- `--max-hours H`: 지금 + H시간을 `stop_at`으로 넘긴다. 그 시각이 지나면 다음 블록을 시작하지 않고
  정상 종료한다(진행 중인 블록은 끝까지). 정지 파일(기본 `runs/ticket014/stage<N>/STOP`)도 같다.
  다시 같은 명령을 내면 끝난 블록은 건너뛰고 블록 안에서는 없는 케이스만 돈다. Ctrl-C는 끝 기록
  "interrupted"로 남는다.
- 실행 전 점검: 009 점검 중 실행 조건 7개(엔진, config_id, 장비, 병렬, Ollama 버전, 전원, 디스크)에
  014 사전 등록 sha256, 태그 `t014-prereg`, 그 단계 계획(`plans.check_plan`), 앞 단계 결과 해시
  (`stages.check_previous`), 단계 2는 묶음 파일 검사를 더한다. 그다음 환경 지문(`fingerprint`)을
  잰다. `src/`가 깨끗하지 않으면 점검 실패(종료 코드 1)다. 앞 단계 지문이나 이 단계 지문과 다르면
  아무것도 실행하지 않고 종료 코드 2로 끝낸다(세션 기록 `fingerprint_mismatch`). 이 단계 지문이
  없으면 `run`이 처음 쓴다(`preflight`는 쓰지 않는다).
- 세션 기록 `runs/ticket014/stage<N>/sessions.jsonl`. 시작 기록에 단계, 계획 sha256, Ollama 버전,
  코드 커밋, stop_at, max_hours, 정지 파일을 남긴다.
- `status`: 단계마다 완료 블록·에피소드 수, 남은 예상 시간(이 단계 기록의 에피소드 시간 평균 / 동시
  실행 수, 기록이 없으면 009b 실측 6.67초), 마지막 세션 끝 이유, 지문 기록 여부와 코드 트리 일치.
  CD·ES 같은 결과 값은 출력하지 않는다.
- `finalize --stage N`: 그 단계 결과를 한 번 쓴다(0은 `pool.finalize_stage0`, 1~5는
  `stages.finalize`). 0과 3 뒤에는 David 확인을 위해 멈춘다.
- 단계 3의 Oracle 대상이 0개이면 `run --stage 4`는 아무것도 돌리지 않고 건너뜀을 알린다.
  `finalize --stage 4`가 `skipped: true`를 기록한다(사전 등록 8절).
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from evidencegate import bench_runner
from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import RUN_WORKERS
from evidencegate.machine import detect_machine
from evidencegate.policy import Policy
from evidencegate.power import check_power
from evidencegate.preflight import (
    Check,
    check_config_id,
    check_disk,
    check_engine,
    check_machine,
    check_parallel,
    preflight_ok,
    reference_policy_config,
)
from evidencegate.run_plan import Block, load_plan
from evidencegate.store import EpisodeStore
from evidencegate.t009.bench import block_check, server_version
from evidencegate.t009.preflight import check_ollama_version, check_power_now
from evidencegate.t014 import fingerprint, plans, pool, stages
from evidencegate.t014.oracle_scale import ACTIVE_S_PER_EPISODE
from evidencegate.t014.propose import check_prereg_tag, check_preregistration
from evidencegate.t014.settings import EPISODE_STAGES, LOG_DIR, PREREG_FILE, STAGES, STOP_STAGES


class FingerprintMismatch(RuntimeError):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def session_log(stage: int) -> str:
    return f"{LOG_DIR}/stage{stage}/sessions.jsonl"


def default_checks(
    args: argparse.Namespace, git: Callable[[Sequence[str]], int]
) -> Callable[[Policy], list[Check]]:
    def checks(policy: Policy) -> list[Check]:
        try:
            engine = check_engine(policy.config, reference_policy_config(args.runs_dir))
        except ValueError as error:
            engine = Check("engine", False, f"cannot read the S0 Dev-A reference: {error}")
        out = [
            engine,
            check_config_id(policy.config),
            check_machine(detect_machine()),
            check_parallel(policy.config),
            check_ollama_version(policy.config),
            check_power_now(),
            check_disk(
                args.runs_dir if args.runs_dir.exists() else args.root, usage=shutil.disk_usage
            ),
            check_preregistration(args.root),
            check_prereg_tag(args.root, git),
            plans.check_plan(args.root, args.stage),
            stages.check_previous(args.root, args.stage),
        ]
        if args.stage == 2:
            problems = pool.groups_ready(args.root)
            out.append(Check("pool_groups", not problems, "; ".join(problems) or "groups ready"))
        return out

    return checks


def _tag_git(git: fingerprint.Git) -> Callable[[Sequence[str]], int]:
    return lambda argv: git(argv)[0]


def _fingerprint_step(
    args: argparse.Namespace,
    checks: list[Check],
    policy: Policy,
    plan: Sequence[Block],
    plan_sha256: str,
    prereg_sha256: str | None,
    git: fingerprint.Git,
) -> list[Check]:
    try:
        now = fingerprint.collect(
            stage=args.stage,
            config=policy.config,
            plan=plan,
            plan_sha256=plan_sha256,
            prereg_sha256=prereg_sha256 or "",
            root=args.root,
            git=git,
        )
    except fingerprint.FingerprintError as error:
        return [*checks, Check("fingerprint", False, str(error))]
    earlier = [
        stages.fingerprint_path(args.runs_dir, s)
        for s in EPISODE_STAGES
        if s < args.stage and stages.fingerprint_path(args.runs_dir, s).exists()
    ]
    problems = fingerprint.compare_environment(earlier, now)
    path = stages.fingerprint_path(args.runs_dir, args.stage)
    if not problems:
        if args.command == "run":
            problems = fingerprint.ensure(path, now)
        elif path.exists():
            recorded = json.loads(path.read_text(encoding="utf-8"))
            problems = fingerprint._differences(
                recorded, now, (*fingerprint.ENVIRONMENT_FIELDS, *fingerprint.STAGE_FIELDS)
            )
    if problems:
        raise FingerprintMismatch(problems)
    return [*checks, Check("fingerprint", True, f"stage {args.stage} environment matches")]


def _run(
    args: argparse.Namespace,
    *,
    plan: Sequence[Block],
    plan_sha256: str,
    prereg_sha256: str | None,
    policy_factory: Callable[[], Policy] | None,
    probe: Callable[[], Any] | None,
    environment_checks: Callable[[Policy], list[Check]] | None,
    power_check: Callable[[], None] | None,
    ollama_version: Callable[[], str] | None,
    git: fingerprint.Git,
    clock: Callable[[], datetime],
    out: TextIO,
    err: TextIO,
) -> int:
    checks_for = environment_checks or default_checks(args, _tag_git(git))

    def preflight(policy: Policy) -> list[Check]:
        checks = checks_for(policy)
        if not preflight_ok(checks):
            return checks
        return _fingerprint_step(args, checks, policy, plan, plan_sha256, prereg_sha256, git)

    stop_at = clock() + timedelta(hours=args.max_hours) if args.max_hours is not None else None
    stop_file = args.stop_file or args.runs_dir / LOG_DIR / f"stage{args.stage}" / "STOP"
    version = ollama_version or server_version(args.ollama_url)
    code, commit = git(["git", "-C", str(args.root), "rev-parse", "HEAD"])
    fields: dict[str, Any] = {
        "stage": args.stage,
        "plan_sha256": plan_sha256,
        "code_commit": commit.strip() if code == 0 else None,
        "stop_at": stop_at.isoformat() if stop_at else None,
        "max_hours": args.max_hours,
        "stop_file": str(stop_file),
    }
    if args.command == "run":
        fields["ollama_version"] = version()
    passthrough = ["--runs-dir", str(args.runs_dir), "--root", str(args.root)]
    passthrough += ["--ollama-url", args.ollama_url]
    if args.max_blocks is not None:
        passthrough += ["--max-blocks", str(args.max_blocks)]
    try:
        return bench_runner.main(
            [args.command, *passthrough],
            policy_factory=policy_factory,
            preflight=preflight,
            probe=probe,
            plan=plan,
            session_log=session_log(args.stage),
            out=out,
            err=err,
            check_power=block_check(power_check or check_power, version),
            stop_at=stop_at,
            stop_file=stop_file,
            log_interrupt=True,
            session_fields=fields,
        )
    except FingerprintMismatch as mismatch:
        for problem in mismatch.problems:
            print(f"error: fingerprint {problem}", file=err)
        print("error: the environment fingerprint differs; nothing was run", file=err)
        log = bench_runner.SessionLog(args.runs_dir, session_log(args.stage))
        log.write("fingerprint_mismatch", stage=args.stage, problems=mismatch.problems)
        return 2


# status -------------------------------------------------------------------------------


def _last_reason(runs_dir: Path, stage: int) -> str | None:
    path = runs_dir / session_log(stage)
    if not path.exists():
        return None
    ends = [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and json.loads(x).get("event") == "session_end"
    ]
    return ends[-1].get("reason") if ends else None


def stage_status(
    runs_dir: Path,
    stage: int,
    plan: Sequence[Block],
    git: fingerprint.Git,
    root: Path,
    policy: Any,
) -> str:
    stored: dict[tuple[str, str], set[tuple[int, tuple[str, str, int]]]] = {}
    durations: list[float] = []
    wanted = {(b.skill_id, b.repeat, c) for b in plan for c in b.cases}
    for split, skill in {(b.split, b.skill_id) for b in plan}:
        block = next(b for b in plan if (b.split, b.skill_id) == (split, skill))
        path = bench_runner.block_store(runs_dir, policy, block).path  # 실행기와 같은 경로
        keys = set()
        if path.exists():
            for r in EpisodeStore(path, repair=False).records():
                case = (r.key.family, r.key.level, r.key.incident_seed)
                if (skill, r.key.repeat, case) in wanted:
                    keys.add((r.key.repeat, case))
                    durations.append(r.duration_s)
        stored[(split, skill)] = keys
    done_blocks = sum(
        1 for b in plan if all((b.repeat, c) in stored[(b.split, b.skill_id)] for c in b.cases)
    )
    total = sum(len(b.cases) for b in plan)
    done = sum(len(v) for v in stored.values())
    per_ep = sum(durations) / len(durations) / RUN_WORKERS if durations else ACTIVE_S_PER_EPISODE
    basis = "this stage" if durations else "Ticket 009b"
    remaining_h = (total - done) * per_ep / 3600
    fp_path = stages.fingerprint_path(runs_dir, stage)
    if fp_path.exists():
        recorded = json.loads(fp_path.read_text(encoding="utf-8")).get("code_tree")
        code, tree = git(["git", "-C", str(root), "rev-parse", "HEAD:src"])
        same = "yes" if code == 0 and tree.strip() == recorded else "no"
        fp = f"fingerprint recorded, code tree matches: {same}"
    else:
        fp = "no fingerprint recorded yet"
    return (
        f"stage {stage}: blocks {done_blocks}/{len(plan)}, episodes {done}/{total}, "
        f"remaining about {remaining_h:.1f} h active ({per_ep:.2f} s/episode, {basis}), "
        f"last session: {_last_reason(runs_dir, stage)}, {fp}"
    )


# main ---------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    policy_factory: Callable[[], Policy] | None = None,
    probe: Callable[[], Any] | None = None,
    plan: Sequence[Block] | None = None,
    plan_sha256: str | None = None,
    prereg_sha256: str | None = None,
    environment_checks: Callable[[Policy], list[Check]] | None = None,
    power_check: Callable[[], None] | None = None,
    ollama_version: Callable[[], str] | None = None,
    git: fingerprint.Git = fingerprint.system_git,
    now: Callable[[], datetime] | None = None,
    status_policy: Any = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t014-run")
    parser.add_argument("command", choices=("run", "preflight", "status", "finalize"))
    parser.add_argument("--stage", type=int, choices=STAGES, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--ollama-url", default=bench_runner.DEFAULT_OLLAMA_URL)
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours < 0:
        print("error: --max-hours must be >= 0", file=err)
        return 1

    if args.command == "finalize":
        return _finalize(args, out, err)
    if args.command == "status":
        wanted = [args.stage] if args.stage is not None else list(EPISODE_STAGES)
        for stage in wanted:
            if stage not in EPISODE_STAGES:
                print(f"stage {stage}: computation only", file=out)
                continue
            path = args.root / plans.plan_path(stage)
            blocks = plan if plan is not None else load_plan(path) if path.exists() else None
            if blocks is None:
                print(f"stage {stage}: no plan file yet ({plans.plan_path(stage)})", file=out)
                continue
            policy = status_policy or bench_runner._StatusPolicy()
            print(stage_status(args.runs_dir, stage, blocks, git, args.root, policy), file=out)
        return 0

    if args.stage not in EPISODE_STAGES:
        print(f"error: stage {args.stage} runs no episodes; use finalize", file=err)
        return 1
    if args.stage == 4 and stages.result_path(args.root, 3).exists() and plan is None:
        targets = plans.load_targets(args.root)
        if not targets.proposer and not targets.controls:
            print("stage 4 is skipped: no Oracle targets; run finalize --stage 4", file=out)
            return 0
    plan_file = args.root / plans.plan_path(args.stage)
    if plan is None and not plan_file.exists():
        print(f"error: {plan_file} does not exist; freeze the stage plan first", file=err)
        return 1
    blocks = list(plan) if plan is not None else list(load_plan(plan_file))
    sha = plan_sha256 or hashlib.sha256(plan_file.read_bytes()).hexdigest()
    prereg = args.root / PREREG_FILE
    prereg_sha = prereg_sha256 or (
        hashlib.sha256(prereg.read_bytes()).hexdigest() if prereg.exists() else None
    )
    return _run(
        args,
        plan=blocks,
        plan_sha256=sha,
        prereg_sha256=prereg_sha,
        policy_factory=policy_factory,
        probe=probe,
        environment_checks=environment_checks,
        power_check=power_check,
        ollama_version=ollama_version,
        git=git,
        clock=now or (lambda: datetime.now().astimezone()),
        out=out,
        err=err,
    )


def _finalize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.stage is None:
        print("error: finalize needs --stage", file=err)
        return 1
    try:
        if args.stage == 0:
            result = pool.finalize_stage0(args.root, args.runs_dir)
        else:
            result = stages.finalize(args.stage, root=args.root, runs_dir=args.runs_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=err)
        return 1
    path = stages.result_path(args.root, args.stage)
    digest = path.with_name(path.name + ".sha256").read_text().split()[0]
    print(f"wrote {path} (sha256 {digest})", file=out)
    if args.stage == 0:
        print(f"stage 0 go: {result.get('go')}", file=out)
    if args.stage == 3:
        targets = result["oracle_targets"]
        print(
            f"Oracle targets: {len(targets['proposer'])} proposer candidates, "
            f"controls {targets['controls']}",
            file=out,
        )
    if args.stage in STOP_STAGES:
        print(f"stop here: stage {args.stage} needs David's confirmation", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
