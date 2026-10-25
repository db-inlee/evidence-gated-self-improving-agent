"""Ticket 006 실행기: L3 split, L3-check, L3 본 실행 계획과 본 실행 (§3, §6 순서 4·6).

    uv run python -m evidencegate.l3_bench plan        # 본 실행 계획 파일과 sha256(없을 때 만듦)
    uv run python -m evidencegate.l3_bench preflight   # 본 실행 점검만(계획 파일 점검 포함)
    uv run python -m evidencegate.l3_bench run [--max-blocks N]   # 본 실행 7,200개, 이어서 실행
    uv run python -m evidencegate.l3_bench status      # 본 실행 진행 확인(CD·ES 없음)
    uv run python -m evidencegate.l3_bench check       # L3-check: S0 x 30개(반복 0), 이어서 실행
    uv run python -m evidencegate.l3_bench summary     # 저장된 L3-check의 운영 지표

- L3 split은 §3 표의 시드 범위다(`L3_SPLIT_SEEDS`). Ticket 005 split 목록(`EVAL_SPLITS`)과
  따로 둔다.
- L3-check는 Ticket 005 실행기(`bench_runner`)를 그대로 쓴다: 실행 전 점검 → 병렬 탐침 → 블록
  실행, 같은 중단 규칙. 세션 로그만 `runs/ticket006/sessions.jsonl`로 나눈다.
- 실행 전 점검은 Ticket 005와 같고, Ticket 005 계획 파일 점검 대신 L3 계획 파일 점검을 한다
  (L3-check는 계획 파일 점검 없음). Ticket 006 사전 등록 해시 점검이 들어 있다.
- 본 실행 계획: 후보 19개(풀 순서) x l3_oracle 반복 0, 그다음 S0 x l3_oracle 반복 0~4를 정해진
  순서로 두고 `random.Random(PLAN_ORDER_SEED)`로 섞는다(Ticket 005 §2와 같은 규칙, 파일럿 없음).
  파일은 Ticket 005 계획과 같은 형식(정렬된 키의 JSON)이고 sha256으로 고정한다.
- **운영 지표만 낸다:** 에피소드 수, 형식 오류, 처분 없음, 종료 이유, 도구 호출 수, 시간, 서버 오류,
  본 실행(7,200개) 예상 시간. CD·ES·조치 종류는 어디에도 내지 않는다(§6 순서 4).
"""

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from evidencegate import bench_runner
from evidencegate.analysis import REPO_ROOT, PoolSpec, load_pool
from evidencegate.bench_runner import (
    DEFAULT_OLLAMA_URL,
    SessionLog,
    StopRules,
    _duration,
    _StatusPolicy,
    block_store,
)
from evidencegate.eval_constants import (
    GATE_REPEAT,
    L3_CHECK_SPLIT,
    L3_LEVEL,
    L3_MAIN_EPISODES,
    L3_ORACLE_SPLIT,
    L3_PLAN_PATH,
    L3_SESSION_LOG,
    L3_SPLIT_SEEDS,
    PLAN_ORDER_SEED,
    RUN_WORKERS,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
)
from evidencegate.eval_data import CaseId
from evidencegate.machine import Machine, detect_machine
from evidencegate.policy import Policy
from evidencegate.preflight import (
    Check,
    check_config_id,
    check_disk,
    check_engine,
    check_frozen_pool,
    check_machine,
    check_parallel,
    check_preregistration_7_8,
    check_preregistration_l3,
    reference_policy_config,
)
from evidencegate.run_plan import S0_NAME, S0_PATH, Block, _skill_path, load_plan, plan_bytes
from evidencegate.runner import EpisodeRecord
from evidencegate.skills import load_skill
from evidencegate.store import EpisodeStore
from opsworld.state import Family

S0_ORACLE_REFERENCE_SPLIT = "oracle"  # 비교용: Ticket 005 S0 Oracle 반복 0(L1/L2)


def l3_split_cases(split: str) -> tuple[CaseId, ...]:
    if split not in L3_SPLIT_SEEDS:
        raise ValueError(f"unknown L3 split {split!r}; expected one of {sorted(L3_SPLIT_SEEDS)}")
    first, per_family = L3_SPLIT_SEEDS[split]
    return tuple(
        sorted(
            (family.value, L3_LEVEL, seed)
            for family in Family
            for seed in range(first, first + per_family)
        )
    )


def l3_check_block(root: Path = REPO_ROOT) -> Block:
    skill_id = load_skill(root / S0_PATH).skill_id
    if skill_id != S0_SKILL_ID:
        raise ValueError(f"S0 at {S0_PATH} has skill_id {skill_id}, expected {S0_SKILL_ID}")
    return Block(
        0,
        S0_NAME,
        "s0",
        skill_id,
        S0_PATH,
        L3_CHECK_SPLIT,
        GATE_REPEAT,
        l3_split_cases(L3_CHECK_SPLIT),
    )


def build_l3_plan(pool: PoolSpec, root: Path = REPO_ROOT) -> tuple[Block, ...]:
    """§3 본 실행 계획: 후보 19 x l3_oracle 반복 0 + S0 x 반복 0~4, Random(0)으로 섞은 순서."""
    skills = {c.name: (c.skill_id, _skill_path(c.name, c.group), c.group) for c in pool.candidates}
    skills[S0_NAME] = (pool.s0_skill_id, S0_PATH, "s0")
    for name, (skill_id, path, _) in skills.items():
        actual = load_skill(root / path).skill_id
        if actual != skill_id:
            raise ValueError(f"{name}: {path} has skill_id {actual}, expected {skill_id}")
    keys = [(c.name, GATE_REPEAT) for c in pool.candidates]
    keys += [(S0_NAME, r) for r in S0_ORACLE_REPEATS]
    random.Random(PLAN_ORDER_SEED).shuffle(keys)
    cases = l3_split_cases(L3_ORACLE_SPLIT)
    return tuple(
        Block(
            index,
            name,
            skills[name][2],
            skills[name][0],
            skills[name][1],
            L3_ORACLE_SPLIT,
            r,
            cases,
        )
        for index, (name, r) in enumerate(keys)
    )


def check_l3_plan(plan_path: Path, root: Path) -> Check:
    """L3 계획 파일이 자기 sha256 기록과 같고, 지금 풀로 다시 만든 계획과 바이트까지 같다."""
    sha_path = plan_path.with_name(plan_path.name + ".sha256")
    if not plan_path.exists() or not sha_path.exists():
        return Check("plan", False, f"missing {plan_path} or {sha_path}")
    data = plan_path.read_bytes()
    recorded = sha_path.read_text().split()[0]
    actual = hashlib.sha256(data).hexdigest()
    if actual != recorded:
        return Check("plan", False, f"plan sha256 {actual} does not match recorded {recorded}")
    if data != plan_bytes(build_l3_plan(load_pool(root), root)):
        return Check("plan", False, "plan file differs from a rebuild from the frozen pool")
    return Check("plan", True, f"L3 plan sha256 {actual}")


def l3_preflight(
    *,
    root: Path,
    runs_dir: Path,
    current_config: Mapping[str, Any],
    machine: Machine,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    plan_path: Path | None = None,
) -> list[Check]:
    """Ticket 005 실행 전 점검에서 Ticket 005 계획 파일 점검을 뺀 것.

    본 실행은 `plan_path`로 L3 계획 파일 점검을 디스크 점검 앞에 더한다.
    """
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
        *([check_l3_plan(plan_path, root)] if plan_path is not None else []),
        check_disk(runs_dir if runs_dir.exists() else root, usage=disk_usage),
    ]


@dataclass(frozen=True)
class OpsSummary:
    """운영 지표. CD·ES·조치 종류는 넣지 않는다."""

    episodes: int
    planned: int
    format_error_episodes: int  # 형식 오류가 1회 이상 있었던 에피소드
    format_errors: int  # 형식 오류 총수
    no_disposition: int  # 처분 없이 끝난 에피소드
    stop_reasons: dict[str, int]
    mean_tool_calls: float
    max_tool_calls: int
    mean_duration_s: float  # 에피소드 하나의 실행 시간 평균(동시 실행 중)
    wall_s: float  # 첫 시작 ~ 마지막 끝
    s_per_episode: float  # wall_s / 에피소드 수(동시 2 처리량)
    main_run_estimate_s: float  # s_per_episode x 본 실행 에피소드 수
    server_errors: int


def ops_summary(
    records: Sequence[EpisodeRecord], *, planned: int, server_errors: int
) -> OpsSummary:
    if not records:
        raise ValueError("no L3-check records")
    starts = [datetime.fromisoformat(r.started_at) for r in records]
    ends = [s + timedelta(seconds=r.duration_s) for s, r in zip(starts, records, strict=True)]
    wall = (max(ends) - min(starts)).total_seconds()
    calls = [len(r.trace.turns) for r in records]
    per = wall / len(records)
    return OpsSummary(
        episodes=len(records),
        planned=planned,
        format_error_episodes=sum(1 for r in records if r.trace.format_errors > 0),
        format_errors=sum(r.trace.format_errors for r in records),
        no_disposition=sum(1 for r in records if r.action is None),
        stop_reasons=dict(sorted(Counter(str(r.trace.stop_reason) for r in records).items())),
        mean_tool_calls=sum(calls) / len(calls),
        max_tool_calls=max(calls),
        mean_duration_s=sum(r.duration_s for r in records) / len(records),
        wall_s=wall,
        s_per_episode=per,
        main_run_estimate_s=per * L3_MAIN_EPISODES,
        server_errors=server_errors,
    )


def render_ops(s: OpsSummary, *, reference: OpsSummary | None) -> str:
    stops = ", ".join(f"{name} {n}" for name, n in s.stop_reasons.items())
    lines = [
        f"L3-check: {s.episodes}/{s.planned} episodes",
        f"  format errors: {s.format_error_episodes} episodes ({s.format_errors} total); "
        f"no disposition: {s.no_disposition}; server errors: {s.server_errors}",
        f"  stop reasons: {stops}",
        f"  tool calls per episode: mean {s.mean_tool_calls:.2f}, max {s.max_tool_calls}",
        f"  duration: mean {s.mean_duration_s:.1f} s per episode; wall {_duration(s.wall_s)} "
        f"-> {s.s_per_episode:.1f} s/episode ({RUN_WORKERS} workers)",
        f"  main run estimate: {L3_MAIN_EPISODES} episodes x {s.s_per_episode:.1f} s/episode "
        f"= {_duration(s.main_run_estimate_s)}",
    ]
    if reference is not None:
        lines.append(
            f"  reference S0 L1/L2 Oracle r0: {reference.episodes} episodes, "
            f"mean tool calls {reference.mean_tool_calls:.2f}, "
            f"mean duration {reference.mean_duration_s:.1f} s per episode"
        )
    return "\n".join(lines)


def check_summary(runs_dir: Path, policy: Policy, block: Block) -> tuple[OpsSummary, str]:
    records = [
        r
        for r in block_store(runs_dir, policy, block).records()
        if r.key.repeat == block.repeat
        and (r.key.family, r.key.level, r.key.incident_seed) in set(block.cases)
    ]
    summary = ops_summary(
        records,
        planned=len(block.cases),
        server_errors=SessionLog(runs_dir, L3_SESSION_LOG).count("server_error"),
    )
    return summary, render_ops(summary, reference=_s0_oracle_reference(runs_dir, policy, block))


def _s0_oracle_reference(runs_dir: Path, policy: Policy, block: Block) -> OpsSummary | None:
    reference_block = Block(
        0,
        block.name,
        block.group,
        block.skill_id,
        block.skill_path,
        S0_ORACLE_REFERENCE_SPLIT,
        0,
        (),
    )
    path = block_store(runs_dir, policy, reference_block).path
    if not path.exists():
        return None
    records = [r for r in EpisodeStore(path, repair=False).records() if r.key.repeat == 0]
    if not records:
        return None
    return ops_summary(records, planned=len(records), server_errors=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    policy_factory: Callable[[], Policy] | None = None,
    preflight: Callable[[Policy], list[Check]] | None = None,
    probe: Callable[[], Any] | None = None,
    status_policy: Policy | None = None,
    rules: StopRules | None = None,
    plan: Sequence[Block] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-l3")
    parser.add_argument(
        "command", choices=("plan", "preflight", "run", "status", "check", "summary")
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--plan", type=Path, default=L3_PLAN_PATH)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args(argv)

    if args.command == "plan":
        return _write_plan(args.plan, args.root, out, err)
    check_block = l3_check_block(args.root)
    if args.command == "summary":
        return _print_summary(args.runs_dir, status_policy, check_block, out, err)

    main_run = args.command in ("preflight", "run", "status")
    blocks = list(plan) if plan is not None else None
    if main_run and blocks is None:
        blocks = list(load_plan(args.plan)) if args.plan.exists() else []

    def default_preflight(policy: Policy) -> list[Check]:
        return l3_preflight(
            root=args.root,
            runs_dir=args.runs_dir,
            current_config=policy.config,
            machine=detect_machine(),
            plan_path=args.plan if main_run else None,
        )

    command = {"check": "run", "run": "run", "preflight": "preflight", "status": "status"}
    forwarded = [
        command[args.command],
        "--runs-dir",
        str(args.runs_dir),
        "--root",
        str(args.root),
        "--ollama-url",
        args.ollama_url,
    ]
    if args.command == "run" and args.max_blocks is not None:
        forwarded += ["--max-blocks", str(args.max_blocks)]
    code = bench_runner.main(
        forwarded,
        policy_factory=policy_factory,
        preflight=preflight or default_preflight,
        probe=probe,
        plan=blocks if main_run else [check_block],
        status_policy=status_policy,
        rules=rules,
        session_log=L3_SESSION_LOG,
        out=out,
        err=err,
    )
    if args.command == "check" and code == 0:
        return _print_summary(args.runs_dir, status_policy, check_block, out, err)
    return code


def _write_plan(path: Path, root: Path, out: TextIO, err: TextIO) -> int:
    plan = build_l3_plan(load_pool(root), root)
    data = plan_bytes(plan)
    digest = hashlib.sha256(data).hexdigest()
    if path.exists():
        if path.read_bytes() != data:
            print(f"error: {path} exists and differs from the plan built now", file=err)
            return 1
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    episodes = sum(len(b.cases) for b in plan)
    print(f"L3 plan {path}: {len(plan)} blocks, {episodes} episodes, sha256 {digest}", file=out)
    return 0


def _print_summary(
    runs_dir: Path, status_policy: Policy | None, block: Block, out: TextIO, err: TextIO
) -> int:
    try:
        _, text = check_summary(runs_dir, status_policy or _StatusPolicy(), block)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    print(text, file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
