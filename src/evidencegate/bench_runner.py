"""Ticket 005 실행기 (§2): 실행 계획을 블록 단위로 돈다.

    uv run python -m evidencegate.bench_runner run [--max-blocks N]
    uv run python -m evidencegate.bench_runner status
    uv run python -m evidencegate.bench_runner preflight

- 세션 시작: Ollama에 연결해 정책 설정을 만들고, 실행 전 점검(§2) → 병렬 탐침 순서로 확인한다.
  하나라도 실패하면 아무것도 실행하지 않는다. 탐침은 모델 적재와 예열을 겸하고 기록하지 않는다.
- 블록은 계획 순서대로 돈다. 이미 다 저장된 블록은 건너뛴다. 블록 안에서는 아직 없는 케이스만
  동시 2로 돈다(쓰기는 이 스레드 하나). `--max-blocks N`은 실행할 일이 있던 블록 N개를 끝내면
  멈춘다. `--pilot`은 계획 맨 앞의 파일럿 두 블록만 돈다.
- 서버 오류(OllamaError): 같은 에피소드를 3회까지 재시도한다. 그래도 실패하면 저장하지 않고 다음으로
  간다(다음 실행 때 다시 돈다). 실패한 에피소드가 연속 3개면 멈춘다.
- 형식 오류: 형식 오류가 1회 이상 있었던 에피소드를 센다. 최근 200개 창(이전 세션 기록 포함,
  시작 시각 순)에서 비율이 5%를 넘으면 멈춘다. 20개째부터 판정한다. 기록은 그대로 둔다.
- 멈추면 이유와 마지막 레코드의 키·시간·형식 오류 수를 출력하고 종료 코드 2를 낸다.
  **CD·ES는 어디에도 출력하지 않는다**(진행 확인 포함).
- 세션 경계, 점검·탐침 결과, 서버 오류는 `runs/ticket005/sessions.jsonl`에 남긴다.
"""

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import (
    EPISODE_RETRIES,
    FORMAT_MAX_RATE,
    FORMAT_MIN_EPISODES,
    FORMAT_WINDOW,
    MAX_CONSECUTIVE_FAILED_EPISODES,
    PILOT_BLOCKS,
    RETRY_WAIT_S,
    RUN_WORKERS,
    SESSION_LOG,
    STATUS_RECENT,
)
from evidencegate.eval_data import CaseId
from evidencegate.ids import config_id, tool_spec_id
from evidencegate.machine import detect_machine
from evidencegate.ollama_policy import (
    PARALLEL_PROBE_MAX_RATIO,
    OllamaError,
    OllamaPolicy,
    OllamaSettings,
    probe_parallelism,
)
from evidencegate.policy import Policy
from evidencegate.preflight import Check, preflight_ok, run_preflight
from evidencegate.run_plan import DEFAULT_PLAN_PATH, Block, load_plan
from evidencegate.runner import EpisodeRecord, run_episode
from evidencegate.skills import Skill, load_skill
from evidencegate.store import EpisodeStore
from opsworld import agent_env
from opsworld.state import Family, Level

DEFAULT_OLLAMA_URL = "http://localhost:11434"
MODEL = "gpt-oss:20b"


class BenchStopped(RuntimeError):
    """중단 규칙에 걸렸다. `last`는 마지막 저장 레코드의 요약(CD·ES 없음)."""

    def __init__(self, reason: str, last: dict[str, Any] | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.last = last


@dataclass(frozen=True)
class StopRules:
    window: int = FORMAT_WINDOW
    min_episodes: int = FORMAT_MIN_EPISODES
    max_rate: Fraction = FORMAT_MAX_RATE
    retries: int = EPISODE_RETRIES
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILED_EPISODES
    retry_wait_s: float = RETRY_WAIT_S


@dataclass(frozen=True)
class RunSummary:
    blocks_run: int
    blocks_skipped: int
    episodes_run: int
    episodes_failed: int
    server_errors: int
    stopped_by_max_blocks: bool


DEFAULT_RULES = StopRules()


class FormatErrorWindow:
    """최근 `window`개 에피소드의 형식 오류 여부. `min_episodes`개부터 판정."""

    def __init__(self, rules: StopRules) -> None:
        self.rules = rules
        self.flags: deque[bool] = deque(maxlen=rules.window)
        self.seen = 0

    def add(self, has_format_error: bool) -> None:
        self.flags.append(has_format_error)
        self.seen += 1

    @property
    def errors(self) -> int:
        return sum(self.flags)

    @property
    def size(self) -> int:
        return len(self.flags)

    @property
    def rate(self) -> Fraction:
        return Fraction(self.errors, self.size) if self.size else Fraction(0)

    def exceeded(self) -> bool:
        return self.seen >= self.rules.min_episodes and self.rate > self.rules.max_rate


def block_store(runs_dir: Path, policy: Policy, block: Block) -> EpisodeStore:
    return EpisodeStore.for_run(
        runs_dir,
        suite=block.split,
        policy_id=policy.policy_id,
        config_id=config_id(policy),
        tool_spec_id=tool_spec_id(agent_env.TOOL_SPECS),
        skill_id=block.skill_id,
    )


def run_blocks(
    plan: Sequence[Block],
    policy: Policy,
    *,
    root: Path,
    runs_dir: Path,
    out: TextIO,
    workers: int = RUN_WORKERS,
    max_blocks: int | None = None,
    rules: StopRules | None = None,
) -> RunSummary:
    rules = rules or DEFAULT_RULES
    log = SessionLog(runs_dir)
    log.write("session_start", blocks=len(plan), max_blocks=max_blocks, workers=workers)
    window = FormatErrorWindow(rules)
    for record in sorted(_plan_records(plan, runs_dir, policy), key=lambda r: r.started_at):
        window.add(record.trace.format_errors > 0)
    total = sum(len(b.cases) for b in plan)
    done = sum(len(_stored_cases(block_store(runs_dir, policy, b), b)) for b in plan)
    counts = {"blocks_run": 0, "blocks_skipped": 0, "run": 0, "failed": 0, "server_errors": 0}
    last: dict[str, Any] | None = None
    start = time.perf_counter()
    try:
        for block in plan:
            if max_blocks is not None and counts["blocks_run"] >= max_blocks:
                log.write("session_end", reason="max_blocks", **_summary_fields(counts))
                print(f"max blocks reached ({max_blocks}); stopping", file=out)
                return _summary(counts, stopped_by_max_blocks=True)
            store = block_store(runs_dir, policy, block)
            skill = _load_block_skill(root, block)
            todo = [c for c in block.cases if c not in _stored_cases(store, block)]
            if not todo:
                counts["blocks_skipped"] += 1
                continue
            print(
                f"block {block.index} {block.name}/{block.split}/r{block.repeat}: "
                f"{len(todo)} to run",
                file=out,
                flush=True,
            )
            consecutive = 0
            for outcome in _execute(policy, skill, block, todo, workers, rules, log, counts):
                if isinstance(outcome, EpisodeRecord):
                    store.append(outcome)
                    done += 1
                    counts["run"] += 1
                    consecutive = 0
                    last = _record_summary(outcome)
                    window.add(outcome.trace.format_errors > 0)
                    elapsed = time.perf_counter() - start
                    eta = elapsed / counts["run"] * (total - done)
                    print(
                        f"[{done}/{total}] block {block.index} {outcome.key.family} "
                        f"{outcome.key.level} {outcome.key.incident_seed} r{outcome.key.repeat} "
                        f"{outcome.duration_s:.1f}s  ETA {_duration(eta)}",
                        file=out,
                        flush=True,
                    )
                    if window.exceeded():
                        raise BenchStopped(
                            f"format error rate {window.errors}/{window.size} in the last "
                            f"{rules.window} episodes exceeds {rules.max_rate}",
                            last,
                        )
                else:
                    counts["failed"] += 1
                    consecutive += 1
                    log.write("episode_failed", case=list(outcome[0]), error=outcome[1])
                    if consecutive >= rules.max_consecutive_failures:
                        raise BenchStopped(
                            f"{consecutive} consecutive episodes failed after "
                            f"{rules.retries} retries: {outcome[1]}",
                            last,
                        )
            counts["blocks_run"] += 1
    except BenchStopped as stop:
        log.write("session_end", reason=stop.reason, last=stop.last, **_summary_fields(counts))
        raise
    log.write("session_end", reason="plan_done", **_summary_fields(counts))
    return _summary(counts, stopped_by_max_blocks=False)


def status_line(plan: Sequence[Block], runs_dir: Path, policy: Policy) -> str:
    """한 줄 진행 확인. CD·ES는 내지 않는다."""
    records = sorted(_plan_records(plan, runs_dir, policy), key=lambda r: r.started_at)
    total = sum(len(b.cases) for b in plan)
    blocks_done = sum(
        1 for b in plan if len(_stored_cases(block_store(runs_dir, policy, b), b)) == len(b.cases)
    )
    window = FormatErrorWindow(DEFAULT_RULES)
    for r in records:
        window.add(r.trace.format_errors > 0)
    recent = records[-STATUS_RECENT:]
    speed = "n/a"
    eta = "n/a"
    if len(recent) >= 2:
        first = datetime.fromisoformat(recent[0].started_at)
        last = datetime.fromisoformat(recent[-1].started_at)
        per = (last - first).total_seconds() / (len(recent) - 1)
        speed = f"{per:.1f}s/ep (last {len(recent)})"
        eta = _duration(per * (total - len(records)))
    server_errors = SessionLog(runs_dir).count("server_error")
    return (
        f"{len(records)}/{total} episodes, {blocks_done}/{len(plan)} blocks, {speed}, "
        f"ETA {eta}, format errors (last {FORMAT_WINDOW}) {window.errors}/{window.size}, "
        f"server errors {server_errors}"
    )


class SessionLog:
    def __init__(self, runs_dir: Path) -> None:
        self.path = runs_dir / SESSION_LOG

    def write(self, event: str, **fields: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"event": event, "time": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def count(self, event: str) -> int:
        if not self.path.exists():
            return 0
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line.strip() and json.loads(line)["event"] == event)


Outcome = EpisodeRecord | tuple[CaseId, str]


def _execute(
    policy: Policy,
    skill: Skill,
    block: Block,
    todo: Sequence[CaseId],
    workers: int,
    rules: StopRules,
    log: SessionLog,
    counts: dict[str, int],
) -> Any:
    """완료되는 순서대로 레코드 또는 (케이스, 오류)를 낸다. 소비자가 멈추면 남은 일은 취소한다."""

    def attempt(case: CaseId) -> Outcome:
        family, level, seed = case
        error = ""
        for n in range(rules.retries + 1):
            try:
                return run_episode(policy, skill, Family(family), seed, block.repeat, Level(level))
            except OllamaError as exc:
                error = str(exc)
                log.write(
                    "server_error", block=block.index, case=list(case), attempt=n, error=error
                )
                counts["server_errors"] += 1
                if n < rules.retries:
                    time.sleep(rules.retry_wait_s)
        return case, error

    pool = ThreadPoolExecutor(max_workers=workers)
    pending: set[Future[Outcome]] = set()
    queue = list(todo)
    try:
        while queue or pending:
            while queue and len(pending) < workers:
                pending.add(pool.submit(attempt, queue.pop(0)))
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _plan_records(plan: Sequence[Block], runs_dir: Path, policy: Policy) -> list[EpisodeRecord]:
    records = []
    seen: set[Path] = set()
    wanted = {(b.skill_id, b.split, b.repeat, c) for b in plan for c in b.cases}
    for b in plan:
        store = block_store(runs_dir, policy, b)
        if store.path in seen:
            continue
        seen.add(store.path)
        for r in EpisodeStore(store.path, repair=False).records():
            case = (r.key.family, r.key.level, r.key.incident_seed)
            if (r.key.skill_id, b.split, r.key.repeat, case) in wanted:
                records.append(r)
    return records


def _stored_cases(store: EpisodeStore, block: Block) -> set[CaseId]:
    return {
        (r.key.family, r.key.level, r.key.incident_seed)
        for r in EpisodeStore(store.path, repair=False).records()
        if r.key.repeat == block.repeat
    } & set(block.cases)


def _load_block_skill(root: Path, block: Block) -> Skill:
    skill = load_skill(root / block.skill_path)
    if skill.skill_id != block.skill_id:
        raise ValueError(
            f"{block.skill_path} has skill_id {skill.skill_id}, expected {block.skill_id}"
        )
    return skill


def _record_summary(r: EpisodeRecord) -> dict[str, Any]:
    """중단 보고용 요약. CD·ES는 넣지 않는다."""
    return {
        "skill_id": r.key.skill_id,
        "case": [r.key.family, r.key.level, r.key.incident_seed],
        "repeat": r.key.repeat,
        "started_at": r.started_at,
        "duration_s": round(r.duration_s, 1),
        "format_errors": r.trace.format_errors,
        "stop_reason": str(r.trace.stop_reason),
    }


def _summary_fields(counts: dict[str, int]) -> dict[str, int]:
    return {
        "blocks_run": counts["blocks_run"],
        "episodes_run": counts["run"],
        "episodes_failed": counts["failed"],
        "server_errors": counts["server_errors"],
    }


def _summary(counts: dict[str, int], *, stopped_by_max_blocks: bool) -> RunSummary:
    return RunSummary(
        blocks_run=counts["blocks_run"],
        blocks_skipped=counts["blocks_skipped"],
        episodes_run=counts["run"],
        episodes_failed=counts["failed"],
        server_errors=counts["server_errors"],
        stopped_by_max_blocks=stopped_by_max_blocks,
    )


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    policy_factory: Callable[[], Policy] | None = None,
    preflight: Callable[[Policy], list[Check]] | None = None,
    probe: Callable[[], Any] | None = None,
    plan: Sequence[Block] | None = None,
    status_policy: Policy | None = None,
    rules: StopRules | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-bench")
    parser.add_argument("command", choices=("run", "status", "preflight"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument(
        "--pilot", action="store_true", help="run only the two pilot blocks at the head of the plan"
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args(argv)
    blocks = list(plan) if plan is not None else list(load_plan(args.plan))
    run_set = blocks[: len(PILOT_BLOCKS)] if args.pilot else blocks

    if args.command == "status":
        print(status_line(blocks, args.runs_dir, status_policy or _StatusPolicy()), file=out)
        return 0

    client = httpx.Client(base_url=args.ollama_url)
    settings = OllamaSettings.for_model(
        MODEL, concurrency=RUN_WORKERS, server_num_parallel=RUN_WORKERS
    )
    try:
        policy = (
            policy_factory()
            if policy_factory
            else OllamaPolicy.connect(settings, client, machine=detect_machine())
        )
    except OllamaError as error:
        print(f"error: {error}", file=err)
        return 1
    checks = (
        preflight(policy)
        if preflight
        else run_preflight(
            root=args.root,
            runs_dir=args.runs_dir,
            current_config=policy.config,
            machine=detect_machine(),
            plan_path=args.plan,
        )
    )
    for c in checks:
        print(f"preflight {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}", file=out)
    log = SessionLog(args.runs_dir)
    if not preflight_ok(checks):
        print("preflight failed; nothing was run", file=err)
        log.write("preflight_failed", checks=[c.__dict__ for c in checks])
        return 1
    if args.command == "preflight":
        return 0

    result = (
        probe()
        if probe
        else probe_parallelism(client, settings, max_ratio=PARALLEL_PROBE_MAX_RATIO)
    )
    trials = ", ".join(f"{r:.2f}" for r in result.ratios)
    print(
        f"parallel probe: ratio {result.ratio:.2f} (trials {trials}; limit {result.max_ratio:.2f})"
        f" -> {'ok' if result.parallel else 'FAILED'}",
        file=out,
    )
    log.write("probe", ratio=result.ratio, ratios=list(result.ratios), parallel=result.parallel)
    if not result.parallel:
        print(
            f"error: the server is not running requests in parallel; restart Ollama with "
            f"OLLAMA_NUM_PARALLEL={RUN_WORKERS}",
            file=err,
        )
        return 1
    try:
        summary = run_blocks(
            run_set,
            policy,
            root=args.root,
            runs_dir=args.runs_dir,
            out=out,
            max_blocks=args.max_blocks,
            rules=rules or DEFAULT_RULES,
        )
    except BenchStopped as stop:
        print(f"stopped: {stop.reason}", file=err)
        print(f"last record: {json.dumps(stop.last, ensure_ascii=False)}", file=err)
        return 2
    print(
        f"done: {summary.blocks_run} blocks run, {summary.blocks_skipped} skipped, "
        f"{summary.episodes_run} episodes, {summary.episodes_failed} failed, "
        f"{summary.server_errors} server errors",
        file=out,
    )
    print(status_line(blocks, args.runs_dir, policy), file=out)
    return 0


class _StatusPolicy:
    """진행 확인용: 저장 경로만 만든다(고정 실행 조건). 정책을 돌리지 않는다."""

    def __init__(self) -> None:
        from evidencegate.eval_constants import POLICY_ID
        from evidencegate.preflight import reference_policy_config

        self.policy_id = POLICY_ID
        self.config = reference_policy_config(Path("runs"))

    def run(self, *args: Any) -> Any:  # pragma: no cover
        raise RuntimeError("status does not run episodes")


if __name__ == "__main__":
    raise SystemExit(main())
