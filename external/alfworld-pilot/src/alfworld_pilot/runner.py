"""P2 실행기: 계획 → 에피소드 기록(jsonl), 이어서 실행, 중단 규칙, 원칙 7 점검.

    PYTHONHASHSEED=0 caffeinate -ims uv run python -m alfworld_pilot.runner \\
        --per-type 2 --repeats 1 --out runs/p3/episodes.jsonl
    # P4-b: 한 유형의 경로순 13~24번째, 계획만 보기
    ... --task-type T --offset 12 --per-type 12 --repeats 3 --out ... --list

- 계획: 게임(경로순) x 반복. LLM seed는 (게임 경로, 반복)에서 sha256으로 정한다. 같은 게임·반복이면
  S0와 탐침이 같은 seed를 쓴다(paired).
- 원칙 7 점검은 시작할 때 한 번, 그리고 블록(BLOCK_SIZE 에피소드)이 끝날 때마다 한다. 어긋나면
  `RunStopped`로 멈춘다. 이미 쓴 기록은 남는다.
- 서버 오류나 그 밖의 예외는 그 에피소드를 기록하지 않고 멈춘다. 같은 명령으로 이어서 돌면 기록된
  에피소드는 건너뛴다. 기록의 설정(config)이 지금 설정과 다르면 이어서 돌지 않는다.
- 기록은 원시 관측만 담는다(행동, 관측, 가능 여부, won, 종료 이유, 사용량). E-a·E-c는 분석에서
  계산한다. `to_v1_like`는 009에서 v1 분석 함수에 넣을 변환기 자리다.
- 동시 실행(2026-11-01 결정 5): `RunSettings.concurrency`가 2 이상이면 spawn 프로세스 풀에서
  블록 안의 에피소드를 나눠 돈다. 결과는 에피소드의 seed와 게임에만 기대므로 동시 수와 무관하다.
  기록은 부모 프로세스 하나만 쓰고, 에피소드마다 한 줄을 한 번에 쓴 뒤 fsync한다(부분 기록 없음).
  한 작업자에서 예외가 나면 아직 시작하지 않은 에피소드를 취소하고, 이미 돌고 있던 에피소드가 끝까지
  마치면 그것만 기록한 뒤 멈춘다. 블록 단위로 돌기 때문에 다음 블록은 시작하지 않는다.
- 동시 수는 정책 설정(config)이 아니라 실행 설정(`run`)으로 기록한다. 이어서 돌 때 동시 수는
  바뀌어도 된다. 시간 필드(`TIME_FIELDS`)를 빼면 기록 내용은 동시 수와 같다.
- LLM 실행은 사용자 확인 뒤에만 한다(Ticket 008 §4).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from alfworld_pilot import constants as C
from alfworld_pilot.agent import (
    AlfworldEnv,
    EpisodeTrace,
    OllamaChat,
    PolicySettings,
    derive_llm_seed,
    run_episode,
)
from alfworld_pilot.p0 import data_root
from alfworld_pilot.power import PowerCheckError, check_power

BLOCK_SIZE = C.BLOCK_SIZE
POOL = Path("records/p1_pilot_pool.json")
S0_SKILL = Path("skills/S0/SKILL.md")
TIME_FIELDS = ("wall_s", "started_at", "finished_at", "run")  # 동시 수와 벽시계에 따라 바뀌는 필드


class RunStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class Episode:
    rel: str
    task_type: str
    repeat: int
    llm_seed: int

    @property
    def key(self) -> str:
        return f"{self.rel}#{self.repeat}"


def plan_episodes(games: Sequence[dict[str, Any]], repeats: int) -> list[Episode]:
    return [
        Episode(g["rel"], g["task_type"], r, derive_llm_seed(g["rel"], r))
        for g in sorted(games, key=lambda g: str(g["rel"]))
        for r in range(repeats)
    ]


@dataclass(frozen=True)
class RunSettings:
    """실행 설정(정책 설정 아님). server_num_parallel은 서버 OLLAMA_NUM_PARALLEL 선언값이다."""

    concurrency: int = 1
    server_num_parallel: int = 1

    def __post_init__(self) -> None:
        if self.concurrency < 1 or self.server_num_parallel < 1:
            raise ValueError("concurrency and server_num_parallel must be >= 1")
        if self.concurrency > self.server_num_parallel:
            raise ValueError(
                f"concurrency {self.concurrency} exceeds server_num_parallel "
                f"{self.server_num_parallel}"
            )


Timed = tuple[EpisodeTrace, float, float]


def _timed(worker: Callable[[Episode], EpisodeTrace], episode: Episode) -> Timed:
    started = time.time()
    trace = worker(episode)
    return trace, started, time.time()


def select_games(
    pool: Sequence[dict[str, Any]],
    per_type: int,
    *,
    task_type: str | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """풀에서 과제 유형마다 경로순 [offset, offset + per_type) 게임(2026-11-03 결정 1).

    task_type을 주면 그 유형만 고른다. 유형 순서는 경로순으로 처음 나오는 순서다(P3 규칙과 같음).
    """
    if per_type < 1 or offset < 0:
        raise ValueError("per_type must be >= 1 and offset >= 0")
    by_type: dict[str, list[dict[str, Any]]] = {}
    for g in sorted(pool, key=lambda g: str(g["rel"])):
        by_type.setdefault(g["task_type"], []).append(g)
    if task_type is not None:
        if task_type not in by_type:
            raise ValueError(f"unknown task type {task_type!r}")
        by_type = {task_type: by_type[task_type]}
    out = []
    for t, games in by_type.items():
        chosen = games[offset : offset + per_type]
        if len(chosen) < per_type:
            raise ValueError(f"{t}: only {len(chosen)} games from offset {offset}")
        out += chosen
    return out


def first_per_type(pool: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """풀에서 과제 유형마다 경로순 앞 n개(P3 선택 규칙, 2026-11-01 결정 4)."""
    return select_games(pool, n)


def _normalized(config: dict[str, Any]) -> Any:
    return json.loads(json.dumps(config, sort_keys=True))


def _read_done(out: Path, config: Any) -> set[str]:
    done: set[str] = set()
    if not out.exists():
        return done
    for n, line in enumerate(out.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunStopped(f"{out}: line {n} is not a complete record") from error
        if record["config"] != config:
            raise RunStopped(f"{out}: recorded config differs from the current config")
        if record["key"] in done:
            raise RunStopped(f"{out}: line {n} repeats {record['key']}")
        done.add(record["key"])
    return done


def run_plan(
    plan: Sequence[Episode],
    out: Path,
    *,
    run_episode: Callable[[Episode], EpisodeTrace],
    check_power: Callable[[], None],
    run_config: dict[str, Any],
    run_settings: RunSettings | None = None,
) -> None:
    """동시 수가 2 이상이면 run_episode는 pickle 가능해야 한다(spawn 프로세스에서 돈다)."""
    settings = run_settings or RunSettings()
    config = _normalized(run_config)
    done = _read_done(out, config)
    todo = [e for e in plan if e.key not in done]

    def check(where: str) -> None:
        try:
            check_power()
        except PowerCheckError as error:
            raise RunStopped(f"power check failed {where}: {error}") from error

    def write(episode: Episode, result: Timed) -> None:
        trace, started, finished = result
        record = {
            "key": episode.key,
            "rel": episode.rel,
            "task_type": episode.task_type,
            "repeat": episode.repeat,
            "llm_seed": episode.llm_seed,
            "config": config,
            "run": asdict(settings),
            "started_at": started,
            "finished_at": finished,
            **trace.as_dict(),
        }
        line = (json.dumps(record, sort_keys=True) + "\n").encode()
        fd = os.open(out, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)  # 한 줄을 한 번에
            os.fsync(fd)
        finally:
            os.close(fd)

    check("at start")
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = [todo[i : i + BLOCK_SIZE] for i in range(0, len(todo), BLOCK_SIZE)]
    if settings.concurrency == 1:
        for block in blocks:
            for episode in block:
                try:
                    result = _timed(run_episode, episode)
                except Exception as error:  # 서버·환경 오류: 그 에피소드는 기록하지 않는다
                    raise RunStopped(f"{episode.key}: {error}") from error
                write(episode, result)
            check(f"after block ending at {block[-1].key}")
        return

    with ProcessPoolExecutor(settings.concurrency, mp_context=get_context("spawn")) as pool:
        for block in blocks:
            futures: dict[Future[Timed], Episode] = {
                pool.submit(_timed, run_episode, e): e for e in block
            }
            finished, pending = wait(futures, return_when=FIRST_EXCEPTION)
            failed = [f for f in finished if f.exception() is not None]
            if failed:
                for f in pending:
                    f.cancel()  # 아직 시작하지 않은 에피소드
                rest, _ = wait(pending)  # 이미 돌던 에피소드는 끝까지 기다린다
                finished |= rest
            # 기록은 계획 순서로(부모 하나만 쓴다)
            for f in sorted(finished, key=lambda f: block.index(futures[f])):
                if not f.cancelled() and f.exception() is None:
                    write(futures[f], f.result())
            if failed:
                first = min(failed, key=lambda f: block.index(futures[f]))
                raise RunStopped(f"{futures[first].key}: {first.exception()}")
            check(f"after block ending at {block[-1].key}")


def to_v1_like(record: dict[str, Any]) -> dict[str, Any]:
    """009 변환기 자리: 칸 = 과제 유형, 장애 = 게임, Correct = won, Evidence-Safe 대응 = E-a."""
    return {
        "family": record["task_type"],
        "incident": record["rel"],
        "repeat": record["repeat"],
        "correct": record["won"],
        "evidence_safe": record["e_a"],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AlfworldWorker:
    """실제 에피소드 작업자. spawn 프로세스로 보내므로 pickle 가능한 값만 가진다."""

    skill_text: str
    settings: PolicySettings
    json_root: str
    base_url: str = C.OLLAMA_URL

    def __call__(self, episode: Episode) -> EpisodeTrace:
        env = AlfworldEnv(
            Path(self.json_root) / episode.rel / "game.tw-pddl", self.settings.max_steps
        )
        try:
            return run_episode(
                env,
                self.skill_text,
                OllamaChat(self.base_url),
                seed=episode.llm_seed,
                settings=self.settings,
            )
        finally:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-runner")
    parser.add_argument("--per-type", type=int, required=True)
    parser.add_argument("--task-type", default=None, help="only this task type")
    parser.add_argument("--offset", type=int, default=0, help="skip this many games per type")
    parser.add_argument(
        "--list", action="store_true", help="print the plan and exit (no server, no run)"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skill", type=Path, default=S0_SKILL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=C.CONCURRENCY)
    parser.add_argument(
        "--server-num-parallel",
        type=int,
        default=C.SERVER_NUM_PARALLEL,
        help="OLLAMA_NUM_PARALLEL the server was started with (declared; not readable via API)",
    )
    args = parser.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != C.REQUIRED_HASH_SEED:
        print("error: run with PYTHONHASHSEED=0 (decision 0)", file=sys.stderr)
        return 2
    run_settings = RunSettings(args.concurrency, args.server_num_parallel)
    games = select_games(
        json.loads(POOL.read_text())["games"],
        args.per_type,
        task_type=args.task_type,
        offset=args.offset,
    )
    plan = plan_episodes(games, args.repeats)
    if args.list:
        for e in plan:
            print(json.dumps({"key": e.key, "task_type": e.task_type, "llm_seed": e.llm_seed}))
        return 0
    settings = PolicySettings()
    run_config = {
        "policy": settings.config(),
        "server": OllamaChat(C.OLLAMA_URL).server_record(settings.model),
        "skill_sha256": _sha(args.skill),
        "hash_seed": C.REQUIRED_HASH_SEED,
    }
    worker = AlfworldWorker(args.skill.read_text(), settings, str(data_root() / "json_2.1.1"))
    try:
        run_plan(
            plan,
            args.out,
            run_episode=worker,
            check_power=check_power,
            run_config=run_config,
            run_settings=run_settings,
        )
    except RunStopped as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
