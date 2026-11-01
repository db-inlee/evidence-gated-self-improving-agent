"""P2 실행기: 계획 → 에피소드 기록(jsonl), 이어서 실행, 중단 규칙, 원칙 7 점검.

    PYTHONHASHSEED=0 caffeinate -ims uv run python -m alfworld_pilot.runner \\
        --per-type 2 --repeats 1 --out runs/p3/episodes.jsonl

- 계획: 게임(경로순) x 반복. LLM seed는 (게임 경로, 반복)에서 sha256으로 정한다. 같은 게임·반복이면
  S0와 탐침이 같은 seed를 쓴다(paired).
- 원칙 7 점검은 시작할 때 한 번, 그리고 블록(BLOCK_SIZE 에피소드)이 끝날 때마다 한다. 어긋나면
  `RunStopped`로 멈춘다. 이미 쓴 기록은 남는다.
- 서버 오류나 그 밖의 예외는 그 에피소드를 기록하지 않고 멈춘다. 같은 명령으로 이어서 돌면 기록된
  에피소드는 건너뛴다. 기록의 설정(config)이 지금 설정과 다르면 이어서 돌지 않는다.
- 기록은 원시 관측만 담는다(행동, 관측, 가능 여부, won, 종료 이유, 사용량). E-a·E-c는 분석에서
  계산한다. `to_v1_like`는 009에서 v1 분석 함수에 넣을 변환기 자리다.
- LLM 실행은 사용자 확인 뒤에만 한다(Ticket 008 §4).
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


def _normalized(config: dict[str, Any]) -> Any:
    return json.loads(json.dumps(config, sort_keys=True))


def run_plan(
    plan: Sequence[Episode],
    out: Path,
    *,
    run_episode: Callable[[Episode], EpisodeTrace],
    check_power: Callable[[], None],
    run_config: dict[str, Any],
) -> None:
    config = _normalized(run_config)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            record = json.loads(line)
            if record["config"] != config:
                raise RunStopped(f"{out}: recorded config differs from the current config")
            done.add(record["key"])
    todo = [e for e in plan if e.key not in done]

    def check(where: str) -> None:
        try:
            check_power()
        except PowerCheckError as error:
            raise RunStopped(f"power check failed {where}: {error}") from error

    check("at start")
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, episode in enumerate(todo):
        try:
            trace = run_episode(episode)
        except Exception as error:  # 서버·환경 오류: 그 에피소드는 기록하지 않는다
            raise RunStopped(f"{episode.key}: {error}") from error
        record = {
            "key": episode.key,
            "rel": episode.rel,
            "task_type": episode.task_type,
            "repeat": episode.repeat,
            "llm_seed": episode.llm_seed,
            "config": config,
            **trace.as_dict(),
        }
        with out.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if (i + 1) % BLOCK_SIZE == 0 or i + 1 == len(todo):
            check(f"after block ending at {episode.key}")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-runner")
    parser.add_argument("--per-type", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skill", type=Path, default=S0_SKILL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != C.REQUIRED_HASH_SEED:
        print("error: run with PYTHONHASHSEED=0 (decision 0)", file=sys.stderr)
        return 2
    pool = json.loads(POOL.read_text())["games"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for g in sorted(pool, key=lambda g: str(g["rel"])):
        by_type.setdefault(g["task_type"], []).append(g)
    games = [g for gs in by_type.values() for g in gs[: args.per_type]]
    settings = PolicySettings()
    chat = OllamaChat()
    skill_text = args.skill.read_text()
    run_config = {
        "policy": settings.config(),
        "server": chat.server_record(settings.model),
        "skill_sha256": _sha(args.skill),
        "hash_seed": C.REQUIRED_HASH_SEED,
    }
    json_root = data_root() / "json_2.1.1"

    def one(episode: Episode) -> EpisodeTrace:
        env = AlfworldEnv(json_root / episode.rel / "game.tw-pddl", settings.max_steps)
        try:
            return run_episode(env, skill_text, chat, seed=episode.llm_seed, settings=settings)
        finally:
            env.close()

    try:
        run_plan(
            plan_episodes(games, args.repeats),
            args.out,
            run_episode=one,
            check_power=check_power,
            run_config=run_config,
        )
    except RunStopped as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
