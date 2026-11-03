"""하네스 진단 H1·H2 — exploratory diagnostic, not used for G1-G6 (2026-11-03 결정 3).

    PYTHONHASHSEED=0 caffeinate -ims uv run python -m alfworld_pilot.diag --variant H1 \\
        --out runs/diag_h1/episodes.jsonl

목적: No-Go가 ALFWorld 때문인지 하네스 때문인지 가려 009 정책 설계에 쓴다.

- H1: 텍스트 응답에는 재촉만 하고 연속 텍스트로 끝내지 않는다. `max_consecutive_text_turns`를 LLM
  호출 상한 + 1로 둔다. 걸음 상한 50과 호출 상한 100은 그대로다.
- H2: 매 요청에서 가장 최근 관측 뒤에 가능한 명령 목록을 붙인다
  (`run_episode`의 `observation_suffix`). 이력과 기록에는 원래 관측만 남는다.
  그 밖은 S0 설정 그대로다.
- 게임: P4-a 게임 중 칸마다 경로순 앞 6개, 반복 0(seed가 P4-a 반복 0과 같아 짝 비교한다).
- 고정 출처를 바꾸지 않는다. 실행기의 `run_plan`, 원칙 7 점검, 기록 형식을 그대로 쓴다.
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alfworld_pilot import constants as C
from alfworld_pilot.agent import (
    AlfworldEnv,
    EpisodeTrace,
    OllamaChat,
    PolicySettings,
    run_episode,
)
from alfworld_pilot.p0 import data_root
from alfworld_pilot.power import check_power
from alfworld_pilot.runner import (
    POOL,
    S0_SKILL,
    Episode,
    RunSettings,
    RunStopped,
    plan_episodes,
    run_plan,
    select_games,
)

LABEL = "exploratory diagnostic, not used for G1-G6"
VARIANTS = ("H1", "H2")
DIAG_PER_TYPE = 6
DIAG_REPEATS = 1  # 반복 0만
SUFFIX_PREFIX = "Available commands: "


def h1_settings() -> PolicySettings:
    return PolicySettings(max_consecutive_text_turns=C.MAX_LLM_CALLS + 1)


def _settings(variant: str) -> PolicySettings:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return h1_settings() if variant == "H1" else PolicySettings()


def diag_plan(pool: Sequence[dict[str, Any]]) -> list[Episode]:
    return plan_episodes(select_games(pool, DIAG_PER_TYPE), DIAG_REPEATS)


def diag_run_config(variant: str, *, skill_sha256: str, server: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(variant)
    return {
        "policy": settings.config(),
        "server": server,
        "skill_sha256": skill_sha256,
        "hash_seed": C.REQUIRED_HASH_SEED,
        "diagnostic": {
            "label": LABEL,
            "variant": variant,
            "command_list_in_prompt": variant == "H2",
            "method": (
                "no stop on consecutive text replies"
                if variant == "H1"
                else "command list appended to the latest observation in each request"
            ),
        },
    }


@dataclass(frozen=True)
class DiagWorker:
    """spawn 프로세스로 보내는 진단 작업자(pickle 가능한 값만)."""

    variant: str
    skill_text: str
    json_root: str
    base_url: str = C.OLLAMA_URL
    settings: PolicySettings = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", _settings(self.variant))

    def __call__(self, episode: Episode) -> EpisodeTrace:
        env = AlfworldEnv(
            Path(self.json_root) / episode.rel / "game.tw-pddl", self.settings.max_steps
        )
        suffix = None
        if self.variant == "H2":
            suffix = lambda: SUFFIX_PREFIX + ", ".join(env.admissible_commands())  # noqa: E731
        try:
            return run_episode(
                env,
                self.skill_text,
                OllamaChat(self.base_url),
                seed=episode.llm_seed,
                settings=self.settings,
                observation_suffix=suffix,
            )
        finally:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-diag")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=C.CONCURRENCY)
    parser.add_argument("--server-num-parallel", type=int, default=C.SERVER_NUM_PARALLEL)
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != C.REQUIRED_HASH_SEED:
        print("error: run with PYTHONHASHSEED=0 (decision 0)", file=sys.stderr)
        return 2
    plan = diag_plan(json.loads(POOL.read_text())["games"])
    if args.list:
        for e in plan:
            print(json.dumps({"key": e.key, "task_type": e.task_type, "llm_seed": e.llm_seed}))
        return 0
    skill = S0_SKILL
    run_config = diag_run_config(
        args.variant,
        skill_sha256=hashlib.sha256(skill.read_bytes()).hexdigest(),
        server=OllamaChat(C.OLLAMA_URL).server_record(C.MODEL),
    )
    worker = DiagWorker(args.variant, skill.read_text(), str(data_root() / "json_2.1.1"))
    try:
        run_plan(
            plan,
            args.out,
            run_episode=worker,
            check_power=check_power,
            run_config=run_config,
            run_settings=RunSettings(args.concurrency, args.server_num_parallel),
        )
    except RunStopped as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
