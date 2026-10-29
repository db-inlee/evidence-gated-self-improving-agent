"""P0 환경 기록과 스모크 실행 (Ticket 008 §4 P0, G1 근거).

    uv run python -m alfworld_pilot.p0 --install-route arm64-native

- 환경 기록: Python 버전과 아키텍처, 주요 패키지 버전, 데이터 위치(홈 경로는 `~`로), 폴더별 크기,
  분할별 게임 파일 수. `records/p0_environment.json`에 쓴다.
- 스모크 실행: train 게임 하나를 손으로 짠 전문가(handcoded) 계획대로 끝까지 돈다. 텍스트 전용
  설치에서 전문가 계획과 가능한 명령 목록이 나오는지도 함께 본다(Ticket 008 §6 위험 6).
- LLM은 쓰지 않는다.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SPLITS = ("train", "valid_seen", "valid_unseen", "valid_train")
PACKAGES = ("alfworld", "textworld", "fast-downward-textworld", "jericho", "opsworld")
DATA_DIRS = ("json_2.1.1", "detectors", "logic")
GAME_FILE = "game.tw-pddl"
SMOKE_TASK = "pick_heat_then_place_in_recep"
DEFAULT_RECORD = Path("records/p0_environment.json")


def data_root() -> Path:
    return Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))


def _display(path: Path) -> str:
    """홈 경로는 `~`로 적는다(기록에 사용자 이름을 남기지 않음)."""
    home = Path.home()
    try:
        return "~/" + path.resolve().relative_to(home.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def count_games(json_root: Path) -> dict[str, int]:
    return {split: len(list((json_root / split).rglob(GAME_FILE))) for split in SPLITS}


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_record(cache: Path, *, install_route: str) -> dict[str, Any]:
    return {
        "install_route": install_route,
        "python": {
            "version": platform.python_version(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
        },
        "os": {"system": platform.system(), "release": platform.mac_ver()[0] or platform.release()},
        "packages": package_versions(),
        "data": {
            "location": _display(cache),
            "bytes": {d: dir_bytes(cache / d) for d in DATA_DIRS if (cache / d).exists()},
            "games": count_games(cache / "json_2.1.1"),
        },
    }


def _make_env(game_file: Path, max_steps: int) -> Any:
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )

    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, extras=["gamefile", "expert_plan"]
    )
    env_id = textworld.gym.register_game(
        str(game_file),
        request_infos,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)],
    )
    return textworld.gym.make(env_id)


def smoke_run(game_file: Path, max_steps: int = 50) -> dict[str, Any]:
    """전문가 계획의 첫 행동을 따라 끝까지 돈다. 계획이 없으면 첫 가능한 명령을 쓴다."""
    env = _make_env(game_file, max_steps)
    start = time.perf_counter()
    obs, infos = env.reset()
    first_obs_sha = hashlib.sha256(obs.encode()).hexdigest()
    expert_seen = bool(infos.get("extra.expert_plan"))
    admissible_seen = bool(infos.get("admissible_commands"))
    steps, done, won = 0, False, False
    while not done and steps < max_steps:
        plan = infos.get("extra.expert_plan") or []
        command = plan[0] if plan else infos["admissible_commands"][0]
        obs, _score, done, infos = env.step(command)
        steps += 1
        won = bool(infos.get("won"))
        expert_seen = expert_seen or bool(plan)
    env.close()
    return {
        "game": _display(game_file),
        "game_sha256": hashlib.sha256(game_file.read_bytes()).hexdigest(),
        "first_observation_sha256": first_obs_sha,
        "steps": steps,
        "done": bool(done),
        "won": won,
        "expert_plan_available": expert_seen,
        "admissible_commands_available": admissible_seen,
        "seconds": round(time.perf_counter() - start, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-p0")
    parser.add_argument("--install-route", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args(argv)
    cache = data_root()
    record = environment_record(cache, install_route=args.install_route)
    games = sorted((cache / "json_2.1.1" / "train").glob(f"{SMOKE_TASK}-*/*/{GAME_FILE}"))
    if not games:
        print(f"error: no {SMOKE_TASK} game under {_display(cache)}", file=sys.stderr)
        return 1
    record["smoke_run"] = smoke_run(games[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["smoke_run"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
