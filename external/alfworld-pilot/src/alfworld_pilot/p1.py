"""P1 결정성과 수량 (Ticket 008 §4 P1). LLM은 쓰지 않는다.

    PYTHONHASHSEED=0 uv run python -m alfworld_pilot.p1

- 칸은 ALFWorld 여섯 과제 유형이다. 근거는 설치된 `alfworld.agents.environment.alfred_tw_env`의
  `TASK_TYPES`(1~6)와 ALFWorld `configs/base_config.yaml`의 `task_types: [1, 2, 3, 4, 5, 6]`.
- 게임은 trial 폴더마다 `traj_data.json`의 `task_type`으로 분류한다. "플레이 가능"은 ALFWorld 로더
  (`collect_game_files`)와 같은 조건이다: 여섯 유형 중 하나, `game.tw-pddl`이 있음, `solvable`이 참.
- 파일럿 풀: 플레이 가능한 train 게임에서 유형마다 같은 수를 고정 시드로 뽑는다(원칙 5).
- 결정성(G2): 같은 게임을 리셋 3회 해 첫 관측과 가능한 명령 목록(순서 포함)을 비교하고, 첫 회의
  전문가 행동 열을 나머지 회에 그대로 넣어 관측 열과 명령 목록 열을 비교한다.
- 전문가 실행: handcoded 전문가를 끝까지 따라 걸음 수와 성공을 기록하고, 걸음마다 가능한 명령이
  가리키는 위치가 현재 위치인지 센다(E-a 전제 조건 확인).
- 기록의 경로는 `json_2.1.1` 기준 상대 경로다(홈 경로 없음).
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alfworld_pilot.p0 import _make_env, data_root

SIX_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
# ALFWorld 논문(arXiv:2010.03768) Table 1
PAPER_TABLE1: dict[str, dict[str, int]] = {
    "train": dict(zip(SIX_TYPES, (790, 308, 650, 459, 533, 813), strict=True)),
    "valid_seen": dict(zip(SIX_TYPES, (35, 13, 27, 16, 25, 24), strict=True)),
    "valid_unseen": dict(zip(SIX_TYPES, (24, 18, 31, 23, 21, 17), strict=True)),
}
SPLITS = ("train", "valid_seen", "valid_unseen", "valid_train")
POOL_PER_TYPE = 30
POOL_SEED = 0
DETERMINISM_PER_TYPE = (4, 4, 3, 3, 3, 3)  # 모두 20개(Ticket 008 §3 G2)
RESETS = 3
EXPERT_SEED = 0  # handcoded 전문가의 전역 random 시드
EXPERT_MAX_STEPS = 150  # ALFWorld base_config의 expert_timeout_steps
RECORDS = Path("records")


@dataclass(frozen=True)
class GameInfo:
    rel: str  # json_2.1.1 기준 trial 폴더 상대 경로
    task_type: str  # traj_data.json의 task_type
    folder_prefix: str  # 과제 폴더 이름의 첫 "-" 앞
    has_game: bool
    solvable: bool
    game_sha256: str | None
    traj_sha256: str

    @property
    def playable(self) -> bool:
        return self.task_type in SIX_TYPES and self.has_game and self.solvable


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_split(json_root: Path, split: str) -> list[GameInfo]:
    out = []
    for traj in sorted((json_root / split).rglob("traj_data.json")):
        trial = traj.parent
        game = trial / "game.tw-pddl"
        has_game = game.exists()
        solvable = bool(has_game and json.loads(game.read_text()).get("solvable"))
        out.append(
            GameInfo(
                rel=trial.relative_to(json_root).as_posix(),
                task_type=json.loads(traj.read_text())["task_type"],
                folder_prefix=trial.parent.name.split("-")[0],
                has_game=has_game,
                solvable=solvable,
                game_sha256=_sha(game) if has_game else None,
                traj_sha256=_sha(traj),
            )
        )
    return out


def count_table(scans: Mapping[str, Sequence[GameInfo]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for split, infos in scans.items():
        playable = Counter(i.task_type for i in infos if i.playable)
        table[split] = {
            "trials": len(infos),
            "playable": {t: playable.get(t, 0) for t in SIX_TYPES},
            "unsolvable": dict(
                sorted(Counter(i.task_type for i in infos if i.has_game and not i.solvable).items())
            ),
            "no_game_file": dict(
                sorted(Counter(i.task_type for i in infos if not i.has_game).items())
            ),
            "other_task_type_with_game": dict(
                sorted(
                    Counter(
                        i.task_type for i in infos if i.has_game and i.task_type not in SIX_TYPES
                    ).items()
                )
            ),
            "folder_prefix_mismatch": sum(1 for i in infos if i.folder_prefix != i.task_type),
        }
    return table


def compare_with_paper(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    diffs = []
    for split, paper in PAPER_TABLE1.items():
        for task_type, n in paper.items():
            data = table[split]["playable"][task_type]
            if data != n:
                diffs.append({"split": split, "task_type": task_type, "paper": n, "data": data})
    return diffs


def overlap(train: Sequence[GameInfo], valid_train: Sequence[GameInfo]) -> dict[str, Any]:
    """valid_train과 train이 겹치는가: 상대 경로(분할 이름 뺀 부분)와 game.tw-pddl sha256 기준."""

    def key(i: GameInfo) -> str:
        return i.rel.split("/", 1)[1]

    train_keys = {key(i) for i in train}
    train_sha = {i.game_sha256 for i in train if i.game_sha256}
    games = [i for i in valid_train if i.has_game]
    return {
        "valid_train_games": len(games),
        "by_relative_path": sorted(key(i) for i in games if key(i) in train_keys),
        "by_game_sha256": sorted(key(i) for i in games if i.game_sha256 in train_sha),
    }


def pilot_pool(games: Sequence[GameInfo], per_type: int, seed: int) -> list[GameInfo]:
    """플레이 가능한 train 게임에서 유형마다 per_type개를 고정 시드로 뽑는다. 결과는 경로순."""
    if any(not g.rel.startswith("train/") for g in games):
        raise ValueError("the pilot pool is drawn from the train split only")
    rng = random.Random(seed)
    pool: list[GameInfo] = []
    for task_type in SIX_TYPES:
        candidates = sorted(
            (g for g in games if g.playable and g.task_type == task_type), key=lambda g: g.rel
        )
        if len(candidates) < per_type:
            raise ValueError(f"{task_type}: only {len(candidates)} playable train games")
        pool += rng.sample(candidates, per_type)
    return sorted(pool, key=lambda g: g.rel)


_VERB_TARGET = (
    ("put ", " in/on "),
    ("move ", " to "),  # alfworld 0.4.2의 놓기 형식
    ("heat ", " with "),
    ("cool ", " with "),
    ("clean ", " with "),
    ("take ", " from "),
)


def location_targets(command: str) -> str | None:
    """명령이 작용하는 수용체(위치). 이동·둘러보기 명령은 None."""
    for verb, sep in _VERB_TARGET:
        if command.startswith(verb) and sep in command:
            return command.split(sep, 1)[1]
    for verb in ("open ", "close ", "use ", "examine "):
        if command.startswith(verb):
            return command[len(verb) :]
    return None


def _verb(command: str) -> str:
    return command.split(" ", 1)[0]


def expert_run(
    game_file: Path, max_steps: int = EXPERT_MAX_STEPS, seed: int = EXPERT_SEED
) -> dict[str, Any]:
    """handcoded 전문가를 끝까지 따른다. 가능한 명령이 현재 위치를 가리키는지 동사별로 센다.

    ALFWorld handcoded 전문가는 시드 없는 전역 `random`으로 물건과 대체 명령을 고른다
    (`handcoded_expert.py`). 환경은 결정적이지만 전문가는 아니므로, 실행마다 전역 시드를 고정한다.
    파일럿 코드의 무작위성은 여전히 `random.Random(seed)`만 쓴다. 이 시드는 외부 전문가용이다.
    """
    random.seed(seed)
    env = _make_env(game_file, max_steps)
    _obs, infos = env.reset()
    infos_keys = sorted(infos)
    location: str | None = None
    at_location: Counter[str] = Counter()
    elsewhere: Counter[str] = Counter()
    actions: list[str] = []
    done, won = False, False
    fallback = 0
    while not done and len(actions) < max_steps:
        for command in infos["admissible_commands"]:
            target = location_targets(command)
            if target is not None:
                (at_location if target == location else elsewhere)[_verb(command)] += 1
        plan = infos.get("extra.expert_plan") or []
        if not plan:
            fallback += 1
        command = plan[0] if plan else infos["admissible_commands"][0]
        _obs, _score, done, infos = env.step(command)
        actions.append(command)
        if command.startswith("go to "):
            location = command[len("go to ") :]
        won = bool(infos.get("won"))
    env.close()
    return {
        "steps": len(actions),
        "won": won,
        "done": bool(done),
        "actions": actions,
        "fallback_steps": fallback,
        "infos_keys": infos_keys,
        "admissible_target_at_location": dict(sorted(at_location.items())),
        "admissible_target_elsewhere": dict(sorted(elsewhere.items())),
    }


def determinism_check(game_files: Sequence[Path], resets: int = RESETS) -> list[dict[str, Any]]:
    out = []
    for game_file in game_files:
        env = _make_env(game_file, EXPERT_MAX_STEPS)
        firsts = []
        replays = []
        actions: list[str] = []
        for r in range(resets):
            random.seed(EXPERT_SEED)  # 전문가 행동 열을 expert_run과 같게(reset에서 계획을 만든다)
            obs, infos = env.reset()
            firsts.append((obs, list(infos["admissible_commands"])))
            seq = []
            if r == 0:
                done = False
                while not done and len(actions) < EXPERT_MAX_STEPS:
                    plan = infos.get("extra.expert_plan") or []
                    command = plan[0] if plan else infos["admissible_commands"][0]
                    obs, _s, done, infos = env.step(command)
                    actions.append(command)
                    seq.append((obs, list(infos["admissible_commands"])))
            else:
                for command in actions:
                    obs, _s, done, infos = env.step(command)
                    seq.append((obs, list(infos["admissible_commands"])))
            replays.append(seq)
        env.close()
        out.append(
            {
                "game": game_file.parent.relative_to(game_file.parents[3]).as_posix(),
                "same_first": all(f == firsts[0] for f in firsts),
                "same_first_as_sets": all(
                    f[0] == firsts[0][0] and set(f[1]) == set(firsts[0][1]) for f in firsts
                ),
                "same_replay": all(s == replays[0] for s in replays),
                "replay_steps": len(actions),
            }
        )
    return out


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 3),
        "max": max(values),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-p1")
    parser.add_argument("--out-dir", type=Path, default=RECORDS)
    args = parser.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        # handcoded 전문가는 집합 순회 순서에도 기댄다. 해시 시드는 인터프리터 시작 때 정해진다.
        print(
            "error: run with PYTHONHASHSEED=0 (the ALFWorld expert depends on it)", file=sys.stderr
        )
        return 2
    json_root = data_root() / "json_2.1.1"
    if not json_root.exists():
        print("error: no ALFWorld data", file=sys.stderr)
        return 1
    scans = {s: scan_split(json_root, s) for s in SPLITS}
    table = count_table(scans)
    pool = pilot_pool(scans["train"], POOL_PER_TYPE, POOL_SEED)
    counts: dict[str, Any] = {
        "six_types": list(SIX_TYPES),
        "table": table,
        "differs_from_paper_table1": compare_with_paper(table),
        "valid_train_overlap": overlap(scans["train"], scans["valid_train"]),
    }
    pool_record = {
        "per_type": POOL_PER_TYPE,
        "seed": POOL_SEED,
        "games": [asdict(g) for g in pool],
    }
    det_games = []
    for task_type, k in zip(SIX_TYPES, DETERMINISM_PER_TYPE, strict=True):
        det_games += [g for g in pool if g.task_type == task_type][:k]
    determinism = determinism_check([json_root / g.rel / "game.tw-pddl" for g in det_games])
    runs = {g.rel: expert_run(json_root / g.rel / "game.tw-pddl") for g in pool}
    at_loc: Counter[str] = Counter()
    away: Counter[str] = Counter()
    for r in runs.values():
        at_loc.update(r["admissible_target_at_location"])
        away.update(r["admissible_target_elsewhere"])
    expert = {
        "max_steps": EXPERT_MAX_STEPS,
        "per_type": {
            t: {
                "won": sum(1 for g in pool if g.task_type == t and runs[g.rel]["won"]),
                "steps_when_won": _summary(
                    [runs[g.rel]["steps"] for g in pool if g.task_type == t and runs[g.rel]["won"]]
                    or [0]
                ),
            }
            for t in SIX_TYPES
        },
        "infos_keys": sorted({k for r in runs.values() for k in r["infos_keys"]}),
        "admissible_target_at_location": dict(sorted(at_loc.items())),
        "admissible_target_elsewhere": dict(sorted(away.items())),
        "games": {
            rel: {k: r[k] for k in ("steps", "won", "done", "fallback_steps", "actions")}
            for rel, r in runs.items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("p1_counts.json", counts),
        ("p1_pilot_pool.json", pool_record),
        ("p1_determinism.json", {"resets": RESETS, "games": determinism}),
        ("p1_expert.json", expert),
    ):
        (args.out_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "differs": counts["differs_from_paper_table1"],
                "overlap": counts["valid_train_overlap"]["by_game_sha256"][:3],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
