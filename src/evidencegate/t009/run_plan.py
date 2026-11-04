"""Ticket 009 단계별 실행 계획(사전 등록 1절, 티켓 §4). v1 `run_plan`은 고치지 않고 그 형식을 쓴다.

    uv run python -m evidencegate.t009.run_plan --phase calibration

| 단계 | 블록 | 에피소드 |
| calibration | S0 x t009_calibration x 반복 0~4 | 1,500 |
| dev_a | S0 x t009_dev_a x 반복 0 | 60 |
| main | 아래 | 풀에 따라 |

main은 proposer x (t009_dev_b, t009_oracle) 반복 0, 통제 x t009_oracle 반복 0,
S0 x t009_dev_b 반복 0, S0 x t009_oracle 반복 0~4다.

- 블록 = (스킬, split, 반복), 블록 안의 케이스는 split의 정렬 순서다(v1 `Block`).
- 블록 순서는 정해진 순서(main은 풀 순서 x split, 그다음 S0)를 `random.Random(PLAN_ORDER_SEED)`로
  섞는다(v1과 같은 규칙). 파일럿 블록은 두지 않는다. main의 파일럿은 그 단계 직전에 정한다.
- 계획 파일은 v1과 같은 바이트 형식(`plan_bytes`)이고 `<계획>.sha256`에 "sha256  이름"으로 고정한다.
  이미 있는 파일과 다르면 거부한다. 각 단계의 계획은 그 단계 직전에 동결한다.
"""

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import PLAN_ORDER_SEED, S0_ORACLE_REPEATS, S0_SKILL_ID
from evidencegate.run_plan import S0_NAME, S0_PATH, Block, BlockKey, plan_bytes
from evidencegate.skills import load_skill
from evidencegate.t009.splits import split_cases

PHASES = ("calibration", "dev_a", "main")
GATE_REPEAT = 0
POOL_PATH = Path("skills/candidates-009/POOL-009.json")


@dataclass(frozen=True)
class PoolEntry:
    name: str
    group: str  # proposer / control
    skill_id: str
    skill_path: str  # 저장소 기준


def plan_path(phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    return Path(f"plans/ticket-009-{phase.replace('_', '-')}-plan.json")


def _blocks(keys: Sequence[BlockKey], skills: dict[str, tuple[str, str]]) -> tuple[Block, ...]:
    ordered = list(keys)
    random.Random(PLAN_ORDER_SEED).shuffle(ordered)
    return tuple(
        Block(i, name, group, *skills[name], split, repeat, tuple(split_cases(split)))
        for i, (name, group, split, repeat) in enumerate(ordered)
    )


def _check_skills(root: Path, skills: dict[str, tuple[str, str]]) -> None:
    for name, (skill_id, path) in skills.items():
        actual = load_skill(root / path).skill_id
        if actual != skill_id:
            raise ValueError(f"{name}: {path} has skill_id {actual}, expected {skill_id}")


def _s0(root: Path) -> dict[str, tuple[str, str]]:
    skills = {S0_NAME: (S0_SKILL_ID, S0_PATH)}
    _check_skills(root, skills)
    return skills


def build_calibration_plan(root: Path = REPO_ROOT) -> tuple[Block, ...]:
    keys = [(S0_NAME, "s0", "t009_calibration", r) for r in S0_ORACLE_REPEATS]
    return _blocks(keys, _s0(root))


def build_dev_a_plan(root: Path = REPO_ROOT) -> tuple[Block, ...]:
    return _blocks([(S0_NAME, "s0", "t009_dev_a", GATE_REPEAT)], _s0(root))


def canonical_main_blocks(pool: Sequence[PoolEntry]) -> list[BlockKey]:
    keys: list[BlockKey] = []
    for c in pool:
        splits = ("t009_dev_b", "t009_oracle") if c.group == "proposer" else ("t009_oracle",)
        keys += [(c.name, c.group, split, GATE_REPEAT) for split in splits]
    keys.append((S0_NAME, "s0", "t009_dev_b", GATE_REPEAT))
    keys += [(S0_NAME, "s0", "t009_oracle", r) for r in S0_ORACLE_REPEATS]
    return keys


def build_main_plan(pool: Sequence[PoolEntry], root: Path = REPO_ROOT) -> tuple[Block, ...]:
    skills = {c.name: (c.skill_id, c.skill_path) for c in pool}
    _check_skills(root, skills)
    skills.update(_s0(root))
    return _blocks(canonical_main_blocks(pool), skills)


def load_pool_009(root: Path) -> list[PoolEntry]:
    """동결된 POOL-009.json에서 풀 순서대로(proposer, 그다음 통제)."""
    data = json.loads((root / POOL_PATH).read_text(encoding="utf-8"))
    return [
        PoolEntry(c["name"], c["group"], c["skill_id"], c["skill_path"]) for c in data["candidates"]
    ]


def build_phase_plan(phase: str, root: Path = REPO_ROOT) -> tuple[Block, ...]:
    if phase == "calibration":
        return build_calibration_plan(root)
    if phase == "dev_a":
        return build_dev_a_plan(root)
    if phase == "main":
        if not (root / POOL_PATH).exists():
            raise FileNotFoundError(f"{root / POOL_PATH} does not exist; freeze the pool first")
        return build_main_plan(load_pool_009(root), root)
    raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009-plan")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    out: Path = args.out or args.root / plan_path(args.phase)
    try:
        plan = build_phase_plan(args.phase, args.root)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    data = plan_bytes(plan)
    digest = hashlib.sha256(data).hexdigest()
    if out.exists():
        if out.read_bytes() != data:
            print(f"error: {out} exists and differs from the plan built now", file=sys.stderr)
            return 1
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(
        f"plan {out}: {len(plan)} blocks, {sum(len(b.cases) for b in plan)} episodes, "
        f"sha256 {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
