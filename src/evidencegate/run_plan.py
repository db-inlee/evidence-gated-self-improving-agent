"""Ticket 005 실행 계획 (§2, §4): 12,540 에피소드를 블록 63개로.

    uv run python -m evidencegate.run_plan --out plans/ticket-005-plan.json

- 블록 = (스킬, split, 반복). 후보 19 x 3 split(반복 0), S0 Dev-B 반복 0, S0 Oracle 반복 0~4.
  S0 Dev-A는 Ticket 004에서 실행했으므로 넣지 않는다(다시 돌리지 않음).
- 블록 안의 케이스 순서는 split의 정렬 순서(칸, incident_seed)다.
- 블록 순서: 파일럿 두 블록을 맨 앞에 두고, 나머지는 정해진 순서(후보 풀 순서 x split, 그다음
  S0)를 `random.Random(0)`으로 섞는다.
- 계획 파일은 정렬된 키의 JSON이고, sha256으로 고정한다. 같은 풀이면 언제나 같은 바이트다.
"""

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT, PoolSpec, load_pool
from evidencegate.eval_constants import (
    EVAL_SPLITS,
    GATE_REPEAT,
    PILOT_BLOCKS,
    PLAN_ORDER_SEED,
    S0_RUN_SPLITS,
)
from evidencegate.eval_data import CaseId, split_cases
from evidencegate.skills import load_skill

PLAN_VERSION = 1
DEFAULT_PLAN_PATH = Path("plans/ticket-005-plan.json")
S0_NAME = "S0"
S0_PATH = "skills/S0/SKILL.md"

BlockKey = tuple[str, str, str, int]  # (이름, 집합, split, 반복)


@dataclass(frozen=True)
class Block:
    index: int
    name: str
    group: str  # proposer / control / s0
    skill_id: str
    skill_path: str  # 저장소 기준
    split: str
    repeat: int
    cases: tuple[CaseId, ...]


def canonical_blocks(pool: PoolSpec) -> list[BlockKey]:
    keys: list[BlockKey] = [
        (c.name, c.group, split, GATE_REPEAT) for c in pool.candidates for split in EVAL_SPLITS
    ]
    keys += [(S0_NAME, "s0", split, r) for split, repeats in S0_RUN_SPLITS for r in repeats]
    return keys


def build_plan(pool: PoolSpec, root: Path = REPO_ROOT) -> tuple[Block, ...]:
    skills = {c.name: (c.skill_id, _skill_path(c.name, c.group)) for c in pool.candidates}
    skills[S0_NAME] = (pool.s0_skill_id, S0_PATH)
    for name, (skill_id, path) in skills.items():
        actual = load_skill(root / path).skill_id
        if actual != skill_id:
            raise ValueError(f"{name}: {path} has skill_id {actual}, expected {skill_id}")

    keys = canonical_blocks(pool)
    pilot = [k for p in PILOT_BLOCKS for k in keys if (k[0], k[2]) == p]
    if len(pilot) != len(PILOT_BLOCKS):
        raise ValueError(f"pilot blocks {PILOT_BLOCKS} are not all in the pool")
    rest = [k for k in keys if k not in pilot]
    random.Random(PLAN_ORDER_SEED).shuffle(rest)
    blocks = []
    for index, (name, group, split, repeat) in enumerate(pilot + rest):
        skill_id, path = skills[name]
        blocks.append(
            Block(index, name, group, skill_id, path, split, repeat, tuple(split_cases(split)))
        )
    return tuple(blocks)


def plan_json(plan: Sequence[Block]) -> dict[str, Any]:
    return {
        "version": PLAN_VERSION,
        "episodes": sum(len(b.cases) for b in plan),
        "blocks": [
            {
                "index": b.index,
                "name": b.name,
                "group": b.group,
                "skill_id": b.skill_id,
                "skill_path": b.skill_path,
                "split": b.split,
                "repeat": b.repeat,
                "cases": [list(c) for c in b.cases],
            }
            for b in plan
        ],
    }


def plan_bytes(plan: Sequence[Block]) -> bytes:
    return (json.dumps(plan_json(plan), sort_keys=True, separators=(",", ":")) + "\n").encode()


def plan_sha256(plan: Sequence[Block]) -> str:
    return hashlib.sha256(plan_bytes(plan)).hexdigest()


def load_plan(path: Path) -> tuple[Block, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        Block(
            b["index"],
            b["name"],
            b["group"],
            b["skill_id"],
            b["skill_path"],
            b["split"],
            b["repeat"],
            tuple((c[0], c[1], int(c[2])) for c in b["cases"]),
        )
        for b in data["blocks"]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    plan = build_plan(load_pool(args.root), args.root)
    data = plan_bytes(plan)
    digest = hashlib.sha256(data).hexdigest()
    if args.out.exists():
        if args.out.read_bytes() != data:
            print(f"error: {args.out} exists and differs from the plan built now", file=sys.stderr)
            return 1
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(data)
    sha_path = args.out.with_name(args.out.name + ".sha256")
    sha_path.write_text(f"{digest}  {args.out.name}\n")
    print(
        f"plan {args.out}: {len(plan)} blocks, {sum(len(b.cases) for b in plan)} episodes, "
        f"sha256 {digest}"
    )
    return 0


def _skill_path(name: str, group: str) -> str:
    folder = "candidates" if group == "proposer" else "controls"
    return f"skills/{folder}/{name}/SKILL.md"


if __name__ == "__main__":
    raise SystemExit(main())
