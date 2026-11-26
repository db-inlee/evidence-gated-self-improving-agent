"""Ticket 014 단계별 실행 계획(사전 등록 8절, 티켓 §11.1).

    uv run python -m evidencegate.t014.plans --stage N

| 단계 | 블록(재개 단위) | 블록 수 |
| 0 | S0 x t014_dev_a x 반복 0 x 칸(10 케이스) | 6 |
| 1 | S0 x t014_calibration x 반복 0~4 x 칸(150) | 30 |
| 2 | (풀 + S0) x t014_dev_b x 반복 0 x 칸(150) | (풀 + 1) x 6 |
| 4 | (Oracle 대상 + S0) x t014_oracle x 반복 0~2 x 칸(100) | (대상 + 1) x 18 |

- 블록 = (스킬, split, 반복, 칸). 같은 (스킬, split)의 칸 블록은 기록 파일 하나를 함께 쓴다
  (`bench_runner`는 블록의 케이스만 센다).
- 블록 순서는 정해진 순서(스킬 순서 x 반복 x 칸)를 `random.Random(PLAN_ORDER_SEED)`로 섞는다.
  중간에 멈춰도 특정 후보·칸에 몰리지 않게 한다.
- 단계 0·1 계획은 사전 등록과 함께 동결한다. 단계 2는 동결 풀(`POOL-014.json`)에서, 단계 4는
  단계 3 결과의 Oracle 대상 목록에서 같은 규칙으로 만든다. 통제 후보(`skills/controls`)는
  `Targets.controls`가 참일 때만 단계 4에 들어간다(사전 등록 13절 1).
- Oracle 대상이 0개이면 단계 4 계획을 만들지 않는다(사전 등록 8절, 단계 4는 실행하지 않는다).
- 계획 파일 `plans/ticket-014-stage<N>-plan.json`과 `.sha256`. 이미 있는 파일과 다르면 거부한다.
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
from evidencegate.eval_constants import S0_SKILL_ID
from evidencegate.eval_data import CELLS, Cell
from evidencegate.preflight import Check
from evidencegate.run_plan import S0_NAME, S0_PATH, Block, plan_bytes
from evidencegate.skills import load_skill
from evidencegate.t014.settings import (
    CALIBRATION,
    CALIBRATION_REPEATS,
    CONTROLS_DIR,
    DEV_A,
    DEV_B,
    EPISODE_STAGES,
    GATE_REPEAT,
    ORACLE,
    ORACLE_REPEATS,
    PLAN_ORDER_SEED,
    POOL_PATH,
    RESULT_NAMES,
    RESULTS_DIR,
)
from evidencegate.t014.splits import cell_cases

Key = tuple[str, str, str, int, Cell]  # (이름, 집합, split, 반복, 칸)


@dataclass(frozen=True)
class Targets:
    """단계 3에서 고정한 Oracle 대상(proposer 후보 이름)과 통제 후보 포함 여부."""

    proposer: tuple[str, ...]
    controls: bool


def plan_path(stage: int) -> Path:
    return Path(f"plans/ticket-014-stage{stage}-plan.json")


def _blocks(
    keys: Sequence[Key], skills: dict[str, tuple[str, str]], root: Path
) -> tuple[Block, ...]:
    for name, (skill_id, path) in skills.items():
        actual = load_skill(root / path).skill_id
        if actual != skill_id:
            raise ValueError(f"{name}: {path} has skill_id {actual}, expected {skill_id}")
    ordered = list(keys)
    random.Random(PLAN_ORDER_SEED).shuffle(ordered)
    return tuple(
        Block(i, name, group, *skills[name], split, repeat, tuple(cell_cases(split, cell)))
        for i, (name, group, split, repeat, cell) in enumerate(ordered)
    )


def _keys(entries: Sequence[tuple[str, str]], split: str, repeats: Sequence[int]) -> list[Key]:
    return [(n, g, split, r, cell) for n, g in entries for r in repeats for cell in CELLS]


S0_SKILLS = {S0_NAME: (S0_SKILL_ID, S0_PATH)}


def load_pool(root: Path) -> list[dict[str, str]]:
    path = root / POOL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; freeze the POOL-014 pool first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in data["candidates"] if c["group"] == "proposer"]


def _controls(root: Path) -> dict[str, tuple[str, str]]:
    paths = sorted((root / CONTROLS_DIR).glob("*/SKILL.md"))
    return {p.parent.name: (load_skill(p).skill_id, p.relative_to(root).as_posix()) for p in paths}


def build_stage4_plan(targets: Targets, root: Path) -> tuple[Block, ...]:
    pool = {c["name"]: (c["skill_id"], c["skill_path"]) for c in load_pool(root)}
    unknown = [n for n in targets.proposer if n not in pool]
    if unknown:
        raise ValueError(f"Oracle targets not in POOL-014: {unknown}")
    if not targets.proposer and not targets.controls:
        raise ValueError("no Oracle targets; stage 4 is skipped (preregistration 8)")
    names = sorted(targets.proposer)
    skills = {n: pool[n] for n in names} | S0_SKILLS
    entries = [(n, "proposer") for n in names]
    if targets.controls:
        controls = _controls(root)
        skills |= controls
        entries += [(n, "control") for n in controls]
    entries.append((S0_NAME, "s0"))
    return _blocks(_keys(entries, ORACLE[0], ORACLE_REPEATS), skills, root)


def load_targets(root: Path) -> Targets:
    path = root / RESULTS_DIR / RESULT_NAMES[3]
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; finalize stage 3 first")
    data = json.loads(path.read_text(encoding="utf-8"))["oracle_targets"]
    return Targets(proposer=tuple(data["proposer"]), controls=bool(data["controls"]))


def build_plan(stage: int, root: Path = REPO_ROOT) -> tuple[Block, ...]:
    if stage == 0:
        return _blocks(_keys([(S0_NAME, "s0")], DEV_A[0], (GATE_REPEAT,)), S0_SKILLS, root)
    if stage == 1:
        keys = _keys([(S0_NAME, "s0")], CALIBRATION[0], CALIBRATION_REPEATS)
        return _blocks(keys, S0_SKILLS, root)
    if stage == 2:
        pool = load_pool(root)
        skills = {c["name"]: (c["skill_id"], c["skill_path"]) for c in pool} | S0_SKILLS
        entries = [(c["name"], "proposer") for c in pool] + [(S0_NAME, "s0")]
        return _blocks(_keys(entries, DEV_B[0], (GATE_REPEAT,)), skills, root)
    if stage == 4:
        return build_stage4_plan(load_targets(root), root)
    raise ValueError(f"stage {stage} runs no episodes; expected one of {EPISODE_STAGES}")


def write_plan(root: Path, stage: int, plan: Sequence[Block]) -> str:
    out = root / plan_path(stage)
    data = plan_bytes(plan)
    if out.exists() and out.read_bytes() != data:
        raise FileExistsError(f"{out} exists and differs from the plan built now")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    return digest


def check_plan(root: Path, stage: int) -> Check:
    path = root / plan_path(stage)
    sha_path = path.with_name(path.name + ".sha256")
    if not path.exists() or not sha_path.exists():
        return Check("plan_014", False, f"missing {path} or {sha_path}")
    data = path.read_bytes()
    actual, recorded = hashlib.sha256(data).hexdigest(), sha_path.read_text().split()[0]
    if actual != recorded:
        return Check("plan_014", False, f"plan sha256 {actual} does not match recorded {recorded}")
    try:
        rebuilt = plan_bytes(build_plan(stage, root))
    except (FileNotFoundError, ValueError) as error:
        return Check("plan_014", False, f"cannot rebuild the stage {stage} plan: {error}")
    if data != rebuilt:
        return Check("plan_014", False, f"stage {stage} plan file differs from a rebuild")
    return Check("plan_014", True, f"stage {stage} plan sha256 {actual}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t014-plans")
    parser.add_argument("--stage", type=int, choices=EPISODE_STAGES, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.stage, args.root)
        digest = write_plan(args.root, args.stage, plan)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"plan {plan_path(args.stage)}: {len(plan)} blocks, "
        f"{sum(len(b.cases) for b in plan)} episodes, sha256 {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
