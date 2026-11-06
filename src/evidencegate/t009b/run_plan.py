"""Ticket 009b 본 실행 계획(사전 등록 1절).

009의 블록 규칙(`t009.run_plan.build_main_plan`)을 009b 풀로 쓴다.

    uv run python -m evidencegate.t009b.run_plan

- 풀: `skills/candidates-009b/POOL-009b.json`(동결 진입점이 만든다). 형식은 009와 같다:
  `{"candidates": [{"name", "group"(proposer/control), "skill_id", "skill_path"}, ...]}`.
- 블록: proposer x (t009_dev_b, t009_oracle) 반복 0, 통제 x t009_oracle 반복 0,
  S0 x t009_dev_b 반복 0, S0 x t009_oracle 반복 0~4.
  순서는 v1 `PLAN_ORDER_SEED`로 섞는다(009와 같음).
- 계획 파일 `plans/ticket-009b-main-plan.json`과 `.sha256`. 이미 있는 파일과 다르면 거부한다.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from evidencegate.analysis import REPO_ROOT
from evidencegate.preflight import Check
from evidencegate.run_plan import Block, plan_bytes
from evidencegate.t009.run_plan import PoolEntry, build_main_plan
from evidencegate.t009b.settings import POOL_PATH

PLAN_PATH = Path("plans/ticket-009b-main-plan.json")

__all__ = ["PLAN_PATH", "POOL_PATH", "build_main_plan_009b", "check_plan_009b", "main"]


def load_pool_009b(root: Path) -> list[PoolEntry]:
    path = root / POOL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; freeze the 009b pool first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        PoolEntry(c["name"], c["group"], c["skill_id"], c["skill_path"]) for c in data["candidates"]
    ]


def build_main_plan_009b(root: Path = REPO_ROOT) -> tuple[Block, ...]:
    return build_main_plan(load_pool_009b(root), root)


def check_plan_009b(plan_file: Path, code_root: Path) -> Check:
    sha_path = plan_file.with_name(plan_file.name + ".sha256")
    if not plan_file.exists() or not sha_path.exists():
        return Check("plan_009b", False, f"missing {plan_file} or {sha_path}")
    data = plan_file.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    recorded = sha_path.read_text().split()[0]
    if actual != recorded:
        return Check("plan_009b", False, f"plan sha256 {actual} does not match recorded {recorded}")
    try:
        rebuilt = plan_bytes(build_main_plan_009b(code_root))
    except (FileNotFoundError, ValueError) as error:
        return Check("plan_009b", False, f"cannot rebuild the 009b plan: {error}")
    if data != rebuilt:
        return Check("plan_009b", False, "009b plan file differs from a rebuild")
    return Check("plan_009b", True, f"009b main plan sha256 {actual}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009b-plan")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    out: Path = args.root / PLAN_PATH
    try:
        plan = build_main_plan_009b(args.root)
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
