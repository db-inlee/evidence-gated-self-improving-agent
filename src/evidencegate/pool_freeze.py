"""후보 풀 동결 (Ticket 004 §6).

    uv run python -m evidencegate.pool_freeze --pool-dir runs/proposer/pool

`runs/proposer/pool`의 결과 중 풀에 남은 후보(`in_pool`)를 `skills/candidates/<candidate_id>/`에
`SKILL.md`(바이트 그대로)와 `evidence.json`으로 쓰고, `skills/candidates/POOL.json`을 만든다.
이미 `POOL.json`이 있으면 덮어쓰지 않는다. 동결은 기계적이다: 후보를 고르거나 빼지 않는다.

`POOL.json`의 `files`는 `skills/candidates/`(POOL.json 제외)와 `skills/controls/` 아래 모든 파일의
sha256이다. 동결 테스트(`tests/test_pool_frozen.py`)가 `verify_frozen_pool`로 이를 다시 계산한다.

`topic_of`는 동결 보고서의 주제 분포(기술용, 게이트 판정과 하위집합 라벨에 쓰지 않음)를 낸다.
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evidencegate.candidate_checks import normalize_lines
from evidencegate.skills import load_skill

PREREGISTRATION_COMMIT = "aa1c809"
POOL_TARGET = 12
SUBSETS = {"pos": "positive", "neg": "negative", "mixed": "mixed", "neutral": "neutral"}
# 주제 분류 키워드(티켓 §6, 결과를 보기 전에 고정). 단어 단위로 맞춘다.
TRAFFIC_KEYWORDS = ("traffic", "load", "request rate", "rps")
TIMING_KEYWORDS = ("time", "timing", "shortly", "minute", "minutes", "started", "happened")


def freeze_pool(root: Path, pool_dir: Path, ledger_path: Path, *, frozen_at: str) -> dict[str, Any]:
    candidates_dir = root / "skills" / "candidates"
    pool_path = candidates_dir / "POOL.json"
    if pool_path.exists():
        raise FileExistsError(f"{pool_path} already exists; the pool is frozen")
    records = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(pool_dir.glob("call-*.json"))
    ]
    records.sort(key=lambda r: int(r["call_index"]))
    configs = {json.dumps(r["proposer_config"], sort_keys=True) for r in records}
    if len(configs) != 1:
        raise ValueError(f"{pool_dir} holds results with {len(configs)} proposer_config values")

    pooled = [r for r in records if r["in_pool"]]
    for r in pooled:
        evidence = r["candidate"]["evidence"]
        folder = candidates_dir / evidence["candidate_id"]
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "SKILL.md").write_bytes(r["candidate"]["skill_md"].encode("utf-8"))
        (folder / "evidence.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    controls_dir = root / "skills" / "controls"
    pool: dict[str, Any] = {
        "frozen_at": frozen_at,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "source": str(pool_dir),
        "proposer_config": records[0]["proposer_config"],
        "target": POOL_TARGET,
        "pool_size": len(pooled),
        "shortfall": len(pooled) < POOL_TARGET,
        "calls": len(records),
        "format_errors": sum(1 for r in records if r["format_error_kind"] is not None),
        "pool_cost_usd": sum(float(r["cost_usd"]) for r in records),
        "ledger_cumulative_usd": _ledger_total(ledger_path),
        "proposer_candidates": [
            {
                "candidate_id": r["candidate"]["evidence"]["candidate_id"],
                "skill_id": load_skill(
                    candidates_dir / r["candidate"]["evidence"]["candidate_id"] / "SKILL.md"
                ).skill_id,
                "call_index": r["call_index"],
                "control_duplicates": r["control_duplicates"],
            }
            for r in pooled
        ],
        "controls": [
            {
                "name": path.parent.name,
                "skill_id": load_skill(path).skill_id,
                "subset": SUBSETS[path.parent.name.split("-")[0]],
            }
            for path in sorted(controls_dir.glob("*/SKILL.md"))
        ],
        "excluded": [_excluded(r) for r in records if not r["in_pool"]],
        "files": _file_hashes(root),
    }
    pool_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pool


def verify_frozen_pool(root: Path) -> list[str]:
    """POOL.json의 파일 해시와 지금 파일을 비교한다. 다르면 문제 목록을 돌려준다."""
    pool = json.loads((root / "skills" / "candidates" / "POOL.json").read_text(encoding="utf-8"))
    recorded: dict[str, str] = pool["files"]
    current = _file_hashes(root)
    problems = [f"missing: {p}" for p in sorted(set(recorded) - set(current))]
    problems += [f"unexpected: {p}" for p in sorted(set(current) - set(recorded))]
    problems += [
        f"changed: {p}" for p in sorted(set(recorded) & set(current)) if recorded[p] != current[p]
    ]
    return problems


def topic_of(text: str, s0_text: str) -> str:
    """S0 대비 바뀐 줄(정규화)로 'traffic' / 'timing' / 'both' / 'other'를 정한다.

    바뀐 줄은 추가·변경된 줄이다. 삭제만 한 후보는 지운 줄을 본다(중복 제거의 변경분과 같은 규칙).
    """
    base, candidate = normalize_lines(s0_text), normalize_lines(text)
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, base, candidate, autojunk=False
    ).get_opcodes():
        if tag in ("replace", "insert"):
            added += candidate[j1:j2]
        if tag in ("replace", "delete"):
            removed += base[i1:i2]
    changed = " ".join(added or removed)
    traffic = _has_any(changed, TRAFFIC_KEYWORDS)
    timing = _has_any(changed, TIMING_KEYWORDS)
    if traffic and timing:
        return "both"
    if traffic:
        return "traffic"
    if timing:
        return "timing"
    return "other"


def _has_any(text: str, keywords: Sequence[str]) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def _excluded(r: dict[str, Any]) -> dict[str, Any]:
    candidate_id = r["candidate"]["evidence"]["candidate_id"] if r["candidate"] else None
    if r["format_error_kind"] is not None:
        reason, detail = "format_error", r["format_error_kind"]
    elif r["invalid_reasons"]:
        reason, detail = "invalid", ", ".join(r["invalid_reasons"])
    else:
        reason, detail = "duplicate", r["duplicate_of"]
    return {
        "call_index": r["call_index"],
        "candidate_id": candidate_id,
        "reason": reason,
        "detail": detail,
    }


def _file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for folder in (root / "skills" / "candidates", root / "skills" / "controls"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "POOL.json" and path.name != ".DS_Store":
                rel = path.relative_to(root).as_posix()
                hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _ledger_total(path: Path) -> float:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(float(json.loads(line).get("cost_usd") or 0.0) for line in lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-freeze")
    parser.add_argument("--pool-dir", type=Path, default=Path("runs/proposer/pool"))
    parser.add_argument("--ledger", type=Path, default=Path("runs/proposer/ledger.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        pool = freeze_pool(args.root, args.pool_dir, args.ledger, frozen_at=frozen_at)
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"frozen at {pool['frozen_at']}: pool {pool['pool_size']}/{pool['target']}, "
        f"calls {pool['calls']}, excluded {len(pool['excluded'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
