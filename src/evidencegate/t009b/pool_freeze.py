"""Ticket 009b 풀 동결과 묶음 파일 검사(사전 등록 2.1·6.6·8절, 티켓 §4의 4).

    uv run python -m evidencegate.t009b.pool_freeze freeze
    uv run python -m evidencegate.t009b.pool_freeze check-groups

동결(`freeze`):
1. 무결성: call_index가 0부터 빈칸 없음, proposer_config 하나, `run_options.json`이 고정 설정과
   같음,
   원장 비용 합이 비용 상한 이하. 어긋나면 종료 코드 1.
2. 사전 등록 멈춤(call 파일에서 계산, v1 `propose_cli`의 확인 순서): 폴더 전체 120회 / 풀 40 /
   마지막 풀 후보 뒤 25회. 하나도 아니면 생성이 끝나지 않은 것이므로 거부(종료 코드 1). 비용 상한·
   잔액 부족·형식 오류율·STOP으로 멈춘 경우다. 마지막 세션 끝 이유는 기록에만 쓴다.
3. 풀 수(proposer 풀 후보만, 통제 후보는 세지 않음)가 N_go(25) 미만이면 `POOL-009b.json`을 쓰지
   않고 `results/ticket-009b/pool-shortfall.json`에 미달을 기록한다(종료 코드 3).
4. 중복 표시(표시만, 풀에서 빼지 않음): v1 후보(`skills/candidates/POOL.json`)와 009 후보 80개
   (`runs/proposer-009/pool`, `plans/ticket-009b-dedupe-groups.json`의 call sha256으로 대조)에 대해
   W(`candidate_dedupe.is_duplicate_words`, t = 0.55). 통제 후보 표시는 호출 때 기록을 쓴다.
5. 응답 모델 요약(009의 `gpt-5.4-2026-03-05`와 같은지)은 기록만 한다(동결을 막지 않음, 위험 7).
6. 후보 SKILL.md(바이트 그대로)와 evidence.json을 `skills/candidates-009b/<candidate_id>/`에 쓰고,
   `POOL-009b.json`(009 `load_pool_009`가 읽는 `candidates` 형식, 통제 7개 포함)과 `.sha256`을
   쓴다. 이미 있는 파일과 바이트가 다르면 거부한다. 같은 입력이면 같은 바이트다(시각을 넣지 않음).

묶음 파일(`check-groups`, 6.6): `plans/ticket-009b-pool-groups.json`은 POOL 동결 뒤 Claude가 S0 대비
변경을 읽어 쓴다. 여기서는 형식만 검사한다. 모든 풀 후보가 정확히 한 묶음에 속하는지, 풀 후보가
아닌 call이 없는지, 메타데이터(작성 시점, 작성자, 기준, 묶음 수, 읽은 아이디어 수)가 있는지, 풀
sha256과 call sha256이 맞는지 본다. 대표는 묶음 안에서 가장 먼저 나온 풀 후보(가장 작은
call_index)다. 통과하면 `.sha256`을 기록한다.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.candidate_dedupe import is_duplicate_words
from evidencegate.skills import Skill, load_skill
from evidencegate.t009b.settings import (
    BUDGET_USD,
    CANDIDATES_DIR,
    DEDUPE_THRESHOLD,
    FIXED_RUN_OPTIONS,
    MAX_CALLS,
    N_GO,
    POOL_GROUPS_PATH,
    POOL_PATH,
    POOL_TARGET,
    PREREG_SHA_FILE,
    PREREG_TAG,
    RUN_DIR,
    RUN_OPTIONS_FILE,
    STOP_AFTER_IDLE,
)

SHORTFALL_PATH = Path("results/ticket-009b/pool-shortfall.json")
GROUPS_STAGE = "after the POOL-009b freeze, before any Dev-B or Oracle run"
RESPONSE_MODEL_009 = "gpt-5.4-2026-03-05"
V1_POOL = Path("skills/candidates/POOL.json")
T009_POOL_DIR = Path("runs/proposer-009/pool")
T009_GROUPS = Path("plans/ticket-009b-dedupe-groups.json")
S0_PATH = Path("skills/S0/SKILL.md")
CONTROLS_DIR = Path("skills/controls")
GROUPS_FIELDS = ("stage", "labeled_by", "method", "n_groups", "n_ideas_read", "pool_sha256")


class FreezeRefused(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_calls(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / RUN_DIR / "pool").glob("call-*.json"))
    calls = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return sorted(calls, key=lambda c: int(c["call_index"]))


def integrity_problems(root: Path, calls: Sequence[dict[str, Any]]) -> list[str]:
    problems = []
    if not calls:
        return [f"no calls in {root / RUN_DIR / 'pool'}"]
    indices = [int(c["call_index"]) for c in calls]
    if indices != list(range(len(calls))):
        missing = sorted(set(range(max(indices) + 1)) - set(indices))
        problems.append(f"call_index has gaps: missing {missing[:5]}")
    configs = {json.dumps(c["proposer_config"], sort_keys=True) for c in calls}
    if len(configs) != 1:
        problems.append(f"{len(configs)} different proposer_config values")
    options_path = root / RUN_DIR / "pool" / RUN_OPTIONS_FILE
    options = json.loads(options_path.read_text()) if options_path.exists() else None
    if options != FIXED_RUN_OPTIONS:
        problems.append(f"{RUN_OPTIONS_FILE} is {options}, not the 009b setting")
    cost = sum(float(r.get("cost_usd") or 0.0) for r in _jsonl(root / RUN_DIR / "ledger.jsonl"))
    if cost > BUDGET_USD:
        problems.append(f"ledger cost ${cost:.4f} exceeds ${BUDGET_USD}")
    return problems


def pool_count(calls: Sequence[dict[str, Any]]) -> int:
    """proposer 풀 후보 수(통제 후보는 출력 폴더에 없고 세지 않는다)."""
    return sum(1 for c in calls if c.get("in_pool"))


def idle_calls(calls: Sequence[dict[str, Any]]) -> int:
    idle = 0
    for c in calls:
        idle = 0 if c.get("in_pool") else idle + 1
    return idle


def stop_basis(calls: Sequence[dict[str, Any]]) -> str | None:
    """사전 등록 멈춤 셋 중 성립한 것(v1 `propose_cli`가 확인하는 순서). 없으면 None."""
    if len(calls) >= MAX_CALLS:
        return f"max total calls {MAX_CALLS} reached"
    if pool_count(calls) >= POOL_TARGET:
        return f"pool reached {POOL_TARGET}"
    if idle_calls(calls) >= STOP_AFTER_IDLE:
        return f"no new pool candidate in the last {STOP_AFTER_IDLE} calls"
    return None


def response_model_summary(root: Path) -> dict[str, Any]:
    counts = Counter(str(r.get("model")) for r in _jsonl(root / RUN_DIR / "response_models.jsonl"))
    return {
        "counts": dict(sorted(counts.items())),
        "same_as_009": set(counts) == {RESPONSE_MODEL_009},
        "model_009": RESPONSE_MODEL_009,
    }


def last_session(root: Path) -> dict[str, Any]:
    ends = [e for e in _jsonl(root / RUN_DIR / "sessions.jsonl") if e.get("event") == "session_end"]
    if not ends:
        return {"reason": None, "time": None}
    return {"reason": ends[-1].get("reason"), "time": ends[-1].get("time")}


def _summary(root: Path, calls: Sequence[dict[str, Any]], basis: str | None) -> dict[str, Any]:
    ledger = _jsonl(root / RUN_DIR / "ledger.jsonl")
    pool = pool_count(calls)
    return {
        "ticket": "009b",
        "source": RUN_DIR.as_posix(),
        "calls": len(calls),
        "pool_size": pool,
        "n_go": N_GO,
        "go": pool >= N_GO,
        "stop_basis": basis,
        "last_session_reason": last_session(root)["reason"],
        "last_session_end": last_session(root)["time"],
        "format_errors": sum(1 for c in calls if c.get("format_error_kind") is not None),
        "invalid": sum(1 for c in calls if c.get("invalid_reasons")),
        "proposer_duplicates": sum(1 for c in calls if c.get("duplicate_of")),
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in ledger), 6),
        "ledger_lines": dict(sorted(Counter(str(r.get("status")) for r in ledger).items())),
        "response_models": response_model_summary(root),
    }


def _v1_texts(root: Path) -> dict[str, str]:
    data = json.loads((root / V1_POOL).read_text(encoding="utf-8"))
    return {
        c["candidate_id"]: (root / "skills/candidates" / c["candidate_id"] / "SKILL.md").read_text(
            encoding="utf-8"
        )
        for c in data["proposer_candidates"]
    }


def _t009_texts(root: Path) -> dict[int, str]:
    hashes = json.loads((root / T009_GROUPS).read_text(encoding="utf-8"))["call_sha256"]
    texts = {}
    for key, expected in sorted(hashes.items(), key=lambda kv: int(kv[0])):
        path = root / T009_POOL_DIR / f"call-{int(key):03d}.json"
        data = path.read_bytes()
        if _sha256(data) != expected:
            raise FreezeRefused(f"{path} sha256 does not match {T009_GROUPS}")
        texts[int(key)] = str(json.loads(data)["candidate"]["skill_md"])
    return texts


def _dump(data: Any) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build_freeze(root: Path) -> tuple[dict[str, Any], dict[Path, bytes]]:
    """(요약, 쓸 파일들). 미달이면 파일은 미달 기록 하나다. 거부면 FreezeRefused."""
    calls = load_calls(root)
    problems = integrity_problems(root, calls)
    if problems:
        raise FreezeRefused("; ".join(problems))
    basis = stop_basis(calls)
    summary = _summary(root, calls, basis)
    if basis is None:
        raise FreezeRefused(
            f"no preregistered stop yet (calls {len(calls)}, pool {summary['pool_size']}, "
            f"idle {idle_calls(calls)}); last session ended with {summary['last_session_reason']!r}"
        )
    if not summary["go"]:
        return summary, {SHORTFALL_PATH: _dump({**summary, "result": "shortfall"})}

    s0 = (root / S0_PATH).read_text(encoding="utf-8")
    v1, t009 = _v1_texts(root), _t009_texts(root)
    files: dict[Path, bytes] = {}
    entries = []
    for c in calls:
        if not c.get("in_pool"):
            continue
        evidence = c["candidate"]["evidence"]
        text = str(c["candidate"]["skill_md"])
        folder = CANDIDATES_DIR / evidence["candidate_id"]
        files[folder / "SKILL.md"] = text.encode("utf-8")
        files[folder / "evidence.json"] = _dump(evidence)
        entries.append(
            {
                "name": evidence["candidate_id"],
                "group": "proposer",
                "skill_id": Skill.from_text(text).skill_id,
                "skill_path": (folder / "SKILL.md").as_posix(),
                "call_index": c["call_index"],
                "hypothesis": evidence.get("hypothesis"),
                "control_duplicates": c.get("control_duplicates") or [],
                "v1_duplicates": [
                    k for k, v in v1.items() if is_duplicate_words(text, v, s0, DEDUPE_THRESHOLD)
                ],
                "t009_duplicates": [
                    k for k, v in t009.items() if is_duplicate_words(text, v, s0, DEDUPE_THRESHOLD)
                ],
            }
        )
    for path in sorted((root / CONTROLS_DIR).glob("*/SKILL.md")):
        entries.append(
            {
                "name": path.parent.name,
                "group": "control",
                "skill_id": load_skill(path).skill_id,
                "skill_path": path.relative_to(root).as_posix(),
            }
        )
    sha_file = root / PREREG_SHA_FILE
    pool = {
        **summary,
        "preregistration": {
            "tag": PREREG_TAG,
            "sha256_file": sha_file.read_text().split("sha256: ")[1].split()[0]
            if sha_file.exists()
            else None,
        },
        "run_options": FIXED_RUN_OPTIONS,
        "proposer_config": calls[0]["proposer_config"],
        "duplicate_rule": {"name": "W (word change set)", "threshold": DEDUPE_THRESHOLD},
        "candidates": entries,
        "files": {p.as_posix(): _sha256(b) for p, b in sorted(files.items())},
    }
    files[POOL_PATH] = _dump(pool)
    return summary, files


def write_freeze(root: Path, files: dict[Path, bytes]) -> None:
    """이미 있는 파일은 바이트가 같아야 한다(하나라도 다르면 아무것도 쓰지 않는다)."""
    differs = [p for p, b in files.items() if (root / p).exists() and (root / p).read_bytes() != b]
    if differs:
        raise FreezeRefused(f"{differs[0]} exists and differs from the freeze built now")
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    if POOL_PATH in files:
        pool = root / POOL_PATH
        pool.with_name(pool.name + ".sha256").write_text(
            f"{_sha256(files[POOL_PATH])}  {pool.name}\n"
        )


@dataclass
class GroupsReport:
    problems: list[str] = field(default_factory=list)
    representatives: dict[str, int] = field(default_factory=dict)


def check_pool_groups(root: Path) -> GroupsReport:
    report = GroupsReport()
    path, pool_path = root / POOL_GROUPS_PATH, root / POOL_PATH
    if not path.exists() or not pool_path.exists():
        report.problems.append(f"missing {path} or {pool_path}")
        return report
    data = json.loads(path.read_text(encoding="utf-8"))
    pool_bytes = pool_path.read_bytes()
    for name in GROUPS_FIELDS:
        if name not in data:
            report.problems.append(f"metadata {name} is missing")
    if data.get("stage") not in (None, GROUPS_STAGE):
        report.problems.append(f"stage must be {GROUPS_STAGE!r}")
    groups: dict[str, Any] = data.get("groups") or {}
    if "n_groups" in data and data["n_groups"] != len(groups):
        report.problems.append(f"n_groups {data['n_groups']} != {len(groups)} groups")
    if "n_ideas_read" in data and not (
        isinstance(data["n_ideas_read"], int) and data["n_ideas_read"] > 0
    ):
        report.problems.append("n_ideas_read must be a positive integer")
    if "pool_sha256" in data and data["pool_sha256"] != _sha256(pool_bytes):
        report.problems.append("pool_sha256 does not match the current POOL-009b.json")
    pooled = {
        int(c["call_index"])
        for c in json.loads(pool_bytes)["candidates"]
        if c["group"] == "proposer"
    }
    seen = Counter(int(i) for g in groups.values() for i in g.get("calls", []))
    for i in sorted(pooled - set(seen)):
        report.problems.append(f"missing: call {i} is in no group")
    for i, n in sorted(seen.items()):
        if n > 1:
            report.problems.append(f"call {i} is in more than one group")
        if i not in pooled:
            report.problems.append(f"call {i} is not a pool candidate")
    hashes = data.get("call_sha256") or {}
    if set(hashes) != {str(i) for i in seen}:
        report.problems.append("call_sha256 keys must be the grouped calls")
    for key, expected in hashes.items():
        call = root / RUN_DIR / "pool" / f"call-{int(key):03d}.json"
        if not call.exists() or _sha256(call.read_bytes()) != expected:
            report.problems.append(f"call {key} sha256 does not match")
    report.representatives = {
        name: min(int(i) for i in g["calls"]) for name, g in groups.items() if g.get("calls")
    }
    return report


def groups_ready(root: Path) -> list[str]:
    """계획 동결 전 확인: 묶음 파일·`.sha256`이 있고 같으며, 검사를 통과한다."""
    path = root / POOL_GROUPS_PATH
    sha = path.with_name(path.name + ".sha256")
    if not path.exists() or not sha.exists():
        return [f"missing {path} or {sha}; write and check the pool groups first"]
    if sha.read_text().split()[0] != _sha256(path.read_bytes()):
        return [f"{path} changed after its check; run check-groups again"]
    return check_pool_groups(root).problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009b-pool-freeze")
    parser.add_argument("command", choices=("freeze", "check-groups"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root: Path = args.root
    if args.command == "check-groups":
        report = check_pool_groups(root)
        for problem in report.problems:
            print(f"error: {problem}", file=sys.stderr)
        if report.problems:
            return 1
        path = root / POOL_GROUPS_PATH
        digest = _sha256(path.read_bytes())
        path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
        print(f"groups ok: {len(report.representatives)} groups, sha256 {digest}")
        print(f"representatives (earliest pool candidate per group): {report.representatives}")
        return 0
    try:
        summary, files = build_freeze(root)
        write_freeze(root, files)
    except (FreezeRefused, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"calls {summary['calls']}, pool {summary['pool_size']} (N_go {N_GO}), "
        f"stop: {summary['stop_basis']}, response models same as 009: "
        f"{summary['response_models']['same_as_009']}"
    )
    if not summary["go"]:
        print(f"shortfall: pool {summary['pool_size']} < N_go {N_GO}; recorded {SHORTFALL_PATH}")
        return 3
    print(f"frozen {POOL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
