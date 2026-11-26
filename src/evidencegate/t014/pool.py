"""Ticket 014 단계 0: Dev-A 내보내기, 풀 동결, 묶음 검사, 단계 0 결과(사전 등록 3·8절).

    uv run python -m evidencegate.t014.pool export
    uv run python -m evidencegate.t014.pool freeze
    uv run python -m evidencegate.t014.pool check-groups

009·009b 모듈은 경로를 고정해 두었으므로 고치지 않고, 경로와 무관한 함수만 가져다 쓴다.

- 내보내기(`export`): 014 Dev-A S0 반복 0 기록 60개에서 실패(CD 거짓 또는 ES 거짓)만, 009와 같은
  규칙(`_to_case`, 정렬, 섞기 seed)으로 `runs/proposer_inputs-014/…json`(+ `.sha256`)에 쓴다. 이미
  있는 파일과 바이트가 다르면 거부한다. 호출 0~119의 층 추출(`check_sampling`)이 안 되면 쓰지 않고
  종료 코드 1이다(사전 등록 3절: 호출하지 않고 멈춘다).
- 동결(`freeze`): 009b와 같은 무결성 확인(고정 설정은 009b 설정 + 제외 폴더)과 사전 등록 멈춤
  확인(120회 / 풀 40 / 25회 포화). 풀이 N_go(25) 미만이면 아무 파일도 쓰지 않고 종료 코드 3이다
  (단계 0 결과가 No-Go를 기록한다). 후보 SKILL.md와 evidence.json, `POOL-014.json`(+ `.sha256`)을
  쓴다. 009b 후보와 중복이라 호출 때 뺀 후보(`excluded_by`)의 목록을 `excluded_009b`로 남긴다.
- 묶음(`check-groups`): 009b와 같은 형식 검사(`plans/ticket-014-pool-groups.json`).
- 단계 0 결과(`finalize_stage0`, `t014.run finalize --stage 0`): Dev-A 계획 완료, 내보내기, 풀,
  묶음의 sha256, 비용, 응답 모델, 제외 목록, 지문을 `results/ticket-014/stage0-pool.json`에 한 번
  쓴다. 층 부족이나 풀 부족이면 `go: false`와 이유를 쓴다.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import S0_SKILL_ID
from evidencegate.proposer_inputs import EXPORT_SHUFFLE_SEED, ProposerCase, _to_case
from evidencegate.run_plan import load_plan
from evidencegate.runner import EpisodeRecord
from evidencegate.skills import Skill
from evidencegate.store import EpisodeStore
from evidencegate.t009.proposer_inputs import check_sampling, strata
from evidencegate.t009b.pool_freeze import idle_calls, pool_count, stop_basis
from evidencegate.t014 import stages
from evidencegate.t014.plans import plan_path
from evidencegate.t014.settings import (
    BUDGET_USD,
    CANDIDATES_DIR,
    DEDUPE_THRESHOLD,
    DEV_A,
    EXPORT_PATH,
    GATE_REPEAT,
    MAX_CALLS,
    N_GO,
    POOL_GROUPS_PATH,
    POOL_PATH,
    RUN_DIR,
    RUN_OPTIONS_FILE,
    fixed_run_options,
)
from evidencegate.t014.splits import split_cases

GROUPS_STAGE = "after the POOL-014 freeze, before any Dev-B run"
GROUPS_FIELDS = ("stage", "labeled_by", "method", "n_groups", "n_ideas_read", "pool_sha256")
RESPONSE_MODEL_009 = "gpt-5.4-2026-03-05"


class FreezeRefused(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _dump(data: Any) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# 내보내기 -----------------------------------------------------------------------------


def dev_a_records(runs_dir: Path) -> list[EpisodeRecord]:
    path = stages.store_path(runs_dir, DEV_A[0], S0_SKILL_ID)
    return EpisodeStore(path, repair=False).records() if path.exists() else []


def build_export(records: Sequence[EpisodeRecord]) -> list[ProposerCase]:
    allowed = set(split_cases(DEV_A[0]))
    runs = {(r.key.policy_id, r.key.config_id, r.key.tool_spec_id, r.key.skill_id) for r in records}
    if len(runs) > 1:
        raise ValueError(f"records from mixed runs: {sorted(runs)}")
    for r in records:
        if (r.key.family, r.key.level, r.key.incident_seed) not in allowed:
            raise ValueError(f"record is not a Ticket 014 Dev-A case: {r.key}")
        if r.key.repeat != GATE_REPEAT:
            raise ValueError(f"Dev-A export uses repeat 0 only, got {r.key}")
    counts = Counter((r.key.family, r.key.level, r.key.incident_seed) for r in records)
    if any(n > 1 for n in counts.values()):
        raise ValueError("duplicate Dev-A records")
    if len(counts) != len(allowed):
        raise ValueError(f"{len(allowed) - len(counts)} missing Dev-A records")
    failures = [r for r in records if not (r.correct_disposition and r.evidence_safe)]
    ordered = sorted(failures, key=lambda r: (r.key.family, r.key.incident_seed, r.key.level))
    cases = [_to_case(r) for r in ordered]
    random.Random(EXPORT_SHUFFLE_SEED).shuffle(cases)
    return cases


def export_bytes(cases: Sequence[ProposerCase]) -> bytes:
    text = json.dumps([c.to_json() for c in cases], indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")  # v1 `save_proposer_cases`와 같은 바이트


def export_main(runs_dir: Path, out: Path) -> int:
    try:
        cases = build_export(dev_a_records(runs_dir))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    counts, sampling = strata(cases), check_sampling(cases, calls=MAX_CALLS)
    print(
        f"strata: failures {counts['failures']}, evidence-only {counts['evidence_only']}, "
        f"wrong {counts['wrong']}; sampling: {sampling.detail}"
    )
    if not sampling.ok:
        print("error: strata too thin for the proposer; do not call it", file=sys.stderr)
        return 1
    data = export_bytes(cases)
    if out.exists() and out.read_bytes() != data:
        print(f"error: {out} exists and differs from the export built now", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    out.with_name(out.name + ".sha256").write_text(f"{_sha256(data)}  {out.name}\n")
    print(f"export {out}: {len(cases)} cases, sha256 {_sha256(data)}")
    return 0


# 풀 동결 ------------------------------------------------------------------------------


def load_calls(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / RUN_DIR / "pool").glob("call-*.json"))
    calls = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return sorted(calls, key=lambda c: int(c["call_index"]))


def integrity_problems(root: Path, calls: Sequence[dict[str, Any]]) -> list[str]:
    if not calls:
        return [f"no calls in {root / RUN_DIR / 'pool'}"]
    problems = []
    indices = [int(c["call_index"]) for c in calls]
    if indices != list(range(len(calls))):
        problems.append(
            f"call_index has gaps: {sorted(set(range(max(indices) + 1)) - set(indices))[:5]}"
        )
    if len({json.dumps(c["proposer_config"], sort_keys=True) for c in calls}) != 1:
        problems.append("more than one proposer_config")
    options_path = root / RUN_DIR / "pool" / RUN_OPTIONS_FILE
    options = json.loads(options_path.read_text()) if options_path.exists() else None
    if options != fixed_run_options(root):
        problems.append(f"{RUN_OPTIONS_FILE} is {options}, not the Ticket 014 setting")
    cost = _cost(root)
    if cost > BUDGET_USD:
        problems.append(f"ledger cost ${cost:.4f} exceeds ${BUDGET_USD}")
    return problems


def _cost(root: Path) -> float:
    return sum(float(r.get("cost_usd") or 0.0) for r in _jsonl(root / RUN_DIR / "ledger.jsonl"))


def _last_session(root: Path) -> str | None:
    ends = [e for e in _jsonl(root / RUN_DIR / "sessions.jsonl") if e.get("event") == "session_end"]
    return ends[-1].get("reason") if ends else None


def summary(root: Path, calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    models = Counter(str(r.get("model")) for r in _jsonl(root / RUN_DIR / "response_models.jsonl"))
    ledger = _jsonl(root / RUN_DIR / "ledger.jsonl")
    size = pool_count(calls)
    return {
        "calls": len(calls),
        "pool_size": size,
        "n_go": N_GO,
        "go": size >= N_GO,
        "stop_basis": stop_basis(calls) if calls else None,
        "idle_calls": idle_calls(calls),
        "last_session_reason": _last_session(root),
        "format_errors": sum(1 for c in calls if c.get("format_error_kind") is not None),
        "invalid": sum(1 for c in calls if c.get("invalid_reasons")),
        "proposer_duplicates": sum(1 for c in calls if c.get("duplicate_of")),
        "excluded_009b": [
            {
                "call_index": c["call_index"],
                "candidate_id": c["candidate"]["evidence"]["candidate_id"],
                "of": c["excluded_by"],
            }
            for c in calls
            if c.get("excluded_by")
        ],
        "cost_usd": round(_cost(root), 6),
        "ledger_lines": dict(sorted(Counter(str(r.get("status")) for r in ledger).items())),
        "response_models": {
            "counts": dict(sorted(models.items())),
            "same_as_009": set(models) == {RESPONSE_MODEL_009},
        },
    }


def build_freeze(root: Path) -> tuple[dict[str, Any], dict[Path, bytes]]:
    """(요약, 쓸 파일들). 풀이 N_go 미만이면 파일은 없다. 거부면 FreezeRefused."""
    calls = load_calls(root)
    problems = integrity_problems(root, calls)
    if problems:
        raise FreezeRefused("; ".join(problems))
    info = summary(root, calls)
    if info["stop_basis"] is None:
        raise FreezeRefused(
            f"no preregistered stop yet (calls {len(calls)}, pool {info['pool_size']}, idle "
            f"{info['idle_calls']}); last session ended with {info['last_session_reason']!r}"
        )
    if not info["go"]:
        return info, {}
    files: dict[Path, bytes] = {}
    entries = []
    for c in calls:
        if not c.get("in_pool"):
            continue
        evidence, text = c["candidate"]["evidence"], str(c["candidate"]["skill_md"])
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
            }
        )
    body = {
        **info,
        "ticket": "014",
        "run_options": fixed_run_options(Path("<root>")),
        "proposer_config": calls[0]["proposer_config"],
        "duplicate_rule": {"name": "W (word change set)", "threshold": DEDUPE_THRESHOLD},
        "candidates": entries,
        "files": {p.as_posix(): _sha256(b) for p, b in sorted(files.items())},
    }
    files[POOL_PATH] = _dump(body)
    return info, files


def write_freeze(root: Path, files: dict[Path, bytes]) -> None:
    differs = [p for p, b in files.items() if (root / p).exists() and (root / p).read_bytes() != b]
    if differs:
        raise FreezeRefused(f"{differs[0]} exists and differs from the freeze built now")
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    pool = root / POOL_PATH
    pool.with_name(pool.name + ".sha256").write_text(f"{_sha256(files[POOL_PATH])}  {pool.name}\n")


def freeze_main(root: Path) -> int:
    try:
        info, files = build_freeze(root)
        if files:
            write_freeze(root, files)
    except (FreezeRefused, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"calls {info['calls']}, pool {info['pool_size']} (N_go {N_GO}), stop: "
        f"{info['stop_basis']}, excluded as 009b duplicates {len(info['excluded_009b'])}"
    )
    if not info["go"]:
        print(f"No-Go: pool {info['pool_size']} < N_go {N_GO}; finalize stage 0 to record it")
        return 3
    print(f"frozen {POOL_PATH}")
    return 0


# 묶음 ---------------------------------------------------------------------------------


def check_pool_groups(root: Path) -> list[str]:
    path, pool_path = root / POOL_GROUPS_PATH, root / POOL_PATH
    if not path.exists() or not pool_path.exists():
        return [f"missing {path} or {pool_path}"]
    data = json.loads(path.read_text(encoding="utf-8"))
    pool_bytes = pool_path.read_bytes()
    problems = [f"metadata {name} is missing" for name in GROUPS_FIELDS if name not in data]
    if data.get("stage") not in (None, GROUPS_STAGE):
        problems.append(f"stage must be {GROUPS_STAGE!r}")
    groups: dict[str, Any] = data.get("groups") or {}
    if "n_groups" in data and data["n_groups"] != len(groups):
        problems.append(f"n_groups {data['n_groups']} != {len(groups)} groups")
    if "n_ideas_read" in data and not (
        isinstance(data["n_ideas_read"], int) and data["n_ideas_read"] > 0
    ):
        problems.append("n_ideas_read must be a positive integer")
    if "pool_sha256" in data and data["pool_sha256"] != _sha256(pool_bytes):
        problems.append("pool_sha256 does not match the current POOL-014.json")
    pooled = {int(c["call_index"]) for c in json.loads(pool_bytes)["candidates"]}
    seen = Counter(int(i) for g in groups.values() for i in g.get("calls", []))
    problems += [f"missing: call {i} is in no group" for i in sorted(pooled - set(seen))]
    for i, n in sorted(seen.items()):
        if n > 1:
            problems.append(f"call {i} is in more than one group")
        if i not in pooled:
            problems.append(f"call {i} is not a pool candidate")
    hashes = data.get("call_sha256") or {}
    if set(hashes) != {str(i) for i in seen}:
        problems.append("call_sha256 keys must be the grouped calls")
    for key, expected in hashes.items():
        call = root / RUN_DIR / "pool" / f"call-{int(key):03d}.json"
        if not call.exists() or _sha256(call.read_bytes()) != expected:
            problems.append(f"call {key} sha256 does not match")
    return problems


def groups_ready(root: Path) -> list[str]:
    path = root / POOL_GROUPS_PATH
    sha = path.with_name(path.name + ".sha256")
    if not path.exists() or not sha.exists():
        return [f"missing {path} or {sha}; write and check the pool groups first"]
    if sha.read_text().split()[0] != _sha256(path.read_bytes()):
        return [f"{path} changed after its check; run check-groups again"]
    return check_pool_groups(root)


def check_groups_main(root: Path) -> int:
    problems = check_pool_groups(root)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    path = root / POOL_GROUPS_PATH
    digest = _sha256(path.read_bytes())
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    print(f"groups ok, sha256 {digest}")
    return 0


# 단계 0 결과 --------------------------------------------------------------------------


def finalize_stage0(root: Path, runs_dir: Path, export: Path | None = None) -> dict[str, Any]:
    export = export or root / EXPORT_PATH
    plan_file = root / plan_path(0)
    plan = load_plan(plan_file)
    stages.collect_records(runs_dir, plan)
    cases = build_export(dev_a_records(runs_dir))
    sampling = check_sampling(cases, calls=MAX_CALLS)
    payload: dict[str, Any] = {
        "dev_a": {
            "plan_sha256": _sha256(plan_file.read_bytes()),
            "episodes": sum(len(b.cases) for b in plan),
            "stores": stages.store_hashes(root, runs_dir, plan),
        },
        "export": {"strata": strata(cases), "sampling": sampling.detail},
        "fingerprint": json.loads(stages.fingerprint_path(runs_dir, 0).read_text())
        if stages.fingerprint_path(runs_dir, 0).exists()
        else None,
    }
    if not sampling.ok:
        payload |= {"go": False, "no_go_reason": f"strata: {sampling.detail}"}
    else:
        data = export.read_bytes()
        if data != export_bytes(cases):
            raise ValueError(f"{export} differs from the export rebuilt from the Dev-A records")
        payload["export"]["sha256"] = _sha256(data)
        info, files = build_freeze(root)
        payload |= info
        if not info["go"]:
            payload["no_go_reason"] = f"pool {info['pool_size']} < N_go {N_GO}"
        else:
            pool_file = root / POOL_PATH
            if pool_file.read_bytes() != files[POOL_PATH]:
                raise ValueError(f"{pool_file} differs from the freeze built now")
            problems = groups_ready(root)
            if problems:
                raise ValueError("; ".join(problems))
            payload["pool_sha256"] = _sha256(pool_file.read_bytes())
            groups = root / POOL_GROUPS_PATH
            payload["groups_sha256"] = _sha256(groups.read_bytes())
    stages.write_result(root, 0, payload)
    result: dict[str, Any] = json.loads(stages.result_path(root, 0).read_text(encoding="utf-8"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t014-pool")
    parser.add_argument("command", choices=("export", "freeze", "check-groups"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root: Path = args.root
    if args.command == "export":
        return export_main(args.runs_dir or root / "runs", root / EXPORT_PATH)
    if args.command == "freeze":
        return freeze_main(root)
    return check_groups_main(root)


if __name__ == "__main__":
    raise SystemExit(main())
