"""실행 요약 009b (Ticket 011 §6의 3). 정의: `writeup/inputs/run_summary_009b_definition.txt`.

    uv run python -m evidencegate.run_summary_009b --runs-dir runs \\
        --out writeup/inputs/run_summary_009b.json

- 정의 파일(sha256 `DEFINITION_SHA256`)의 네 절을 그대로 따른다.
  A 009 풀(커밋된 manifest), B 009b 본 실행 세션·기록·정지 구간, C 009b proposer(동결된
  POOL-009b.json), D 같은 seed 처분 일치(v1 보정, 순차 대 동시 2).
- v1 `run_summary`의 `record_summary`·`idle_gaps`·`session_summary`를 import해서 그대로 쓴다(v1
  동작 불변). 정지 구간 목록(`idle_spans`)만 새로 만들고, 같은 구간 규칙임을 테스트가 v1 요약과
  대조한다.
- 입력을 **읽기만** 한다. 고정 입력 sha256(`PINS`)이나 정의 파일 sha256이 다르면 아무것도 쓰지 않고
  거부한다. proposer 원장은 읽지 않는다.
- 출력: `{"run": …, "provenance": …, "notes": […]}`. 정의에서 kind가 run인 값은 `run`에,
  provenance인 값(시각·문구·경로·sha256·태그)은 `provenance`에 둔다. CD·ES·조치 종류는 담지
  않는다. 정렬된 키, 생성 시각 없음. 같은 입력이면 같은 바이트다.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import (
    CONFIG_ID,
    IDLE_GAP_S,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
    TOOL_SPEC_ID,
)
from evidencegate.ids import content_id
from evidencegate.run_plan import Block, load_plan
from evidencegate.run_summary import (
    _iso,
    _r,
    _rel,
    _t,
    idle_gaps,
    record_summary,
    session_summary,
    summary_bytes,
)
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009 import propose as t009_propose
from evidencegate.t009.splits import CALIBRATION_V1
from evidencegate.t009b import settings as t009b_settings

DEFINITION_PATH = Path("writeup/inputs/run_summary_009b_definition.txt")
DEFINITION_SHA256 = "8b3462ab36855446c81fc6704bcdaf94ecd2eebafc6dc2269d7384bf5f6e18f5"
DEFAULT_OUT = Path("writeup/inputs/run_summary_009b.json")

# 저장소 기준
MANIFEST_009 = Path("results/ticket-009/proposer-calls-manifest.json")
PLAN_009B = Path("plans/ticket-009b-main-plan.json")
POOL_009B = t009b_settings.POOL_PATH
# runs 폴더 기준
SESSIONS_009B = Path("ticket009b/main/sessions.jsonl")
SEQ_STORE = Path("calibration/ollama__gpt-oss__20b__24bf38dea859/e9b4996f7529/bf65e569ae37.jsonl")
PAR_STORE = Path(
    f"calibration/{POLICY_ID.replace(':', '__')}__{CONFIG_ID}/{TOOL_SPEC_ID}/{S0_SKILL_ID}.jsonl"
)

# 정의 파일에 적은 고정 sha256(이름 -> (경로, sha256))
PINS: dict[str, tuple[Path, str]] = {
    "t009_manifest": (
        MANIFEST_009,
        "b2e5e759e613e66ec427f7bf56e526644fd3acab9859e331cc4cba749620da28",
    ),
    "t009b_plan": (PLAN_009B, "4aaab082e8aa773a84c6978e4ada41ae1cf4ade69379252be05d4f7e5e9737c7"),
    "t009b_sessions": (
        SESSIONS_009B,
        "d34bd28e05d6714695c631961f4ad45d098e6eeb74fe9715545eecc6b17adfb5",
    ),
    "t009b_pool": (POOL_009B, "6f4a9a90f619dd8ef51f394d988d06eeb2a9c4536fb0056e2263903ba4aef239"),
    "calibration_sequential": (
        SEQ_STORE,
        "150336a014bad3fc487da66ff15ae89b2acce6ef949b3379722ca01d285e37c9",
    ),
    "calibration_concurrent": (
        PAR_STORE,
        "1041e046ca3acf62839cf2a21db99d885bd9b962692dd510c0814bd4f50d8158",
    ),
}
DEFAULT_PINS: dict[str, str] = {name: digest for name, (_, digest) in PINS.items()} | {
    "definition": DEFINITION_SHA256
}

SAME_SEED_CASES = tuple(CALIBRATION_V1)
SAME_SEED_REPEATS = S0_ORACLE_REPEATS
WHERE = ("in_session", "between_sessions")
NOTES = (
    "Run logs, not part of any analysis; definition in "
    "writeup/inputs/run_summary_009b_definition.txt (values were read or counted as it fixes).",
    f"An idle span is a stretch of at least {IDLE_GAP_S:g} s with no episode running; no cause is "
    "attached to a span.",
    "Same-seed agreement uses the v1 calibration runs (sequential against two concurrent "
    "episodes), not the Ticket 009 calibration.",
    "The proposer ledgers are not read.",
)

Section = tuple[dict[str, Any], dict[str, Any]]  # (run, provenance)


# ---------------------------------------------------------------------------
# A. 009 풀
# ---------------------------------------------------------------------------


def pool_009(manifest: Mapping[str, Any]) -> Section:
    summary = manifest["summary"]
    target = t009_propose.POOL_TARGET
    stated = re.search(r"target (\d+)", manifest["status"])
    if stated is None or int(stated.group(1)) != target:
        raise ValueError(f"manifest status does not state the design target {target}")
    run = {
        "calls": summary["calls"],
        "size": summary["pool"],
        "target": target,
        "duplicates": summary["proposer_duplicates"],
        "cost_usd": summary["cost_usd"],
        "server_errors": summary["ledger_lines"]["server_error"],
    }
    prov = {"status": manifest["status"], "tags": list(manifest["preregistration"]["tags"])}
    return run, prov


# ---------------------------------------------------------------------------
# C. 009b proposer
# ---------------------------------------------------------------------------


def proposer_009b(pool: Mapping[str, Any]) -> Section:
    if pool["n_go"] != t009b_settings.N_GO:
        raise ValueError(f"pool n_go {pool['n_go']} is not settings.N_GO {t009b_settings.N_GO}")
    models = list(pool["response_models"]["counts"])
    if len(models) != 1:
        raise ValueError(f"expected one response model, got {models}")
    run = {
        "calls": pool["calls"],
        "pool_size": pool["pool_size"],
        "target": t009b_settings.POOL_TARGET,
        "n_go": pool["n_go"],
        "go": pool["go"],
        "duplicates": pool["proposer_duplicates"],
        "format_errors": pool["format_errors"],
        "invalid": pool["invalid"],
        "cost_usd": pool["cost_usd"],
        "budget_usd": pool["proposer_config"]["budget_usd"],
    }
    prov = {
        "stop_basis": pool["stop_basis"],
        "last_session_reason": pool["last_session_reason"],
        "last_session_end": pool["last_session_end"],
        "response_model": models[0],
    }
    return run, prov


# ---------------------------------------------------------------------------
# D. 같은 seed 처분 일치
# ---------------------------------------------------------------------------

PairKey = tuple[str, str, int, int]


def _pair_key(r: EpisodeRecord) -> PairKey:
    return (r.key.family, r.key.level, r.key.incident_seed, r.key.repeat)


def _in_case_set(
    records: Sequence[EpisodeRecord], wanted: set[PairKey], side: str
) -> tuple[dict[PairKey, EpisodeRecord], int, dict[str, Any]]:
    kept = {_pair_key(r): r for r in records if _pair_key(r) in wanted}
    missing = sorted(wanted - set(kept))
    if missing:
        raise ValueError(f"{side}: {len(missing)} pairs missing, first {missing[0]}")
    configs = {content_id(r.policy_config) for r in kept.values()}
    if len(configs) != 1:
        raise ValueError(f"{side}: {len(configs)} different policy configs in the case set")
    ignored = sum(1 for r in records if _pair_key(r) not in wanted)
    return kept, ignored, next(iter(kept.values())).policy_config


def same_seed(
    sequential: Sequence[EpisodeRecord],
    concurrent: Sequence[EpisodeRecord],
    *,
    cases: Sequence[tuple[str, str, int]] = SAME_SEED_CASES,
    repeats: Sequence[int] = SAME_SEED_REPEATS,
) -> Section:
    wanted = {(f, lv, s, r) for f, lv, s in cases for r in repeats}
    seq, seq_ignored, seq_config = _in_case_set(sequential, wanted, "sequential")
    par, par_ignored, par_config = _in_case_set(concurrent, wanted, "concurrent_2")
    differ = {
        k for k in set(seq_config) | set(par_config) if seq_config.get(k) != par_config.get(k)
    }
    if differ != {"parallel"}:
        raise ValueError(f"policy configs must differ only in 'parallel', they differ in {differ}")
    for key in sorted(wanted):
        if seq[key].llm_seed != par[key].llm_seed:
            raise ValueError(f"{key}: llm_seed {seq[key].llm_seed} != {par[key].llm_seed}")
    run = {
        "pairs": len(wanted),
        "same_disposition": sum(seq[k].action == par[k].action for k in wanted),
        "same_tool_sequence": sum(
            [t.tool for t in seq[k].trace.turns] == [t.tool for t in par[k].trace.turns]
            for k in wanted
        ),
        "ignored": {"sequential": seq_ignored, "concurrent_2": par_ignored},
    }
    prov = {
        "configs": {
            "sequential": content_id(seq_config),
            "concurrent_2": content_id(par_config),
        }
    }
    return run, prov


# ---------------------------------------------------------------------------
# B. 009b 본 실행
# ---------------------------------------------------------------------------


def _merged(records: Sequence[EpisodeRecord]) -> list[list[datetime]]:
    """v1 `idle_gaps`와 같은 규칙: 에피소드 구간을 겹치면 합친다."""
    spans = sorted(
        (_t(r.started_at), _t(r.started_at) + timedelta(seconds=r.duration_s)) for r in records
    )
    merged: list[list[datetime]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def idle_spans(
    records: Sequence[EpisodeRecord],
    sessions: Sequence[Mapping[str, str]],
    min_gap_s: float = IDLE_GAP_S,
) -> dict[str, Any]:
    """정지 구간마다 시작·끝·길이와 세션 안/사이. 세션 경계를 걸치면 between_sessions."""
    bounds = [(_t(s["start"]), _t(s["end"])) for s in sessions]
    spans: list[dict[str, Any]] = []
    for a, b in pairwise(_merged(records)):
        start, end = a[1], b[0]
        length = (end - start).total_seconds()
        if length < min_gap_s:
            continue
        inside = any(s <= start and end <= e for s, e in bounds)
        spans.append(
            {
                "start": _iso(start),
                "end": _iso(end),
                "length_s": _r(length),
                "where": "in_session" if inside else "between_sessions",
            }
        )
    by_where = {
        w: {
            "count": sum(1 for s in spans if s["where"] == w),
            "total_s": _r(sum(s["length_s"] for s in spans if s["where"] == w)),
        }
        for w in WHERE
    }
    return {"min_gap_s": min_gap_s, "spans": spans, "by_where": by_where}


def _store_path(runs_dir: Path, block: Block) -> Path:
    return EpisodeStore.for_run(
        runs_dir,
        suite=block.split,
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=block.skill_id,
    ).path


def plan_records(
    plan: Sequence[Block], runs_dir: Path, inputs: dict[str, str]
) -> list[EpisodeRecord]:
    """계획 블록의 저장 파일에서 계획에 있는 (skill, split, repeat, case)만. 수가 다르면 거부."""
    wanted = {(b.skill_id, b.split, b.repeat, c) for b in plan for c in b.cases}
    by_path: dict[Path, str] = {}
    for b in plan:
        by_path.setdefault(_store_path(runs_dir, b), b.split)
    records: list[EpisodeRecord] = []
    for path, split in sorted(by_path.items()):
        if not path.exists():
            raise ValueError(f"missing store {path}")
        inputs[_rel(path, runs_dir)] = _sha(path)
        for r in EpisodeStore(path, repair=False).records():
            case = (r.key.family, r.key.level, r.key.incident_seed)
            if (r.key.skill_id, split, r.key.repeat, case) in wanted:
                records.append(r)
    episodes = sum(len(b.cases) for b in plan)
    if len(records) != episodes:
        raise ValueError(f"kept {len(records)} records, the plan has {episodes} episodes")
    return records


def main_run_009b(records: Sequence[EpisodeRecord], log: Path, names: Mapping[str, str]) -> Section:
    sessions = session_summary(log)
    timed = [{"start": s["start"], "end": s["end"]} for s in sessions["sessions"]]
    run = {
        "sessions": [
            {k: v for k, v in s.items() if k not in ("start", "end")} for s in sessions["sessions"]
        ],
        "server_error_events": sessions["server_error_events"],
        "records": record_summary(records, names),
        "idle": idle_gaps(records),
        "idle_spans": idle_spans(records, timed),
    }
    return run, {"sessions": timed}


def pool_names(pool: Mapping[str, Any]) -> dict[str, str]:
    return {S0_SKILL_ID: "S0", **{c["skill_id"]: c["name"] for c in pool["candidates"]}}


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run_summary_009b(
    runs_dir: Path, root: Path = REPO_ROOT, *, pins: Mapping[str, str] | None = None
) -> dict[str, Any]:
    pins = dict(DEFAULT_PINS if pins is None else pins)
    definition = root / DEFINITION_PATH
    if _sha(definition) != pins["definition"]:
        raise ValueError(f"{DEFINITION_PATH} sha256 {_sha(definition)} != {pins['definition']}")
    inputs: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, (rel, _) in PINS.items():
        base = runs_dir if rel in (SESSIONS_009B, SEQ_STORE, PAR_STORE) else root
        path = base / rel
        if not path.exists():
            raise ValueError(f"missing input {path}")
        actual = _sha(path)
        if actual != pins[name]:
            raise ValueError(f"{name}: {path} sha256 {actual} != pinned {pins[name]}")
        inputs[_rel(path, runs_dir) if base is runs_dir else rel.as_posix()] = actual
        paths[name] = path

    manifest = json.loads(paths["t009_manifest"].read_text(encoding="utf-8"))
    pool = json.loads(paths["t009b_pool"].read_text(encoding="utf-8"))
    plan = load_plan(paths["t009b_plan"])
    a_run, a_prov = pool_009(manifest)
    c_run, c_prov = proposer_009b(pool)
    b_run, b_prov = main_run_009b(
        plan_records(plan, runs_dir, inputs), paths["t009b_sessions"], pool_names(pool)
    )
    d_run, d_prov = same_seed(
        EpisodeStore(paths["calibration_sequential"], repair=False).records(),
        EpisodeStore(paths["calibration_concurrent"], repair=False).records(),
    )
    return {
        "run": {
            "t009": {"pool": a_run},
            "t009b": {"main": b_run, "proposer": c_run},
            "same_seed": d_run,
        },
        "provenance": {
            "t009": {"pool": a_prov},
            "t009b": {"main": b_prov, "proposer": c_prov},
            "same_seed": d_prov,
            "definition": {"path": DEFINITION_PATH.as_posix(), "sha256": pins["definition"]},
            "inputs": dict(sorted(inputs.items())),
        },
        "notes": list(NOTES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-run-summary-009b")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out: Path = args.out or args.root / DEFAULT_OUT
    try:
        summary = build_run_summary_009b(args.runs_dir, args.root, pins=DEFAULT_PINS)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    data = summary_bytes(summary)
    if out.exists() and out.read_bytes() != data:
        print(f"error: {out} exists and differs from the summary built now", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"run summary 009b {out}: sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
