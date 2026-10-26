"""실행 요약 (Ticket 007 §1의 세 번째 출처: "run logs, not part of the analysis").

    uv run python -m evidencegate.run_summary --runs-dir runs --out writeup/inputs/run_summary.json

- 실행 기록(`runs/<split>/…/<skill>.jsonl`)과 세션 기록(`runs/ticket00N/sessions.jsonl`)을
  **읽기만** 해서 운영 지표를 모은다. Ticket 004 숫자는 git에 있는
  `skills/candidates/POOL.json`에서 읽는다. proposer 장부(`runs/proposer/ledger.jsonl`)는
  읽지 않는다.
- 담는 것: 에피소드 수, 형식 오류, 처분 없음, 종료 이유, 도구 호출 수, 정책 시간, 벽시계, 정지 구간,
  세션, Ticket 004 호출·비용·제외. **CD·ES·조치 종류는 담지 않는다.**
- 읽은 입력 파일마다 sha256을 남긴다. 결과 파일 두 개는 건드리지 않는다.
- 같은 입력이면 같은 바이트다(정렬된 키, 생성 시각 없음).
- Ticket 005 기록은 실행 계획의 12,540개다. Ticket 004에서 실행한 S0 Dev-A 60개는 시간대가 달라
  뺀다.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT, load_pool
from evidencegate.eval_constants import (
    CONFIG_ID,
    EVAL_SPLITS,
    IDLE_GAP_S,
    L3_CHECK_SPLIT,
    L3_ORACLE_SPLIT,
    POLICY_ID,
    SESSION_LOG,
    TOOL_SPEC_ID,
)
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore

POOL_PATH = Path("skills/candidates/POOL.json")
L3_SESSION_LOG_PATH = "ticket006/sessions.jsonl"
DEFAULT_OUT = Path("writeup/inputs/run_summary.json")
NOTES = (
    "Run logs, not part of the analysis: operational numbers read from the episode records and "
    "session logs; they never enter a verdict.",
    "Ticket 005 records are the 12,540 planned episodes; the 60 S0 Dev-A episodes run in "
    "Ticket 004 are not included.",
    f"An idle span is a stretch of at least {IDLE_GAP_S:g} s with no episode running.",
    "Ticket 004 numbers come from skills/candidates/POOL.json; the proposer ledger is not read.",
)


def _t(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _r(x: float) -> float:
    return round(x, 3)


def record_summary(records: Sequence[EpisodeRecord], names: Mapping[str, str]) -> dict[str, Any]:
    if not records:
        raise ValueError("no records")
    calls = [len(r.trace.turns) for r in records]
    starts = [_t(r.started_at) for r in records]
    ends = [s + timedelta(seconds=r.duration_s) for s, r in zip(starts, records, strict=True)]
    by_skill = Counter(names.get(r.key.skill_id, r.key.skill_id) for r in records)
    errors = Counter(
        names.get(r.key.skill_id, r.key.skill_id) for r in records if r.trace.format_errors > 0
    )
    return {
        "episodes": len(records),
        "episodes_by_skill": dict(sorted(by_skill.items())),
        "format_error_episodes": dict(sorted(errors.items())),
        "format_errors_total": sum(r.trace.format_errors for r in records),
        "no_disposition": sum(1 for r in records if r.action is None),
        "stop_reasons": dict(sorted(Counter(str(r.trace.stop_reason) for r in records).items())),
        "tool_calls": {
            "mean": _r(sum(calls) / len(calls)),
            "min": min(calls),
            "max": max(calls),
            "counts": {str(k): v for k, v in sorted(Counter(calls).items())},
        },
        "policy_time_mean_s": _r(sum(r.duration_s for r in records) / len(records)),
        "first_start": _iso(min(starts)),
        "last_end": _iso(max(ends)),
        "wall_s": _r((max(ends) - min(starts)).total_seconds()),
    }


def idle_gaps(records: Sequence[EpisodeRecord], min_gap_s: float = IDLE_GAP_S) -> dict[str, Any]:
    """에피소드가 하나도 돌지 않은 구간 중 min_gap_s 이상인 것, 그리고 한 개 이상 돈 시간."""
    spans = sorted(
        (_t(r.started_at), _t(r.started_at) + timedelta(seconds=r.duration_s)) for r in records
    )
    merged: list[list[datetime]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [(a[1], b[0]) for a, b in pairwise(merged) if (b[0] - a[1]).total_seconds() >= min_gap_s]
    lengths = [(b - a).total_seconds() for a, b in gaps]
    return {
        "min_gap_s": min_gap_s,
        "count": len(gaps),
        "total_s": _r(sum(lengths)),
        "longest_s": _r(max(lengths, default=0.0)),
        "first_start": _iso(gaps[0][0]) if gaps else None,
        "last_end": _iso(gaps[-1][1]) if gaps else None,
        "active_s": _r(sum((e - s).total_seconds() for s, e in merged)),
    }


def session_summary(log: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    sessions: list[dict[str, Any]] = []
    open_start: str | None = None
    probe: float | None = None
    for e in events:
        if e["event"] == "probe":
            probe = e.get("ratio")
        elif e["event"] == "session_start":
            if open_start is not None:
                raise ValueError(f"{log}: session at {open_start} without an end")
            open_start = e["time"]
        elif e["event"] == "session_end":
            if open_start is None:
                raise ValueError(f"{log}: session end at {e['time']} without a start")
            sessions.append(
                {
                    "start": open_start,
                    "end": e["time"],
                    "duration_s": _r((_t(e["time"]) - _t(open_start)).total_seconds()),
                    "reason": e.get("reason"),
                    "blocks_run": e.get("blocks_run"),
                    "episodes_run": e.get("episodes_run"),
                    "episodes_failed": e.get("episodes_failed"),
                    "server_errors": e.get("server_errors"),
                    "probe_ratio": probe,
                }
            )
            open_start, probe = None, None
    if open_start is not None:
        raise ValueError(f"{log}: session at {open_start} without an end")
    return {
        "sessions": sessions,
        "server_error_events": sum(1 for e in events if e["event"] == "server_error"),
    }


def ticket004_summary(pool: Mapping[str, Any]) -> dict[str, Any]:
    config = pool["proposer_config"]
    return {
        "model": config["model"],
        "reasoning_effort": config.get("reasoning_effort"),
        "budget_usd": config["budget_usd"],
        "target": pool["target"],
        "pool_size": pool["pool_size"],
        "shortfall": pool["shortfall"],
        "calls": pool["calls"],
        "format_errors": pool["format_errors"],
        "excluded": dict(sorted(Counter(x["reason"] for x in pool["excluded"]).items())),
        "candidates_overlapping_a_control": sum(
            1 for c in pool["proposer_candidates"] if c["control_duplicates"]
        ),
        "controls": len(pool["controls"]),
        "pool_cost_usd": pool["pool_cost_usd"],
        "ledger_cumulative_usd": pool["ledger_cumulative_usd"],
        "frozen_at": pool["frozen_at"],
    }


def default_stores(runs_dir: Path, root: Path) -> dict[str, list[Path]]:
    pool = load_pool(root)
    candidates = [c.skill_id for c in pool.candidates]

    def path(split: str, skill: str) -> Path:
        return EpisodeStore.for_run(
            runs_dir,
            suite=split,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id=skill,
        ).path

    t005 = [path(s, c) for s in EVAL_SPLITS for c in candidates]
    t005 += [path(s, pool.s0_skill_id) for s in EVAL_SPLITS if s != "dev_a"]
    return {
        "ticket005": t005,
        "ticket006": [path(L3_ORACLE_SPLIT, s) for s in (*candidates, pool.s0_skill_id)],
        "ticket006_check": [path(L3_CHECK_SPLIT, pool.s0_skill_id)],
    }


def default_names(root: Path) -> dict[str, str]:
    pool = load_pool(root)
    return {pool.s0_skill_id: "S0", **{c.skill_id: c.name for c in pool.candidates}}


def build_run_summary(
    runs_dir: Path,
    root: Path = REPO_ROOT,
    *,
    stores: Mapping[str, Sequence[Path]] | None = None,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    stores = stores if stores is not None else default_stores(runs_dir, root)
    names = names if names is not None else default_names(root)
    inputs: dict[str, str] = {}

    def read(paths: Sequence[Path]) -> list[EpisodeRecord]:
        out: list[EpisodeRecord] = []
        for p in paths:
            inputs[_rel(p, runs_dir)] = _sha(p)
            out += EpisodeStore(p, repair=False).records()
        return out

    def section(paths: Sequence[Path], log: Path | None) -> dict[str, Any]:
        records = read(paths)
        out: dict[str, Any] = {
            "records": record_summary(records, names) if records else None,
            "idle": idle_gaps(records) if records else None,
        }
        if log is not None:
            inputs[_rel(log, runs_dir)] = _sha(log)
            out["sessions"] = session_summary(log)
        return out

    pool_path = root / POOL_PATH
    inputs[POOL_PATH.as_posix()] = _sha(pool_path)
    summary = {
        "ticket004": ticket004_summary(json.loads(pool_path.read_text(encoding="utf-8"))),
        "ticket005": section(stores["ticket005"], runs_dir / SESSION_LOG),
        "ticket006": section(stores["ticket006"], runs_dir / L3_SESSION_LOG_PATH),
        "ticket006_check": section(stores["ticket006_check"], None),
        "notes": list(NOTES),
    }
    summary["inputs"] = dict(sorted(inputs.items()))
    return summary


def summary_bytes(summary: Mapping[str, Any]) -> bytes:
    return (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _rel(path: Path, runs_dir: Path) -> str:
    """runs/ 아래 경로는 'runs/…'로 적는다(장비마다 다른 절대 경로를 남기지 않음)."""
    return (Path("runs") / path.resolve().relative_to(runs_dir.resolve())).as_posix()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(
    argv: Sequence[str] | None = None,
    *,
    stores: Mapping[str, Sequence[Path]] | None = None,
    names: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-run-summary")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    data = summary_bytes(build_run_summary(args.runs_dir, args.root, stores=stores, names=names))
    if args.out.exists() and args.out.read_bytes() != data:
        print(f"error: {args.out} exists and differs from the summary built now", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    args.out.with_name(args.out.name + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"run summary {args.out}: sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
