"""P3 시간 측정 요약 (Ticket 008 §4 P3). LLM은 쓰지 않는다.

    uv run python -m alfworld_pilot.p3

입력(`runs/p3/`, git 무시): `episodes.jsonl`, `memory.log`(30초 표본), `server_log_offset.txt`와
Ollama 서버 로그(`~/.ollama/logs/server.log`, 읽기만). 출력: `records/p3_summary.json`.

- 처리량 s̄: 실제 벽시계(첫 시작 ~ 마지막 끝) / 에피소드 수. 모델 적재 포함과 제외 두 값을 낸다
  (2026-11-02 결정 1). 적재 시간은 실행 구간의 서버 로그 "llama-server started in N seconds"에서
  읽는다. G5 추정에는 제외 값을 쓴다.
- 실효 동시 수: 에피소드 시간 합 / 벽시계.
- 프롬프트 토큰 최댓값을 6,144(= num_ctx 8192 - 생성 몫 2048)와 비교한다(결정 3).
- 원칙 9 해당 여부: 성공 0개 또는 전부.
- E-a·E-b·E-c는 참고로 같이 센다(기준률 판단은 P4).
"""

import hashlib
import json
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alfworld_pilot import constants as C
from alfworld_pilot.evidence import StepRecord, e_a_holds, e_b_rate, e_c_holds

PROMPT_LIMIT = C.NUM_CTX - C.RESERVE_TOKENS
RUN_DIR = Path("runs/p3")
OUT = Path("records/p3_summary.json")
SERVER_LOG = Path.home() / ".ollama" / "logs" / "server.log"
_LOAD = re.compile(r"llama-server started in ([0-9.]+) seconds")
_FREE = re.compile(r"System-wide memory free percentage: (\d+)%")
_PS_COLUMNS = ("NAME", "SIZE", "PROCESSOR", "CONTEXT")


def stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def principle_9_case(won: int, episodes: int) -> str:
    if won == 0:
        return "zero_successes"
    if won == episodes:
        return "all_successes"
    return "not_applicable"


def load_seconds(server_log: str) -> list[float]:
    """적재마다 한 값. 한 적재("starting llama-server")를 기다린 요청마다 "started in" 줄이 따로
    찍히므로(동시 2면 두 줄), 같은 적재 안에서는 가장 큰 값 하나만 센다."""
    out = []
    for segment in server_log.split('msg="starting llama-server"'):
        values = [float(m) for m in _LOAD.findall(segment)]
        if values:
            out.append(max(values))
    return out


def _ps_row(header: str, line: str) -> str:
    starts = [header.index(c) for c in _PS_COLUMNS]
    cols = header.split()
    ends = []
    for name in _PS_COLUMNS:
        i = cols.index(name)
        ends.append(header.index(cols[i + 1]) if i + 1 < len(cols) else len(line))
    return " ".join(" ".join(line[s:e].split()) for s, e in zip(starts, ends, strict=True))


def memory_summary(log: str) -> dict[str, Any]:
    free = [int(m) for m in _FREE.findall(log)]
    rows: list[str] = []
    header = ""
    for line in log.splitlines():
        if line.startswith("NAME"):
            header = line
        elif header and line.strip() and not _FREE.search(line) and line[:1].isalpha():
            row = _ps_row(header, line)
            if row not in rows:
                rows.append(row)
    return {
        "samples": len(free),
        "min_free_percent": min(free) if free else None,
        "max_free_percent": max(free) if free else None,
        "ollama_ps": rows,
    }


def _steps(record: Mapping[str, Any]) -> list[StepRecord]:
    return [StepRecord(s["command"], s["observation"], s["admissible"]) for s in record["steps"]]


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    load_s: Sequence[float],
    expert_steps: Mapping[str, int],
) -> dict[str, Any]:
    n = len(records)
    won = [r for r in records if r["won"]]
    start = min(r["started_at"] for r in records)
    end = max(r["finished_at"] for r in records)
    span = end - start
    load = sum(load_s) if load_s else None
    prompts = [p for r in records for p in r["prompt_tokens"]]
    over = [p for p in prompts if p > PROMPT_LIMIT]
    admissible = total = 0
    for r in records:
        a, t = e_b_rate(_steps(r))
        admissible, total = admissible + a, total + t
    e_c = [
        e_c_holds(won=r["won"], steps=len(r["steps"]), expert_steps=expert_steps.get(r["rel"]))
        for r in records
    ]
    return {
        "episodes": n,
        "won": len(won),
        "won_by_type": dict(sorted(Counter(r["task_type"] for r in won).items())),
        "episodes_by_type": dict(sorted(Counter(r["task_type"] for r in records).items())),
        "principle_9": principle_9_case(len(won), n),
        "throughput": {
            "wall_clock_s": span,
            "load_s": load,
            "s_bar_including_load": span / n,
            "s_bar_excluding_load": (span - load) / n if load is not None else None,
            "effective_concurrency": sum(r["wall_s"] for r in records) / span,
            "concurrency": sorted({r["run"]["concurrency"] for r in records}),
        },
        "per_episode": {
            "wall_s": stats([r["wall_s"] for r in records]),
            "steps": stats([len(r["steps"]) for r in records]),
            "llm_calls": stats([r["llm_calls"] for r in records]),
            "max_prompt_tokens": stats([max(r["prompt_tokens"] or [0]) for r in records]),
            "s_per_llm_call": stats(
                [r["wall_s"] / r["llm_calls"] for r in records if r["llm_calls"]]
            ),
        },
        "prompt_tokens": stats(prompts),
        "completion_tokens": stats([c for r in records for c in r["completion_tokens"]]),
        "prompt_tokens_over_limit": {
            "limit": PROMPT_LIMIT,
            "exceeded": bool(over),
            "calls": len(over),
        },
        "stop_reasons": dict(sorted(Counter(r["stop_reason"] for r in records).items())),
        "format_errors": sum(r["format_errors"] for r in records),
        "text_turns": sum(r["text_turns"] for r in records),
        "truncated_calls": sum(r["truncated_calls"] for r in records),
        "max_omitted": max(r["max_omitted"] for r in records),
        "e_a": {
            "won": len(won),
            "holds": sum(1 for r in won if e_a_holds(r["first_observation"], _steps(r), won=True)),
        },
        "e_b": {"admissible_steps": admissible, "steps": total},
        "e_c": {"defined": sum(1 for x in e_c if x is not None), "holds": sum(1 for x in e_c if x)},
    }


def _expert_steps(path: Path) -> dict[str, int]:
    games = json.loads(path.read_text())["games"]
    return {rel: g["steps"] for rel, g in games.items() if g["won"]}


def main() -> int:
    episodes = RUN_DIR / "episodes.jsonl"
    records = [json.loads(line) for line in episodes.read_text().splitlines()]
    offset = int((RUN_DIR / "server_log_offset.txt").read_text())
    with SERVER_LOG.open("rb") as f:
        f.seek(offset)
        excerpt = f.read().decode(errors="replace")
    (RUN_DIR / "server_log_excerpt.txt").write_text(excerpt)
    summary = summarize(
        records,
        load_s=load_seconds(excerpt),
        expert_steps=_expert_steps(Path("records/p1_expert.json")),
    )
    summary["memory"] = memory_summary((RUN_DIR / "memory.log").read_text())
    summary["config"] = records[0]["config"]
    summary["sources"] = {
        name: hashlib.sha256((RUN_DIR / name).read_bytes()).hexdigest()
        for name in ("episodes.jsonl", "memory.log", "server_log_excerpt.txt")
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("won", "principle_9", "throughput")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
