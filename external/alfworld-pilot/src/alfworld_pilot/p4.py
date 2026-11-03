"""P4 S0 기준률과 잡음 (Ticket 008 §4 P4). LLM은 쓰지 않는다.

    uv run python -m alfworld_pilot.p4 --run-dir runs/p4a --out records/p4a_summary.json
    uv run python -m alfworld_pilot.p4 --run-dir runs/p4a runs/p4b --out records/p4_summary.json

P5 표적 칸 규칙(2026-11-03 결정 3, constants.P5_TARGET_RULE): P4 반복 3회 평균 성공률이 가장
높은 칸, 동률이면 SIX_TYPES 순서에서 앞. 기준률이 바닥인 칸은 하락을 보일 수 없어서(G6)
미리 정하지 않고 규칙으로 고정한다.

요약(결정 2: G4·G3는 "P4-a 기준 미리 보기", 최종 판정은 P6):
- 칸별 성공률(에피소드 전체와 반복별), Clopper-Pearson 95%(`evidencegate.exact`), d_c(반복 쌍
  불일치율의 3쌍 평균), 종료 이유
- G4 미리 보기: 점추정이 [0.15, 0.85] 안인 칸 수 >= 4
- G3 미리 보기: 고정 식(z 1.96, h 0.20, d 하한 0.10, n_B 35/50, n_A 10)과 유형별 남는 train 수
- E-a 기준률과 "성공 중 E-a 불성립 비율"(결론 규칙), E-b, E-c
- 텍스트 턴 진단 요약(유형별), 전문가 실패 게임 표시, 처리량(적재 포함·제외), 메모리
"""

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evidencegate.exact import clopper_pearson

from alfworld_pilot.evidence import StepRecord, e_a_holds, e_b_rate, e_c_holds
from alfworld_pilot.p1 import SIX_TYPES
from alfworld_pilot.p3 import SERVER_LOG, load_seconds, memory_summary, stats

REPEATS = 3
G4_RANGE = (0.15, 0.85)
G4_MIN_CELLS = 4
G3 = {"z": 1.96, "h": 0.2, "d_floor": 0.1, "n_B_num": 35, "n_B_den": 50, "n_A": 10}
POOL_PER_TYPE = 30  # 파일럿 풀에서 유형마다 뺀 수(P1)


def choose_target_cell(mean_success: Mapping[str, float]) -> str:
    if set(mean_success) != set(SIX_TYPES):
        raise ValueError("need the mean success rate of all six task types")
    best = max(mean_success[t] for t in SIX_TYPES)
    return next(t for t in SIX_TYPES if mean_success[t] == best)


def disagreement_rate(outcomes: Mapping[str, Sequence[bool]]) -> float:
    """같은 게임에서 두 반복의 성공 여부가 다른 비율, 반복 쌍 3개의 평균(G3 d_c)."""
    if any(len(o) != REPEATS for o in outcomes.values()):
        raise ValueError(f"each game needs {REPEATS} repeats")
    pairs = list(itertools.combinations(range(REPEATS), 2))
    games = list(outcomes.values())
    return sum(sum(o[a] != o[b] for o in games) / len(games) for a, b in pairs) / len(pairs)


def _by_game(records: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for r in sorted(records, key=lambda r: (r["rel"], r["repeat"])):
        out.setdefault(r["rel"], []).append(bool(r["won"]))
    return out


def cell_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    won = sum(1 for r in records if r["won"])
    by_repeat = []
    for k in range(REPEATS):
        rs = [r for r in records if r["repeat"] == k]
        by_repeat.append(sum(1 for r in rs if r["won"]) / len(rs) if rs else None)
    games = _by_game(records)
    complete = {g: o for g, o in games.items() if len(o) == REPEATS}
    return {
        "games": len(games),
        "episodes": len(records),
        "won": won,
        "success_rate": won / len(records),
        "success_rate_by_repeat": by_repeat,
        "clopper_pearson_95": clopper_pearson(won, len(records)),
        "d_c": disagreement_rate(complete) if complete else None,
        "stop_reasons": dict(sorted(Counter(r["stop_reason"] for r in records).items())),
    }


def g4_preview(rates: Mapping[str, float]) -> dict[str, Any]:
    inside = [t for t in SIX_TYPES if G4_RANGE[0] <= rates[t] <= G4_RANGE[1]]
    return {
        "range": list(G4_RANGE),
        "inside": len(inside),
        "inside_cells": inside,
        "needed": G4_MIN_CELLS,
        "passes_preview": len(inside) >= G4_MIN_CELLS,
    }


def g3_preview(d: Mapping[str, float], available: Mapping[str, int]) -> dict[str, Any]:
    z, h = G3["z"], G3["h"]
    d_hat = {t: max(d[t], G3["d_floor"]) for t in SIX_TYPES}
    n_o = {t: math.ceil(z**2 * d_hat[t] / (h / 2) ** 2 - 1e-9) for t in SIX_TYPES}
    n_b = {t: math.ceil(n_o[t] * G3["n_B_num"] / G3["n_B_den"]) for t in SIX_TYPES}
    n_min = max(G3["n_A"] + n_b[t] + n_o[t] for t in SIX_TYPES)
    return {
        "d_hat": d_hat,
        "n_O": n_o,
        "n_B": n_b,
        "n_A": G3["n_A"],
        "n_min": n_min,
        "available_train": dict(available),
        "passes_preview": all(available[t] >= n_min for t in SIX_TYPES),
    }


def text_turn_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in sorted({r["task_type"] for r in records}):
        diags = [
            d for r in records if r["task_type"] == t for d in r.get("text_turn_diagnostics", [])
        ]
        openings = Counter(" ".join(d["content"].split())[:60] for d in diags)
        out[t] = {
            "text_turns": len(diags),
            "empty_content": sum(1 for d in diags if not d["content"].strip()),
            "content_chars": stats([d["content_chars"] for d in diags]),
            "thinking_chars": stats([d["thinking_chars"] for d in diags]),
            "top_openings": [[k, n] for k, n in openings.most_common(5)],
        }
    return out


def throughput(records: Sequence[Mapping[str, Any]], *, load_s: Sequence[float]) -> dict[str, Any]:
    """한 실행(run 폴더)의 처리량. 실행 사이의 빈 시간을 넣지 않도록 실행마다 따로 계산한다."""
    span = max(r["finished_at"] for r in records) - min(r["started_at"] for r in records)
    load = sum(load_s) if load_s else None
    return {
        "episodes": len(records),
        "wall_clock_s": span,
        "load_s": load,
        "s_bar_including_load": span / len(records),
        "s_bar_excluding_load": (span - load) / len(records) if load is not None else None,
        "effective_concurrency": sum(r["wall_s"] for r in records) / span,
    }


def _steps(r: Mapping[str, Any]) -> list[StepRecord]:
    return [StepRecord(s["command"], s["observation"], s["admissible"]) for s in r["steps"]]


def summarize_p4(
    records: Sequence[Mapping[str, Any]],
    *,
    load_s: Sequence[float],
    expert_steps: Mapping[str, int],
    expert_failed: set[str],
    available_train: Mapping[str, int],
) -> dict[str, Any]:
    cells = {
        t: cell_summary([r for r in records if r["task_type"] == t])
        for t in SIX_TYPES
        if any(r["task_type"] == t for r in records)
    }
    rates = {t: c["success_rate"] for t, c in cells.items()}
    full = set(cells) == set(SIX_TYPES)
    won = [r for r in records if r["won"]]
    holds = sum(1 for r in won if e_a_holds(r["first_observation"], _steps(r), won=True))
    admissible = total = 0
    for r in records:
        a, n = e_b_rate(_steps(r))
        admissible, total = admissible + a, total + n
    e_c = [
        e_c_holds(won=r["won"], steps=len(r["steps"]), expert_steps=expert_steps.get(r["rel"]))
        for r in records
    ]
    games = _by_game(records)
    return {
        "preview_only": True,
        "episodes": len(records),
        "won": len(won),
        "cells": cells,
        "target_cell": choose_target_cell(rates) if full else None,
        "g4_preview": g4_preview(rates) if full else None,
        "g3_preview": (
            g3_preview({t: c["d_c"] or 0.0 for t, c in cells.items()}, available_train)
            if full
            else None
        ),
        "e_a": {
            "episodes": len(records),
            "won": len(won),
            "holds": holds,
            "base_rate": holds / len(records),
            "share_of_wins_failing": (len(won) - holds) / len(won) if won else None,
        },
        "e_b": {"admissible_steps": admissible, "steps": total},
        "e_c": {"defined": sum(1 for x in e_c if x is not None), "holds": sum(1 for x in e_c if x)},
        "text_turns": text_turn_summary(records),
        "expert_failed_games": [
            {"rel": g, "won": o} for g, o in sorted(games.items()) if g in expert_failed
        ],
        "throughput": throughput(records, load_s=load_s),
        "per_episode": {
            "wall_s": stats([r["wall_s"] for r in records]),
            "steps": stats([len(r["steps"]) for r in records]),
            "llm_calls": stats([r["llm_calls"] for r in records]),
            "max_prompt_tokens": stats([max(r["prompt_tokens"] or [0]) for r in records]),
        },
        "truncated_calls": sum(r["truncated_calls"] for r in records),
        "format_errors": sum(r["format_errors"] for r in records),
        "text_turn_total": sum(r["text_turns"] for r in records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-p4")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for run_dir in args.run_dir:
        rs = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
        excerpt_file = run_dir / "server_log_excerpt.txt"
        if excerpt_file.exists():
            # 이미 자른 구간을 그대로 쓴다. 다시 자르면 뒤 실행의 로그가 섞인다
            excerpt = excerpt_file.read_text()
        else:
            offset = int((run_dir / "server_log_offset.txt").read_text())
            with SERVER_LOG.open("rb") as f:
                f.seek(offset)
                excerpt = f.read().decode(errors="replace")
            excerpt_file.write_text(excerpt)
        runs[run_dir.name] = {
            "throughput": throughput(rs, load_s=load_seconds(excerpt)),
            "memory": memory_summary((run_dir / "memory.log").read_text()),
        }
        for name in ("episodes.jsonl", "memory.log", "server_log_excerpt.txt"):
            sources[f"{run_dir.name}/{name}"] = hashlib.sha256(
                (run_dir / name).read_bytes()
            ).hexdigest()
        records += rs
    expert = json.loads(Path("records/p1_expert.json").read_text())["games"]
    counts = json.loads(Path("records/p1_counts.json").read_text())["table"]["train"]["playable"]
    summary = summarize_p4(
        records,
        load_s=[],
        expert_steps={rel: g["steps"] for rel, g in expert.items() if g["won"]},
        expert_failed={rel for rel, g in expert.items() if not g["won"]},
        available_train={t: counts[t] - POOL_PER_TYPE for t in SIX_TYPES},
    )
    summary["throughput"] = None  # 실행마다 따로: runs.<이름>.throughput
    summary["runs"] = runs
    summary["config"] = records[0]["config"]
    summary["sources"] = sources
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("won", "target_cell", "runs")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
