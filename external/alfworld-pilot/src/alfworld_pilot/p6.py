"""P6 판정 계산 (Ticket 008 §3, 고정 파일 `fixed/p3_fixed.json`). LLM은 쓰지 않는다.

- G6: 칸마다 Δ = 탐침 1회 성공 - S0 반복 3회 다수결의 게임 평균. 잡음은 게임을 복원 추출(|G|개,
  2,000회, seed 0, `evidencegate.exact.stratified_bootstrap_indices`)하고, 재표본마다 S0 반복 하나
  r을 골라(`random.Random(0)` 순서) (S0_r - 다수결)의 게임 평균을 낸 분포다. q_lo / q_hi는 그
  분포의 2.5 / 97.5 백분위수(`percentile_exact`, 선형 보간). 통과: 표적 칸 Δ < q_lo 그리고 비표적
  칸마다 Δ가 [q_lo, q_hi] 안. 비교는 유리수로 한다.
- G5: E_009 = 후보 수 x 칸 합 + S0 반복 x 칸 합, T_009 = E_009 x s̄ x 1.1.
- 게임 단위 CP(2026-11-03 결정 2): 게임마다 반복 다수결, n = 게임 수.
- 진단 짝 비교(H1·H2): exploratory diagnostic, not used for G1-G6.
"""

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate.exact import clopper_pearson, percentile_exact, stratified_bootstrap_indices

from alfworld_pilot.p0 import data_root
from alfworld_pilot.p1 import SIX_TYPES, GameInfo, scan_split
from alfworld_pilot.p4 import g3_preview, g4_preview, text_turn_summary

G6_RESAMPLES = 2000
G6_SEED = 0
G6_PERCENTILES = (Fraction(25, 1000), Fraction(975, 1000))
G5_MARGIN = 1.1
DIAG_LABEL = "exploratory diagnostic, not used for G1-G6"
STOPS = ("won", "max_steps", "no_tool_call", "format_error", "max_llm_calls")


def majority(outcomes: Sequence[bool]) -> bool:
    if len(outcomes) != 3:
        raise ValueError("majority needs exactly three repeats")
    return sum(outcomes) >= 2


def g6_cell(probe: Mapping[str, bool], s0: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    games = sorted(s0)
    if set(probe) != set(games):
        raise ValueError("probe and S0 must cover the same games")
    maj = [int(majority(s0[g])) for g in games]
    delta = Fraction(sum(int(probe[g]) for g in games) - sum(maj), len(games))
    (index,) = stratified_bootstrap_indices([len(games)], G6_RESAMPLES, G6_SEED)
    pick = random.Random(G6_SEED)
    noise = []
    for row in index:
        r = pick.randrange(3)
        total = sum(int(s0[games[i]][r]) - maj[i] for i in row)
        noise.append(Fraction(total, len(games)))
    return {
        "games": len(games),
        "delta": delta,
        "q_lo": percentile_exact(noise, G6_PERCENTILES[0]),
        "q_hi": percentile_exact(noise, G6_PERCENTILES[1]),
    }


def g6_judgement(cells: Mapping[str, Mapping[str, Fraction]], *, target: str) -> dict[str, Any]:
    t = cells[target]
    outside = [
        c
        for c in SIX_TYPES
        if c != target
        and c in cells
        and not cells[c]["q_lo"] <= cells[c]["delta"] <= cells[c]["q_hi"]
    ]
    target_drop = t["delta"] < t["q_lo"]
    return {
        "target": target,
        "target_drop": target_drop,
        "non_targets_inside": not outside,
        "non_target_outside": outside,
        "passes": target_drop and not outside,
    }


def g5_estimate(
    sizes: Mapping[str, Mapping[str, int]],
    *,
    s_bar: float,
    candidates: int,
    s0_repeats: int,
    budget_days: float,
) -> dict[str, Any]:
    per_candidate = sum(s["n_A"] + s["n_B"] + s["n_O"] for s in sizes.values())
    e = candidates * per_candidate + s0_repeats * per_candidate
    t = e * s_bar * G5_MARGIN
    return {
        "per_candidate": per_candidate,
        "candidates": candidates,
        "s0_repeats": s0_repeats,
        "E_009": e,
        "s_bar": s_bar,
        "T_009_s": t,
        "T_009_days": t / 86400,
        "budget_days": budget_days,
        "passes": t <= budget_days * 86400,
    }


def game_level_cp(games: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    won = sum(1 for o in games.values() if majority(o))
    return {
        "games": len(games),
        "won_majority": won,
        "clopper_pearson_95": clopper_pearson(won, len(games)),
    }


def stop_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in sorted({r["task_type"] for r in records}):
        rs = [r for r in records if r["task_type"] == t]
        counts = Counter(r["stop_reason"] for r in rs)
        out[t] = {
            "episodes": len(rs),
            **{s: counts.get(s, 0) for s in STOPS},
            "episodes_with_format_errors": sum(1 for r in rs if r["format_errors"]),
        }
    return out


def diag_compare(
    baseline: Sequence[Mapping[str, Any]], variant: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """같은 게임·반복 짝 비교. 표본이 작아 큰 차이만 해석한다."""
    base = {r["key"]: r for r in baseline}
    var = {r["key"]: r for r in variant}
    if set(base) != set(var):
        raise ValueError("baseline and variant must have the same episodes")
    cells: dict[str, Any] = {}
    for t in sorted({r["task_type"] for r in baseline}):
        keys = [k for k in sorted(base) if base[k]["task_type"] == t]

        def side(d: Mapping[str, Mapping[str, Any]], ks: list[str] = keys) -> dict[str, Any]:
            return {
                "won": sum(1 for k in ks if d[k]["won"]),
                "mean_steps": sum(len(d[k]["steps"]) for k in ks) / len(ks),
                "text_turns": sum(d[k]["text_turns"] for k in ks),
            }

        b, v = side(base), side(var)
        cells[t] = {
            "games": len(keys),
            "won": {"baseline": b["won"], "variant": v["won"]},
            "mean_steps": {"baseline": b["mean_steps"], "variant": v["mean_steps"]},
            "text_turns": {"baseline": b["text_turns"], "variant": v["text_turns"]},
        }
    moves = Counter(f"{base[k]['stop_reason']} -> {var[k]['stop_reason']}" for k in base)
    return {
        "label": DIAG_LABEL,
        "paired_games": len(base),
        "cells": cells,
        "moves": dict(sorted(moves.items())),
        "won": {
            "baseline": sum(1 for r in baseline if r["won"]),
            "variant": sum(1 for r in variant if r["won"]),
        },
    }


def available_after_folder_exclusion(
    train: Sequence[GameInfo], pool_rels: Sequence[str]
) -> dict[str, int]:
    """파일럿 풀의 과제 폴더에 속한 train trial을 모두 뺀 유형별 플레이 가능 게임 수.

    009 제외 규칙(과제 폴더 단위)을 G3 남는 수에 반영한다.
    """
    folders = {rel.rsplit("/", 1)[0] for rel in pool_rels}
    out = dict.fromkeys(SIX_TYPES, 0)
    for g in train:
        if g.playable and g.rel.rsplit("/", 1)[0] not in folders:
            out[g.task_type] += 1
    return out


GATES = ("G1", "G2", "G3", "G4", "G5", "G6")


def overall_verdict(gates: Mapping[str, str], *, reduced_design_fits: bool) -> dict[str, Any]:
    """티켓 §3 종합 규칙. gates 값은 "pass" / "fail" / "conditional"(G1 conda)."""
    # "fail"과 "not_run"(실행하지 못함)은 통과가 아니다
    failed = [g for g in GATES if gates[g] not in ("pass", "conditional")]
    conditional = [g for g in GATES if gates[g] == "conditional"]
    if not failed and not conditional:
        verdict, rule = "Go", "G1~G6 모두 통과"
    elif not conditional and failed and set(failed) <= {"G3", "G5"} and reduced_design_fits:
        verdict, rule = "Conditional Go", "G3나 G5만 미달이고 축소 설계로 B_009 안"
    elif not failed and conditional == ["G1"]:
        verdict, rule = "Conditional Go", "G1이 조건부(conda)이고 나머지 통과"
    else:
        verdict, rule = "No-Go", "그 밖"
    return {"verdict": verdict, "rule": rule, "failed": failed, "conditional": conditional}


def reduced_designs(
    sizes: Mapping[str, Mapping[str, int]], *, s_bar: float, budget_days: float
) -> list[dict[str, Any]]:
    """축소 설계 표: 후보 수 x 칸당 Dev-B·Oracle 배율 x S0 반복 수(n_A는 그대로)."""
    rows = []
    for candidates in (10, 20, 40):
        for scale in (1.0, 0.5, 0.25):
            scaled = {
                t: {
                    "n_A": s["n_A"],
                    "n_B": max(1, math.ceil(s["n_B"] * scale)),
                    "n_O": max(1, math.ceil(s["n_O"] * scale)),
                }
                for t, s in sizes.items()
            }
            for repeats in (3, 1):
                e = g5_estimate(
                    scaled,
                    s_bar=s_bar,
                    candidates=candidates,
                    s0_repeats=repeats,
                    budget_days=budget_days,
                )
                rows.append(
                    {
                        "candidates": candidates,
                        "cell_scale": scale,
                        "s0_repeats": repeats,
                        "per_candidate": e["per_candidate"],
                        "E_009": e["E_009"],
                        "T_009_days": e["T_009_days"],
                        "fits": e["passes"],
                    }
                )
    return rows


# --- main: 기록에서 P6 요약을 만든다 ---------------------------------------------------------


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _outcomes(records: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for r in sorted(records, key=lambda r: (r["rel"], r["repeat"])):
        out.setdefault(r["rel"], []).append(bool(r["won"]))
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"fraction": f"{value.numerator}/{value.denominator}", "float": float(value)}
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    return value


def build_p6(root: Path, json_root: Path) -> dict[str, Any]:
    runs = root / "runs"
    records = root / "records"
    s0 = _read(runs / "p4a" / "episodes.jsonl") + _read(runs / "p4b" / "episodes.jsonl")
    p4 = json.loads((records / "p4_summary.json").read_text())
    pool = json.loads((records / "p1_pilot_pool.json").read_text())["games"]

    # G1, G2
    p0 = json.loads((records / "p0_environment.json").read_text())
    g1 = "pass" if p0["smoke_run"]["won"] and p0["install_route"] != "conda" else "fail"
    det = json.loads((records / "p1_determinism.json").read_text())["games"]
    g2 = "pass" if all(g["same_first"] and g["same_replay"] for g in det) else "fail"

    # G3: P4 d_c, 남는 train은 과제 폴더 단위 제외 뒤(009 규칙)와 풀 30만 뺀 값 둘 다
    d = {t: p4["cells"][t]["d_c"] or 0.0 for t in SIX_TYPES}
    train = scan_split(json_root, "train")
    folder_available = available_after_folder_exclusion(train, [g["rel"] for g in pool])
    g3 = g3_preview(d, folder_available)
    g3_pool_only = g3_preview(d, {t: p4["g3_preview"]["available_train"][t] for t in SIX_TYPES})

    # G4: 점추정 [0.15, 0.85] 칸 수, 에피소드 단위와 게임 단위 CP
    rates = {t: p4["cells"][t]["success_rate"] for t in SIX_TYPES}
    g4 = g4_preview(rates)
    by_cell_games = {t: _outcomes([r for r in s0 if r["task_type"] == t]) for t in SIX_TYPES}
    g4_cells = {
        t: {
            "success_rate": rates[t],
            "episode_cp_95": p4["cells"][t]["clopper_pearson_95"],
            "game_majority": game_level_cp(by_cell_games[t]),
        }
        for t in SIX_TYPES
    }

    # G5
    s_bar = p4["runs"]["p4a"]["throughput"]["s_bar_excluding_load"]
    sizes = {
        t: {
            "n_A": g3_pool_only["n_A"],
            "n_B": g3_pool_only["n_B"][t],
            "n_O": g3_pool_only["n_O"][t],
        }
        for t in SIX_TYPES
    }
    g5 = g5_estimate(sizes, s_bar=s_bar, candidates=40, s0_repeats=3, budget_days=7)
    reduced = reduced_designs(sizes, s_bar=s_bar, budget_days=7)

    # G6
    harm_path = runs / "p5_harm" / "episodes.jsonl"
    g6: dict[str, Any] = {"available": harm_path.exists()}
    if harm_path.exists():
        harm = _read(harm_path)
        probe = {r["rel"]: bool(r["won"]) for r in harm}
        cells = {}
        for t in SIX_TYPES:
            games = sorted({r["rel"] for r in harm if r["task_type"] == t})
            s0_t = {g: by_cell_games[t][g] for g in games}
            cells[t] = g6_cell({g: probe[g] for g in games}, s0_t)
        g6 = {
            "available": True,
            "episodes": len(harm),
            "cells": cells,
            "judgement": g6_judgement(cells, target=p4["target_cell"]),
            "stops": stop_breakdown(harm),
        }
    gates = {
        "G1": g1,
        "G2": g2,
        # 고정 기준: (train - 파일럿 게임 수) >= n_min.
        # 과제 폴더 단위 제외는 고정 뒤 정한 009 규칙이라 판정에 쓰지 않고 함께 보고만 한다
        # (사후에 기준을 엄격하게 바꾸지 않는다)
        "G3": "pass" if g3_pool_only["passes_preview"] else "fail",
        "G4": "pass" if g4["passes_preview"] else "fail",
        "G5": "pass" if g5["passes"] else "fail",
        "G6": (
            "not_run" if not g6["available"] else "pass" if g6["judgement"]["passes"] else "fail"
        ),
    }
    verdict = overall_verdict(gates, reduced_design_fits=any(r["fits"] for r in reduced))

    # 진단 (exploratory diagnostic, not used for G1-G6)
    p4a_r0 = {r["key"]: r for r in _read(runs / "p4a" / "episodes.jsonl") if r["repeat"] == 0}
    diagnostics: dict[str, Any] = {"label": DIAG_LABEL}
    for name in ("diag_h1", "diag_h2"):
        path = runs / name / "episodes.jsonl"
        if path.exists():
            variant = _read(path)
            base = [p4a_r0[r["key"]] for r in variant]
            diagnostics[name] = diag_compare(base, variant)
        else:
            diagnostics[name] = None

    return {
        "gates": gates,
        "verdict": verdict,
        "G1": {"install_route": p0["install_route"], "smoke_run": p0["smoke_run"]},
        "G2": {
            "games": len(det),
            "same_first_and_replay": sum(1 for g in det if g["same_first"] and g["same_replay"]),
        },
        "G3": {"fixed_rule_pool_only": g3_pool_only, "caveat_folder_exclusion_009": g3},
        "G4": {**g4, "cells": g4_cells},
        "G5": g5,
        "reduced_designs": reduced,
        "G6": g6,
        "e_a_conclusion": p4["e_a"],
        "failure_breakdown": stop_breakdown(s0),
        "text_turns": text_turn_summary(s0),
        "diagnostics": diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alfworld-pilot-p6")
    parser.add_argument("--out", type=Path, default=Path("records/p6_summary.json"))
    args = parser.parse_args(argv)
    summary = build_p6(Path("."), data_root() / "json_2.1.1")
    args.out.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(json.dumps(_json_ready({k: summary[k] for k in ("gates", "verdict")}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
