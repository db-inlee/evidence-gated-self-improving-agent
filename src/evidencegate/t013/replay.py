"""Ticket 013 4단계: 009b 탐색 재생(Exploratory). 동결 매개변수 그대로, 새 기준 없음.

    uv run python -m evidencegate.t013.replay       # results/ticket-013/replay-009b.json 한 번 쓰기
    uv run python -m evidencegate.t013.replay --c6  # C6-IUT 재생(replay-009b-c6.json, 한 번)

- 009b Dev-B 짝 기록(후보 40 x 420, 반복 0, S0 반복 0과 같은 seed)에 SAGE λ1·λ2, 칸별 SAGE,
  신용형 게이트를 적용한다(CD). 매개변수는 동결 계획 값(`sim_core` 상수, 테스트로 대조)이다.
- C4·C5는 커밋된 `results/ticket-009b/analysis.json`의 판정을 쓴다. 같은 기록에서 기존 함수로 다시
  계산한 C4·C5가 기록과 하나라도 다르면 멈춘다(입력 확인).
- Oracle 라벨(Severe, Useful, Clean Useful)은 커밋된 analysis.json에서만 읽는다. 표는 전체 풀(40)과
  6.6 대표 풀(묶음마다 대표 하나)이다.
- 기록 파일 sha256은 커밋된 실행 요약(`run_summary_009b.json`)과 대조한다. 결과는 한 번만 쓴다.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import S0_SKILL_ID
from evidencegate.eval_data import CELLS, CaseId
from evidencegate.gates import gate_c4
from evidencegate.t009.rq9 import cell_condition
from evidencegate.t009b.settings import POOL_PATH
from evidencegate.t013 import sim
from evidencegate.t013 import sim_core as sc

OUT = Path("results/ticket-013/replay-009b.json")
OUT_C6 = Path("results/ticket-013/replay-009b-c6.json")
C6_MARGINS = (Fraction(1, 20), Fraction(1, 10))
C6_HEADER = (
    "Exploratory replay of C6-IUT on the 009b Dev-B pairs (Tango score primary, exact "
    "conditional secondary, frozen alpha). This result does not change m_c or n_c of the "
    "confirmatory study D: m_c = 0.05 was fixed by David on 2026-11-26 before this replay, and "
    "n_c comes from the frozen Ticket 013 simulation."
)
ANALYSIS = Path("results/ticket-009b/analysis.json")
REPLAY_GATES = ("SAGE_l1", "SAGE_l2", "SAGE_cell_l1", "SAGE_cell_l2", "credit")
CD = "correct_disposition"


def parameters() -> dict[str, Any]:
    return {
        "SAGE": {"lambda": list(sc.SAGE_LAMBDAS), "tau": sc.SAGE_TAU, "alpha": float(sc.ALPHA)},
        "credit_type": {"epsilon": sc.CREDIT_EPS},
        "plan_sha256": sim.PLAN_SHA256,
        "addendum_sha256": sim.ADDENDUM_SHA256,
        "note": "our one-round implementations; tau = 0.05 is our value (the SAGE paper tuned it)",
    }


def cell_counts(cand: Mapping[CaseId, int], s0: Mapping[CaseId, int]) -> list[dict[str, int]]:
    """칸마다 w(후보 성공·S0 실패), l(후보 실패·S0 성공), n_s(S0 성공), n."""
    if set(cand) != set(s0):
        raise ValueError("candidate and S0 records hold different cases")
    out = []
    for cell in CELLS:
        cases = [c for c in s0 if c[:2] == cell]
        out.append(
            {
                "w": sum(1 for c in cases if cand[c] == 1 and s0[c] == 0),
                "l": sum(1 for c in cases if cand[c] == 0 and s0[c] == 1),
                "n_s": sum(s0[c] for c in cases),
                "n": len(cases),
            }
        )
    return out


def replay_gates(counts: Sequence[Mapping[str, int]]) -> dict[str, bool]:
    """동결 매개변수의 SAGE λ1·λ2, 칸별 SAGE, 신용형(sim의 판정 함수 그대로)."""
    w = sum(c["w"] for c in counts)
    loss = sum(c["l"] for c in counts)
    n_s = sum(c["n_s"] for c in counts)
    n = sum(c["n"] for c in counts)
    cells_ok = sim.sage_cells_ok(
        [c["l"] for c in counts], [c["n_s"] for c in counts], tau=sc.SAGE_TAU
    )
    out: dict[str, bool] = {}
    for lam in sc.SAGE_LAMBDAS:
        sage = sim.sage_accept(w, loss, n_s, lam=lam, tau=sc.SAGE_TAU, alpha=sc.ALPHA)
        out[f"SAGE_l{lam}"] = sage
        out[f"SAGE_cell_l{lam}"] = sage and cells_ok
    out["credit"] = sim.credit_accept(w, loss, n, n_s, eps=sc.CREDIT_EPS)
    return out


def tally(
    accepted: Mapping[str, bool], labels: Mapping[str, Mapping[str, Any]], names: Sequence[str]
) -> dict[str, Any]:
    """채택 수와 그중 Severe, Useful(확정), Clean Useful 수."""
    acc = [n for n in names if accepted[n]]
    return {
        "candidates": len(names),
        "accepted": len(acc),
        "severe": sum(1 for n in acc if labels[n]["severe"]),
        "useful": sum(1 for n in acc if labels[n]["useful"] is True),
        "clean_useful": sum(1 for n in acc if labels[n]["clean_useful"]),
        "accepted_names": acc,
    }


def check_recorded(
    recomputed: Mapping[str, Mapping[str, bool]], recorded: Mapping[str, Mapping[str, bool]]
) -> None:
    for name, gates in recorded.items():
        for gate in ("C4", "C5"):
            if recomputed[name][gate] != gates[gate]:
                raise ValueError(f"{name}: recomputed {gate} differs from the committed analysis")


@dataclass(frozen=True)
class Pairs:
    """009b Dev-B 짝 기록에서 읽은 값(두 재생이 같이 쓴다)."""

    names: list[str]
    representatives: list[str]
    labels: dict[str, dict[str, Any]]
    recorded: dict[str, dict[str, bool]]  # 커밋된 C4·C5
    counts: dict[str, list[dict[str, int]]]
    used: dict[str, str]


def load_pairs(root: Path) -> Pairs:
    """동결 파일과 기록 sha256을 확인하고, C4·C5를 다시 계산해 기록과 대조한 뒤 짝 수를 돌려준다."""
    from evidencegate.t009b import analysis as an
    from evidencegate.t013 import audit

    sim.load_addendum(root)
    if sim._sha(root / ANALYSIS) != audit.T009B_ANALYSIS_SHA256:
        raise ValueError(f"{ANALYSIS} differs from its pinned sha256")
    analysis = json.loads((root / ANALYSIS).read_text(encoding="utf-8"))
    pool = json.loads((root / POOL_PATH).read_text(encoding="utf-8"))["candidates"]
    proposers = sorted((c["name"], c["skill_id"]) for c in pool if c["group"] == "proposer")
    pins = audit.record_pins(root)
    used: dict[str, str] = {}
    runs = root / "runs"
    s0_records = audit._read(audit._store(runs, an.DEV_B_SPLIT, S0_SKILL_ID), root, pins, used)
    thresholds = an.load_thresholds(root)
    s0 = an._outcomes(s0_records, S0_SKILL_ID, an.DEV_B_SPLIT, CD, 0)
    labels, recorded, counts = {}, {}, {}
    recomputed = {}
    for name, skill in proposers:
        records = audit._read(audit._store(runs, an.DEV_B_SPLIT, skill), root, pins, used)
        cand = an._outcomes(records, skill, an.DEV_B_SPLIT, CD, 0)
        table = an._table("dev_b", CD, cand, s0)
        c4 = gate_c4(table, thresholds.delta[CD][an.N_FULL]).accepted
        cond = cell_condition(an._cell_means(table), thresholds.h[CD][an.N_FULL // len(CELLS)])
        recomputed[name] = {"C4": c4, "C5": c4 and cond}
        counts[name] = cell_counts(cand, s0)
        entry = analysis["candidates"][name][CD]
        labels[name] = {k: entry[k] for k in ("severe", "useful", "clean_useful")}
        recorded[name] = {"C4": entry["gates"]["C4"], "C5": entry["gates"]["C5"]}
    check_recorded(recomputed, recorded)
    names = [n for n, _ in proposers]
    reps = sorted(
        n for n in analysis["cluster_sensitivity"]["representatives"].values() if n in labels
    )
    return Pairs(names, reps, labels, recorded, counts, used)


def _inputs(pairs: Pairs) -> dict[str, Any]:
    from evidencegate.t013 import audit

    return {
        "analysis": {"path": str(ANALYSIS), "sha256": audit.T009B_ANALYSIS_SHA256},
        "records": dict(sorted(pairs.used.items())),
    }


def _labels_in_pool(pairs: Pairs) -> dict[str, dict[str, int]]:
    labels = pairs.labels
    return {
        label: {
            "severe": sum(1 for n in subset if labels[n]["severe"]),
            "useful": sum(1 for n in subset if labels[n]["useful"] is True),
            "clean_useful": sum(1 for n in subset if labels[n]["clean_useful"]),
        }
        for label, subset in (("all", pairs.names), ("representatives", pairs.representatives))
    }


def build_replay(root: Path) -> dict[str, Any]:
    pairs = load_pairs(root)
    names, labels = pairs.names, pairs.labels
    gates = {n: {**pairs.recorded[n], **replay_gates(pairs.counts[n])} for n in names}
    pools = {
        label: {
            gate: tally({n: gates[n][gate] for n in names}, labels, subset)
            for gate in ("C4", "C5", *REPLAY_GATES)
        }
        for label, subset in (("all", names), ("representatives", pairs.representatives))
    }
    return {
        "status": "Exploratory",
        "what": "009b Dev-B pairs (CD, repeat 0, same seed as S0 repeat 0) under the frozen Ticket "
        "013 parameters; Oracle labels from the committed 009b analysis; no parameter or "
        "criterion was changed",
        "parameters": parameters(),
        "metric": "correct_disposition",
        "inputs": _inputs(pairs),
        "consistency": {"c4_c5_recomputed_equal_record": True},
        "labels_in_pool": _labels_in_pool(pairs),
        "pools": pools,
        "candidates": {
            n: {
                "gates": gates[n],
                "labels": labels[n],
                "cells": {sc.CELL_NAMES[i]: c for i, c in enumerate(pairs.counts[n])},
            }
            for n in names
        },
    }


def c6_accept(counts: Sequence[Mapping[str, int]], *, c4: bool, m: Fraction, bound: str) -> bool:
    """C4 그리고 모든 칸에서 한쪽 하한 > -m(Tango 또는 정확 조건부, 동결 alpha)."""
    lower = sim.tango_lower if bound == "tango" else sim.exact_conditional_lower
    return c4 and all(lower(c["w"], c["l"], c["n"], sc.ALPHA) > -float(m) for c in counts)


def build_replay_c6(root: Path) -> dict[str, Any]:
    pairs = load_pairs(root)
    names, labels = pairs.names, pairs.labels
    gates: dict[str, dict[str, bool]] = {}
    for n in names:
        row = dict(pairs.recorded[n])
        for m in C6_MARGINS:
            for bound in ("tango", "exact_conditional"):
                key = f"C6_IUT_{bound}_m{m}"
                row[key] = c6_accept(pairs.counts[n], c4=pairs.recorded[n]["C4"], m=m, bound=bound)
        gates[n] = row
    gate_keys = list(gates[names[0]])
    pools = {
        label: {g: tally({n: gates[n][g] for n in names}, labels, subset) for g in gate_keys}
        for label, subset in (("all", names), ("representatives", pairs.representatives))
    }
    lowers = {
        n: {
            sc.CELL_NAMES[i]: {
                "tango": sim.tango_lower(c["w"], c["l"], c["n"], sc.ALPHA),
                "exact_conditional": sim.exact_conditional_lower(c["w"], c["l"], c["n"], sc.ALPHA),
            }
            for i, c in enumerate(pairs.counts[n])
        }
        for n in names
    }
    return {
        "status": "Exploratory",
        "header": C6_HEADER,
        "primary_bound": "tango",
        "secondary_bound": "exact_conditional",
        "margins": [str(m) for m in C6_MARGINS],
        "alpha": str(sc.ALPHA),
        "parameters": {"plan_sha256": sim.PLAN_SHA256, "addendum_sha256": sim.ADDENDUM_SHA256},
        "metric": "correct_disposition",
        "rule": "C6-IUT = C4 (committed 009b decision) and, in every cell, one-sided lower bound "
        "of delta_c > -m_c",
        "inputs": _inputs(pairs),
        "consistency": {"c4_c5_recomputed_equal_record": True},
        "labels_in_pool": _labels_in_pool(pairs),
        "pools": pools,
        "candidates": {
            n: {"gates": gates[n], "labels": labels[n], "cell_lower_bounds": lowers[n]}
            for n in names
        },
    }


def _write_once(root: Path, rel: Path, data: dict[str, Any]) -> None:
    out = root / rel
    payload = (json.dumps(data, indent=1, sort_keys=True) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    out.with_name(out.name + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {out.name}\n"
    )


def write_replay_c6(root: Path) -> dict[str, Any]:
    if (root / OUT_C6).exists():
        raise FileExistsError(f"{OUT_C6} exists; it is written once")
    data = build_replay_c6(root)
    _write_once(root, OUT_C6, data)
    return data


def write_replay(root: Path) -> dict[str, Any]:
    out = root / OUT
    if out.exists():
        raise FileExistsError(f"{OUT} exists; it is written once")
    data = build_replay(root)
    payload = (json.dumps(data, indent=1, sort_keys=True) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    out.with_name(out.name + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {out.name}\n"
    )
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t013-replay")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--c6", action="store_true", help="the C6-IUT replay ([2'])")
    args = parser.parse_args(argv)
    try:
        data = write_replay_c6(args.root) if args.c6 else write_replay(args.root)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    rows = {
        g: (v["accepted"], v["severe"], v["useful"], v["clean_useful"])
        for g, v in data["pools"]["all"].items()
    }
    print(
        json.dumps(
            {
                "wrote": str(OUT_C6 if args.c6 else OUT),
                "all (accepted, severe, useful, clean)": rows,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
