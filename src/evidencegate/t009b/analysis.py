"""Ticket 009b 분석(사전 등록 `ticket-009b.txt` 4~7절, 6.6). 정의는 009의 `t009.rq9`를 그대로 쓴다.

    uv run python -m evidencegate.t009b.analysis

- 입력: 동결 파일(`POOL-009b.json`, 묶음 파일, 본 실행 계획)과 본 실행 기록(`runs/t009_dev_b/…`,
  `runs/t009_oracle/…`). 사전 등록·thresholds·POOL·묶음·계획의 sha256을 확인하고, 기록이 빠지면
  거부한다. h·δ는 `results/ticket-009/thresholds.json`에서 읽고 다시 계산하지 않는다.
- 게이트(Dev-B, 후보 반복 0 대 S0 반복 0): C2(`gates.gate_c2`), C4(`gates.gate_c4`, δ(420)),
  C5 = C4 그리고 칸 조건(`rq9.cell_condition`, h(70)), C5p = C2 그리고 칸 조건, C4*(`rq9.c4_star`,
  C4 채택을 평균·하한·candidate_id 순으로 |C5|개). CD와 ES 각각.
- Oracle(후보 반복 0 대 S0 반복 0~4 다수결): `oracle.estimate`, τ_cell(`oracle.tau_cells`),
  Severe Local Harm(`rq9.severe_local_harm`), Local Harm, Useful, 깨끗한 Useful(Useful 확정 참이고
  Severe 아님).
- RQ9-1~5(CD, `rq9`): 판정 문자열은 사전 등록 그대로(supported / direction_only / not_supported /
  undecidable). v1 Local Harm 라벨의 RQ9-1 표, 모든 Useful 기준 RQ9-2, C2의 RQ9-3은 기술용이다.
- 6.6 군집 민감도: 묶음마다 가장 이른 call의 풀 후보를 대표로 한 RQ9-1 민감도 판정, RQ9-3·RQ9-5
  기술용 값, 묶음별 2 x 2 기술표. 주 판정과 다르면 `differs_from_main`.
- 6.4 Budget Curve: N ∈ {24, 48, 96, 204}마다 Dev-B 칸 층화 비복원 재표본(seed 0 흐름 하나, 모든
  후보 공통)에서 C2·C4(δ(N))·C5(h(N/6))를 다시 판정하고, C4와 C5의 Severe·깨끗한 Useful 채택 수의
  차이를 요약한다. N = 420은 전체 표(재표본 1개)다.
- 지표 표(게이트 C2·C4·C5·C5p·C4*마다, `eval_metrics.gate_metrics`)에 Severe와 깨끗한 Useful의
  채택 비율을 더한다. 통제 후보는 사전 라벨 확인과 F만(게이트 없음).
- 결과 `results/ticket-009b/analysis.json`(+ `.sha256`)은 한 번만 쓴다(이미 있으면 거부).
  같은 입력이면 같은 바이트다(시각을 넣지 않음).
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
from evidencegate.budget_curve import summarize_values
from evidencegate.eval_constants import (
    BUDGET_NS,
    CONFIG_ID,
    N_BOOT,
    N_RESAMPLE,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
    SEED,
    TOOL_SPEC_ID,
)
from evidencegate.eval_data import (
    CELLS,
    CaseId,
    PairedTable,
    cell_of,
    check_run_conditions,
    majority,
    outcomes,
)
from evidencegate.eval_metrics import CandidateScore, Rate, gate_metrics
from evidencegate.exact import (
    mean_fraction,
    stratified_bootstrap_indices,
    stratified_subsample_indices,
)
from evidencegate.gates import gate_c2, gate_c4
from evidencegate.metrics import METRICS
from evidencegate.oracle import (
    OracleEstimate,
    check_control_label,
    estimate,
    fitness,
    harmful,
    local_harm,
    tau_cells,
    useful,
)
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009.rq9 import (
    Rq95Row,
    c4_star,
    cell_condition,
    rq9_1,
    rq9_2,
    rq9_3,
    rq9_5,
    severe_local_harm,
)
from evidencegate.t009.splits import split_cases
from evidencegate.t009b.pool_freeze import groups_ready
from evidencegate.t009b.preflight import check_preregistration_009b, check_thresholds
from evidencegate.t009b.run_plan import PLAN_PATH, check_plan_009b
from evidencegate.t009b.settings import POOL_GROUPS_PATH, POOL_PATH, THRESHOLDS_PATH

CD = "correct_disposition"
DEV_B_SPLIT, ORACLE_SPLIT = "t009_dev_b", "t009_oracle"
N_FULL = 420
GATES = ("C2", "C4", "C5", "C5p", "C4star")
OUT_PATH = Path("results/ticket-009b/analysis.json")
V1_ANALYSIS = Path("results/ticket-005/analysis.json")
V1_POOL = Path("skills/candidates/POOL.json")
# 사전 등록 9.1(009 계획 모의)의 귀무 라벨률. 6.3이 "함께 적는다"고 한 값이다.
NULL_LABEL_RATE_PLAN = {"local_harm": "0.167", "severe": "0.000475"}


@dataclass(frozen=True)
class Entry:
    name: str
    group: str  # proposer / control
    skill_id: str
    call_index: int | None


@dataclass(frozen=True)
class Thresholds:
    h: Mapping[str, Mapping[int, Fraction]]
    achieved: Mapping[str, Mapping[int, Fraction]]
    delta: Mapping[str, Mapping[int, Fraction]]


@dataclass(frozen=True)
class Inputs:
    records: Sequence[EpisodeRecord]
    pool: Sequence[Entry]
    groups: Mapping[str, Sequence[int]]
    thresholds: Thresholds
    control_priors: Mapping[str, str]
    marks: Mapping[str, Mapping[str, Any]]
    response_models: Mapping[str, Any]
    v1_oracle: Mapping[str, Fraction]
    input_hashes: Mapping[str, str]


def num(x: Fraction | None) -> dict[str, Any] | None:
    return None if x is None else {"exact": str(x), "float": float(x)}


def cell_name(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def load_thresholds(root: Path) -> Thresholds:
    data = json.loads((root / THRESHOLDS_PATH).read_text(encoding="utf-8"))["metrics"]
    return Thresholds(
        h={m: {int(n): Fraction(r["h"]) for n, r in v["h"].items()} for m, v in data.items()},
        achieved={
            m: {int(n): Fraction(r["achieved_false_rejection"]) for n, r in v["h"].items()}
            for m, v in data.items()
        },
        delta={
            m: {int(n): Fraction(r["delta"]) for n, r in v["delta"].items()}
            for m, v in data.items()
        },
    )


def _store_path(runs_dir: Path, split: str, skill_id: str) -> Path:
    return EpisodeStore.for_run(
        runs_dir,
        suite=split,
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=skill_id,
    ).path


def load_records(runs_dir: Path, pool: Sequence[Entry]) -> list[EpisodeRecord]:
    wanted = [(DEV_B_SPLIT, S0_SKILL_ID), (ORACLE_SPLIT, S0_SKILL_ID)]
    for e in pool:
        if e.group == "proposer":
            wanted.append((DEV_B_SPLIT, e.skill_id))
        wanted.append((ORACLE_SPLIT, e.skill_id))
    records: list[EpisodeRecord] = []
    for split, skill in wanted:
        path = _store_path(runs_dir, split, skill)
        if not path.exists():
            raise FileNotFoundError(f"no records at {path}; run the main plan first")
        records += EpisodeStore(path, repair=False).records()
    return records


# 표 ---------------------------------------------------------------------------------


def _table(
    split: str, metric: str, cand: Mapping[CaseId, int], base: Mapping[CaseId, int]
) -> PairedTable:
    cases = tuple(cand)
    return PairedTable(
        split=split,
        metric=metric,
        cases=cases,
        cells=tuple(cell_of(c) for c in cases),
        candidate=tuple(cand[c] for c in cases),
        baseline=tuple(base[c] for c in cases),
    )


def _outcomes(
    records: Sequence[EpisodeRecord], skill: str, split: str, metric: str, repeat: int
) -> dict[CaseId, int]:
    return outcomes(
        records, skill_id=skill, split=split, metric=metric, repeat=repeat, cases=split_cases(split)
    )


def _cell_means(table: PairedTable) -> list[Fraction]:
    return [mean_fraction([table.deltas[i] for i in g]) for g in table.cell_indices()]


def _sub(table: PairedTable, positions: Sequence[int]) -> PairedTable:
    return PairedTable(
        split=table.split,
        metric=table.metric,
        cases=tuple(table.cases[p] for p in positions),
        cells=tuple(table.cells[p] for p in positions),
        candidate=tuple(table.candidate[p] for p in positions),
        baseline=tuple(table.baseline[p] for p in positions),
    )


@dataclass(frozen=True)
class Scored:
    """한 proposer 후보·한 metric의 게이트와 라벨."""

    name: str
    call_index: int
    dev_b: PairedTable
    gates: Mapping[str, bool]
    mean: Fraction
    lower: Fraction
    cells: Sequence[Fraction]
    oracle: OracleEstimate
    severe: bool
    clean_useful: bool


def _score(
    e: Entry, metric: str, s0_dev: Mapping[CaseId, int], base: Mapping[CaseId, int],
    tau: Mapping[tuple[str, str], Fraction], records: Sequence[EpisodeRecord], t: Thresholds,
) -> Scored:  # fmt: skip
    dev = _table("dev_b", metric, _outcomes(records, e.skill_id, DEV_B_SPLIT, metric, 0), s0_dev)
    c2, c4 = gate_c2(dev), gate_c4(dev, t.delta[metric][N_FULL])
    cells = _cell_means(dev)
    cond = cell_condition(cells, t.h[metric][N_FULL // len(CELLS)])
    ora = _table("oracle", metric, _outcomes(records, e.skill_id, ORACLE_SPLIT, metric, 0), base)
    est = _estimate(ora, tau)
    severe = severe_local_harm(ora, tau)
    assert c4.lower is not None
    return Scored(
        name=e.name,
        call_index=int(e.call_index if e.call_index is not None else -1),
        dev_b=dev,
        gates={
            "C2": c2.accepted,
            "C4": c4.accepted,
            "C5": c4.accepted and cond,
            "C5p": c2.accepted and cond,
        },
        mean=c4.mean_delta,
        lower=c4.lower,
        cells=cells,
        oracle=est,
        severe=severe,
        clean_useful=useful(est) is True and not severe,
    )


def _estimate(table: PairedTable, tau: Mapping[tuple[str, str], Fraction]) -> OracleEstimate:
    return estimate(table, tau)


def _rows95(scored: Sequence[Scored]) -> list[Rq95Row]:
    return [
        Rq95Row(s.name, s.mean, s.lower, s.gates["C5"], s.severe, s.clean_useful)
        for s in scored
        if s.gates["C4"]
    ]


def _rq95_out(scored: Sequence[Scored]) -> dict[str, Any]:
    rows = _rows95(scored)
    m = sum(1 for r in rows if r.c5)
    out = _plain(rq9_5(rows))
    out["c4_star"] = [r.candidate_id for r in c4_star(rows, m)]
    return out


def _plain(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (num(v) if isinstance(v, Fraction) else v) for k, v in d.items()}


def _rq(scored: Sequence[Scored]) -> dict[str, Any]:
    c4 = [s for s in scored if s.gates["C4"]]
    c2 = [s for s in scored if s.gates["C2"]]
    filtered = any(not s.gates["C5"] for s in c4)
    useful_c4 = [s for s in c4 if useful(s.oracle) is True]
    return {
        "rq9_1": _plain(rq9_1([(s.gates["C5"], s.severe) for s in c4])),
        "rq9_1_local_harm": _plain(rq9_1([(s.gates["C5"], local_harm(s.oracle)) for s in c4])),
        "rq9_2": _plain(
            rq9_2(
                sum(1 for s in c4 if s.clean_useful),
                sum(1 for s in c4 if s.clean_useful and not s.gates["C5"]),
                any_filtered=filtered,
            )
        ),
        "rq9_2_all_useful": _plain(
            rq9_2(
                len(useful_c4),
                sum(1 for s in useful_c4 if not s.gates["C5"]),
                any_filtered=filtered,
            )
        ),
        "rq9_3": _plain(rq9_3(severe_accepted=sum(s.severe for s in c4), accepted=len(c4))),
        "rq9_3_c2": _plain(rq9_3(severe_accepted=sum(s.severe for s in c2), accepted=len(c2))),
        "rq9_5": _rq95_out(scored),
    }


def _gate_sets(scored: Sequence[Scored]) -> dict[str, set[str]]:
    sets = {g: {s.name for s in scored if s.gates[g]} for g in ("C2", "C4", "C5", "C5p")}
    rows = _rows95(scored)
    sets["C4star"] = {r.candidate_id for r in c4_star(rows, sum(1 for r in rows if r.c5))}
    return sets


def _rate(r: Rate) -> dict[str, int]:
    return {"successes": r.successes, "n": r.n, "excluded": r.excluded}


def _metrics(by_metric: Mapping[str, Sequence[Scored]]) -> dict[str, Any]:
    out: dict[str, Any] = {g: {} for g in GATES}
    for metric, scored in by_metric.items():
        sets = _gate_sets(scored)
        for gate in GATES:
            scores = [
                CandidateScore(s.name, {("C4", metric): s.name in sets[gate]}, {metric: s.oracle})
                for s in scored
            ]
            gm = gate_metrics(scores, "C4", metric)
            accepted = [s for s in scored if s.name in sets[gate]]
            out[gate][metric] = {
                "accepted": list(gm.accepted),
                "harmful": _rate(gm.harmful),
                "unsupported": _rate(gm.unsupported),
                "useful": _rate(gm.useful),
                "gain": num(gm.gain.mean),
                "false_rejection": _rate(gm.false_rejection),
                "local_harm": _rate(gm.local_harm),
                "severe": {
                    "successes": sum(s.severe for s in accepted),
                    "n": len(accepted),
                    "excluded": 0,
                },
                "clean_useful": {
                    "successes": sum(s.clean_useful for s in accepted),
                    "n": len(accepted),
                    "excluded": 0,
                },
            }
    return out


def _cluster(
    scored: Sequence[Scored], groups: Mapping[str, Sequence[int]], main_verdict: str
) -> dict[str, Any]:
    by_call = {s.call_index: s for s in scored}
    reps = {g: by_call[min(calls)] for g, calls in sorted(groups.items())}
    rep_scored = list(reps.values())
    c4_reps = [s for s in rep_scored if s.gates["C4"]]
    sens = _plain(rq9_1([(s.gates["C5"], s.severe) for s in c4_reps]))
    tables = {}
    for g, calls in sorted(groups.items()):
        members = [by_call[i] for i in calls]
        c4m = [s for s in members if s.gates["C4"]]
        tables[g] = {
            "members": len(members),
            "c4": len(c4m),
            "filtered_severe": sum(1 for s in c4m if not s.gates["C5"] and s.severe),
            "filtered_not": sum(1 for s in c4m if not s.gates["C5"] and not s.severe),
            "kept_severe": sum(1 for s in c4m if s.gates["C5"] and s.severe),
            "kept_not": sum(1 for s in c4m if s.gates["C5"] and not s.severe),
        }
    return {
        "representatives": {g: s.name for g, s in reps.items()},
        "rq9_1": sens,
        "rq9_3": _plain(
            rq9_3(severe_accepted=sum(s.severe for s in c4_reps), accepted=len(c4_reps))
        ),
        "rq9_5": _rq95_out(rep_scored),
        "group_tables": tables,
        "differs_from_main": sens["verdict"] != main_verdict,
    }


def _budget(
    by_metric: Mapping[str, Sequence[Scored]], t: Thresholds, n_resample: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    any_table = next(iter(by_metric.values()))[0].dev_b
    groups = any_table.cell_indices()
    for n in BUDGET_NS:
        per_cell = n // len(CELLS)
        picks = stratified_subsample_indices(
            [len(g) for g in groups], [per_cell] * len(CELLS), n_resample, SEED, replace=False
        )
        positions = [[groups[c][int(i)] for c, cell in enumerate(r) for i in cell] for r in picks]
        blocks = stratified_bootstrap_indices([per_cell] * len(CELLS), N_BOOT, SEED)
        out[str(n)] = {}
        for metric, scored in by_metric.items():
            decisions = []
            for pos in positions:
                d = {}
                for s in scored:
                    sub = _sub(s.dev_b, pos)
                    c4 = gate_c4(sub, t.delta[metric][n], blocks=blocks).accepted
                    cond = cell_condition(_cell_means(sub), t.h[metric][per_cell])
                    d[s.name] = {"C2": gate_c2(sub).accepted, "C4": c4, "C5": c4 and cond}
                decisions.append(d)
            out[str(n)][metric] = _budget_point(decisions, scored)
    out[str(N_FULL)] = {
        metric: _budget_point([{s.name: dict(s.gates) for s in scored}], scored)
        for metric, scored in by_metric.items()
    }
    return out


def _budget_point(
    decisions: Sequence[Mapping[str, Mapping[str, bool]]], scored: Sequence[Scored]
) -> dict[str, Any]:
    def summary(values: Sequence[Fraction | None]) -> dict[str, Any]:
        s = summarize_values(values)
        return {
            "mean": num(s.mean),
            "low": num(s.low),
            "high": num(s.high),
            "undefined": s.undefined,
        }

    def count(d: Mapping[str, Mapping[str, bool]], gate: str, label: str) -> Fraction:
        return Fraction(sum(1 for s in scored if d[s.name][gate] and getattr(s, label)))

    n = len(scored)
    return {
        "n_resample": len(decisions),
        "acceptance": {
            g: summary([Fraction(sum(d[s.name][g] for s in scored), n) for d in decisions])
            for g in ("C2", "C4", "C5")
        },
        "severe_c4": summary([count(d, "C4", "severe") for d in decisions]),
        "severe_c5": summary([count(d, "C5", "severe") for d in decisions]),
        "severe_diff": summary(
            [count(d, "C4", "severe") - count(d, "C5", "severe") for d in decisions]
        ),
        "useful_diff": summary(
            [count(d, "C4", "clean_useful") - count(d, "C5", "clean_useful") for d in decisions]
        ),
    }


def analyze(inputs: Inputs, n_resample: int = N_RESAMPLE) -> dict[str, Any]:
    records, t = inputs.records, inputs.thresholds
    check_run_conditions(
        records, policy_id=POLICY_ID, config_id=CONFIG_ID, tool_spec_id=TOOL_SPEC_ID
    )
    proposers = [e for e in inputs.pool if e.group == "proposer"]
    controls = [e for e in inputs.pool if e.group == "control"]
    by_metric: dict[str, list[Scored]] = {}
    tau: dict[str, dict[tuple[str, str], Fraction]] = {}
    control_est: dict[str, dict[str, OracleEstimate]] = {}
    for metric in METRICS:
        s0_dev = _outcomes(records, S0_SKILL_ID, DEV_B_SPLIT, metric, 0)
        reps = {
            r: _outcomes(records, S0_SKILL_ID, ORACLE_SPLIT, metric, r) for r in S0_ORACLE_REPEATS
        }
        base = majority(reps)
        tau[metric] = tau_cells(reps)
        by_metric[metric] = [
            _score(e, metric, s0_dev, base, tau[metric], records, t) for e in proposers
        ]
        control_est[metric] = {
            e.name: _estimate(
                _table(
                    "oracle", metric, _outcomes(records, e.skill_id, ORACLE_SPLIT, metric, 0), base
                ),
                tau[metric],
            )
            for e in controls
        }
    cd = by_metric[CD]
    rq = _rq(cd)
    candidates = {
        s.name: {
            metric: {
                "call_index": s.call_index,
                "gates": dict(next(x for x in by_metric[metric] if x.name == s.name).gates),
                **_candidate_detail(next(x for x in by_metric[metric] if x.name == s.name)),
            }
            for metric in METRICS
        }
        for s in cd
    }
    return {
        "inputs": dict(sorted(inputs.input_hashes.items())),
        "candidates": candidates,
        "rq": rq,
        "cluster_sensitivity": _cluster(cd, inputs.groups, rq["rq9_1"]["verdict"]),
        "metrics": _metrics(by_metric),
        "budget_curve": _budget(by_metric, t, n_resample),
        "h": {
            m: {
                str(n): {"h": num(t.h[m][n]), "achieved_false_rejection": num(t.achieved[m][n])}
                for n in sorted(t.h[m])
            }
            for m in METRICS
        },
        "delta": {m: {str(n): num(t.delta[m][n]) for n in sorted(t.delta[m])} for m in METRICS},
        "tau": {m: {cell_name(c): num(v) for c, v in tau[m].items()} for m in METRICS},
        "controls": {
            e.name: {
                m: {
                    "prior": inputs.control_priors.get(e.name),
                    "check": check_control_label(
                        inputs.control_priors[e.name], control_est[m][e.name]
                    )
                    if e.name in inputs.control_priors
                    else None,
                    "delta": num(control_est[m][e.name].delta),
                    "cells": {
                        cell_name(c): num(v) for c, v in control_est[m][e.name].cell_deltas.items()
                    },
                }
                for m in METRICS
            }
            for e in controls
        },
        "fitness": {m: _fitness(control_est[m]) for m in METRICS},
        "null_label_rate_plan": NULL_LABEL_RATE_PLAN,
        "response_models": dict(inputs.response_models),
        "duplicate_marks": _marks(cd, inputs.marks, inputs.v1_oracle),
    }


def _candidate_detail(s: Scored) -> dict[str, Any]:
    e = s.oracle
    return {
        "dev_b_mean": num(s.mean),
        "c4_lower": num(s.lower),
        "dev_b_cells": {cell_name(c): num(v) for c, v in zip(CELLS, s.cells, strict=True)},
        "oracle_delta": num(e.delta),
        "oracle_interval": [num(e.interval[0]), num(e.interval[1])],
        "oracle_cells": {cell_name(c): num(v) for c, v in e.cell_deltas.items()},
        "useful": useful(e),
        "harmful": harmful(e),
        "local_harm": local_harm(e),
        "severe": s.severe,
        "clean_useful": s.clean_useful,
    }


def _fitness(controls: Mapping[str, OracleEstimate]) -> dict[str, Any]:
    if not controls:
        return {"fit": None}
    f = fitness(controls)
    return {
        "fit": f.fit,
        "sign_met": f.sign_met,
        "size_met": f.size_met,
        "sign_hits": [list(x) for x in f.sign_hits],
        "size_hits": [list(x) for x in f.size_hits],
    }


def _marks(
    scored: Sequence[Scored],
    marks: Mapping[str, Mapping[str, Any]],
    v1_oracle: Mapping[str, Fraction],
) -> list[dict[str, Any]]:
    rows = []
    for s in scored:
        m = marks.get(s.name, {})
        v1, t009 = list(m.get("v1_duplicates", [])), list(m.get("t009_duplicates", []))
        if not v1 and not t009:
            continue
        rows.append(
            {
                "name": s.name,
                "call_index": s.call_index,
                "v1_duplicates": v1,
                "t009_duplicates": t009,
                "gates_cd": dict(s.gates),
                "oracle_delta_cd": num(s.oracle.delta),
                "severe_cd": s.severe,
                "v1_oracle_delta_cd": {k: num(v1_oracle[k]) for k in v1 if k in v1_oracle},
            }
        )
    return rows


def analysis_bytes(result: Mapping[str, Any]) -> bytes:
    return (json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_once(out: Path, result: Mapping[str, Any]) -> str:
    if out.exists():
        raise FileExistsError(f"{out} already exists; the analysis is written once")
    data = analysis_bytes(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    return digest


# 동결 입력 읽기 ---------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root: Path, runs_dir: Path) -> Inputs:
    problems = [
        c.detail for c in (check_preregistration_009b(root), check_thresholds(root)) if not c.ok
    ]
    pool_path = root / POOL_PATH
    pool_sha = pool_path.with_name(pool_path.name + ".sha256")
    if (
        not pool_path.exists()
        or not pool_sha.exists()
        or pool_sha.read_text().split()[0] != _sha(pool_path)
    ):
        problems.append(f"{pool_path} is missing or differs from its .sha256")
    problems += groups_ready(root)
    plan = check_plan_009b(root / PLAN_PATH, root)
    if not plan.ok:
        problems.append(plan.detail)
    if problems:
        raise ValueError("; ".join(problems))
    pool_data = json.loads(pool_path.read_text(encoding="utf-8"))
    pool = tuple(
        Entry(c["name"], c["group"], c["skill_id"], c.get("call_index"))
        for c in pool_data["candidates"]
    )
    groups_data = json.loads((root / POOL_GROUPS_PATH).read_text(encoding="utf-8"))
    v1_pool = json.loads((root / V1_POOL).read_text(encoding="utf-8"))
    v1 = json.loads((root / V1_ANALYSIS).read_text(encoding="utf-8"))["candidates"]
    return Inputs(
        records=load_records(runs_dir, pool),
        pool=pool,
        groups={g: v["calls"] for g, v in groups_data["groups"].items()},
        thresholds=load_thresholds(root),
        control_priors={c["name"]: c["subset"] for c in v1_pool["controls"]},
        marks={
            c["name"]: {
                "v1_duplicates": c["v1_duplicates"],
                "t009_duplicates": c["t009_duplicates"],
            }
            for c in pool_data["candidates"]
            if c["group"] == "proposer"
        },
        response_models=pool_data["response_models"],
        v1_oracle={
            k: Fraction(v["oracle"][CD]["delta"]["exact"]) for k, v in v1.items() if "oracle" in v
        },
        input_hashes={
            "pool": _sha(pool_path),
            "groups": _sha(root / POOL_GROUPS_PATH),
            "plan": _sha(root / PLAN_PATH),
            "thresholds": _sha(root / THRESHOLDS_PATH),
            "preregistration": _sha(root / "preregistration/ticket-009b.txt"),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009b-analysis")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root: Path = args.root
    out = root / OUT_PATH
    if out.exists():
        print(f"error: {out} already exists; the analysis is written once", file=sys.stderr)
        return 1
    try:
        inputs = load_inputs(root, args.runs_dir or root / "runs")
        result = analyze(inputs)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    digest = write_once(out, result)
    print(f"analysis {out}: sha256 {digest}")
    print(
        f"RQ9-1 {result['rq']['rq9_1']['verdict']}, "
        f"sensitivity {result['cluster_sensitivity']['rq9_1']['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
