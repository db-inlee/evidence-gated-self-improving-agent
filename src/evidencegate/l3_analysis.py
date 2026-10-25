"""Ticket 006 L3 분석 (§4, RQ-L3-0~3, §10 해석). 본 실행이 끝난 뒤 한 번만 돌린다.

    uv run python -m evidencegate.l3_analysis --runs-dir runs \\
        --noise runs/calibration/<policy>__7c66ee5bf13c/e9b4996f7529/bf65e569ae37.jsonl \\
        --out-dir results/ticket-006 --analysis-commit <commit> --plan-sha256 <hash>

순서:
1. 사전 등록 해시 확인(7.8, Ticket 006).
2. **Ticket 005 결과 재현 확인:** 같은 레코드로 Ticket 005 분석을 다시 계산해
   `results/ticket-005/analysis.json`과 바이트까지 같아야 한다. 다르면 멈춘다. 그래서 아래의 게이트
   결정(본 판정, Budget Curve 재표본 판정, C1 민감도 판정)은 Ticket 005와 같다(고정 결정 3).
3. 실행 조건 확인(L1/L2와 L3 레코드, 노이즈 기준이 한 조건).
4. τ_cell: L1/L2 6칸(Ticket 005), L3 family 3칸, 보조로 L1/L2를 family로 묶은 3칸.
5. 후보마다 L3 oracle Δ(후보 반복 0 대 S0 반복 0~4 다수결, 3칸 층화 bootstrap 95% 양측).
6. 지표 6개를 Ticket 005 게이트 결정 x L3 oracle로 다시 계산하고, L1/L2 값과 나란히 둔다.
7. Local Harm: L3 3칸(주)과 L1/L2 family 3칸(보조).
8. RQ-L3-0(조작 확인), RQ-L3-1(Harmful 이동), RQ-L3-2(판정 범주 보존), RQ-L3-3(라벨 전이).

Oracle 적합성(F)은 L3에 적용하지 않는다(§4). 판정은 유리수로 하고, 결과 JSON은 입력이 같으면
바이트까지 같다(시각을 넣지 않는다).
"""

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from evidencegate.analysis import (
    CD,
    ES,
    GROUPS,
    PRIMARY_METRIC,
    REPO_ROOT,
    PoolSpec,
    _cell,
    _frac,
    _metrics_json,
    _rate_json,
    _rq,
    _summary_json,
    _verdict_json,
    check_preregistration,
    load_pool,
    load_pool_spec,
    load_records,
    run_analysis,
)
from evidencegate.budget_curve import (
    BudgetCandidate,
    budget_decisions,
    budget_points,
    sensitivity_decisions,
    sensitivity_rates,
)
from evidencegate.eval_constants import (
    BUDGET_NS,
    CONFIG_ID,
    EPSILON,
    GATE_REPEAT,
    L3_LEVEL,
    L3_ORACLE_SPLIT,
    N_BOOT,
    ORACLE_CI_QS,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    SEED,
    TICKET005_ANALYSIS_SHA256,
    TOOL_SPEC_ID,
)
from evidencegate.eval_data import (
    CELLS,
    ORACLE_SPLIT,
    CaseId,
    Cell,
    PairedTable,
    check_run_conditions,
    majority,
    outcomes,
    paired_gate,
    paired_oracle,
    repeat_outcomes,
)
from evidencegate.eval_metrics import GATE_NAMES, CandidateScore, GateMetrics, Rate, gate_metrics
from evidencegate.exact import (
    mean_fraction,
    null_diff_quantile_exact,
    percentile_exact,
    stratified_bootstrap_indices,
)
from evidencegate.gates import bootstrap_quantiles, decide_gates
from evidencegate.l3_bench import l3_split_cases
from evidencegate.metrics import METRICS
from evidencegate.oracle import (
    OracleEstimate,
    _judge,
    classify_effect,
    estimate,
    harmful,
    local_harm,
    subset_label,
    tau_cells,
    unsupported,
    useful,
)
from evidencegate.preflight import check_preregistration_l3
from evidencegate.rq import NOT_SUPPORTED, PARTIAL, SUPPORTED, Verdict, _compare, _gt, _verdict
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from opsworld.state import Family

SECTION10_VERSION = "2026-10-25 (Ticket 006 §10 items 1-12)"  # 결과 파일에 기록하는 해석 버전
TICKET005_RESULT = Path("results/ticket-005/analysis.json")

L3_CELLS: tuple[Cell, ...] = tuple(sorted((f.value, L3_LEVEL) for f in Family))
FAMILY_LEVEL = "L1+L2"  # 보조 지표: L1/L2를 family 단위로 다시 묶은 칸
FAMILY_CELLS: tuple[Cell, ...] = tuple((c[0], FAMILY_LEVEL) for c in L3_CELLS)
RQ_L3_1_GATES = ("C1", "C2", "C4")
RQ_L3_2_SENTENCES = ("RQ2_first", "RQ2_second", "RQ3_first", "RQ3_trend")
INDICATORS = ("harmful", "unsupported", "useful", "gain", "false_rejection", "local_harm")
CONFIRMED, NOT_CONFIRMED, UNDECIDABLE = "confirmed", "not_confirmed", "undecidable"


def l3_split() -> tuple[CaseId, ...]:
    return l3_split_cases(L3_ORACLE_SPLIT)


def ticket005_bytes(result: Mapping[str, Any]) -> bytes:
    """Ticket 005 분석이 결과 파일을 쓰는 방식 그대로(analysis.main)."""
    return (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


# ---------------------------------------------------------------------------
# L3 표와 추정
# ---------------------------------------------------------------------------


def l3_repeats(
    records: Sequence[EpisodeRecord], *, skill_id: str, metric: str, repeats: Sequence[int]
) -> dict[int, dict[CaseId, int]]:
    return {
        r: outcomes(
            records,
            skill_id=skill_id,
            split=L3_ORACLE_SPLIT,
            metric=metric,
            repeat=r,
            cases=l3_split(),
        )
        for r in repeats
    }


def l3_paired(
    records: Sequence[EpisodeRecord], *, candidate_skill_id: str, s0_skill_id: str, metric: str
) -> PairedTable:
    """후보 반복 0 대 S0 반복 0~4 다수결(§4)."""
    (cand,) = l3_repeats(
        records, skill_id=candidate_skill_id, metric=metric, repeats=(GATE_REPEAT,)
    ).values()
    base = majority(
        l3_repeats(records, skill_id=s0_skill_id, metric=metric, repeats=S0_ORACLE_REPEATS)
    )
    cases = tuple(cand)
    return PairedTable(
        split=L3_ORACLE_SPLIT,
        metric=metric,
        cases=cases,
        cells=tuple((c[0], c[1]) for c in cases),
        candidate=tuple(cand[c] for c in cases),
        baseline=tuple(base[c] for c in cases),
    )


def l3_estimate(table: PairedTable, tau: Mapping[Cell, Fraction]) -> OracleEstimate:
    """Ticket 005 `oracle.estimate`와 같고 칸만 L3 3칸이다(§4, §10 해석 5)."""
    if table.split != L3_ORACLE_SPLIT:
        raise ValueError(f"L3 oracle scoring uses {L3_ORACLE_SPLIT}, got {table.split}")
    low, high = bootstrap_quantiles(table, ORACLE_CI_QS, cells=L3_CELLS)
    groups = table.cell_indices(L3_CELLS)
    cells = {
        cell: mean_fraction([table.deltas[i] for i in g])
        for cell, g in zip(L3_CELLS, groups, strict=True)
    }
    return OracleEstimate(table.metric, mean_fraction(table.deltas), (low, high), cells, dict(tau))


def _family_key(case: CaseId) -> Cell:
    return case[0], FAMILY_LEVEL


def family_estimate(
    table: PairedTable, l12: OracleEstimate, tau: Mapping[Cell, Fraction]
) -> OracleEstimate:
    """보조 지표용(§4, §10 해석 6): L1/L2 oracle 짝 표를 family 칸(L1+L2 100개)으로 다시 묶는다.
    전체 Δ와 구간은 L1/L2 추정 그대로이고, 칸 Δ와 τ만 family 단위다."""
    cells = {
        cell: mean_fraction(
            [d for c, d in zip(table.cases, table.deltas, strict=True) if c[0] == cell[0]]
        )
        for cell in FAMILY_CELLS
    }
    return OracleEstimate(l12.metric, l12.delta, l12.interval, cells, dict(tau))


# ---------------------------------------------------------------------------
# RQ-L3-0: 조작 확인
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManipulationSample:
    """칸마다 케이스의 (CD, ES) S0 다수결 값. 칸 순서가 층화 순서다."""

    groups: tuple[tuple[tuple[int, int], ...], ...]
    n: int

    def __post_init__(self) -> None:
        if sum(len(g) for g in self.groups) != self.n or any(not g for g in self.groups):
            raise ValueError("every cell needs cases and n must be the total")


def manipulation_check(
    l12: ManipulationSample, l3: ManipulationSample, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    """Δ₀ = CD(L3) - CD(L1/L2) < -ε, g(L3) - g(L1/L2) > ε(g = CD - ES).

    두 표본은 짝이 없어 각자의 원래 설계 칸으로 층화한 독립 bootstrap을 한다(§10 해석 1). 재표본
    색인은 칸 크기 목록 [L1/L2 칸..., L3 칸...]으로 seed 흐름 하나에서 뽑고, CD와 g가 같은 재표본을
    쓴다(§10 해석 4). 구간은 2.5·97.5 백분위수, 판정은 양 끝 규칙(Ticket 005 §8-13)이다.
    """
    groups = [*l12.groups, *l3.groups]
    blocks = stratified_bootstrap_indices([len(g) for g in groups], n_boot=n_boot, seed=seed)
    totals = {key: np.zeros(n_boot, dtype=np.int64) for key in ("cd12", "es12", "cd3", "es3")}
    for i, (group, block) in enumerate(zip(groups, blocks, strict=True)):
        side = "12" if i < len(l12.groups) else "3"
        values = np.asarray(group, dtype=np.int64)
        totals["cd" + side] += values[:, 0][block].sum(axis=1)
        totals["es" + side] += values[:, 1][block].sum(axis=1)

    def rates(sample: ManipulationSample) -> tuple[Fraction, Fraction]:
        cases = [x for g in sample.groups for x in g]
        return (
            Fraction(sum(c for c, _ in cases), sample.n),
            Fraction(sum(e for _, e in cases), sample.n),
        )

    cd12, es12 = rates(l12)
    cd3, es3 = rates(l3)
    cd_draws = [
        Fraction(int(a), l3.n) - Fraction(int(b), l12.n)
        for a, b in zip(totals["cd3"], totals["cd12"], strict=True)
    ]
    gap_draws = [
        Fraction(int(a3 - b3), l3.n) - Fraction(int(a12 - b12), l12.n)
        for a3, b3, a12, b12 in zip(
            totals["cd3"], totals["es3"], totals["cd12"], totals["es12"], strict=True
        )
    ]
    cd = _side(cd12, cd3, cd_draws, lambda x: x < -EPSILON)
    gap = _side(cd12 - es12, cd3 - es3, gap_draws, lambda x: x > EPSILON)
    holds = (cd["holds"], gap["holds"])
    if True in holds:
        verdict = CONFIRMED
    elif holds == (False, False):
        verdict = NOT_CONFIRMED
    else:
        verdict = UNDECIDABLE
    return {"cd": cd, "gap": gap, "verdict": verdict}


def _side(l12: Fraction, l3: Fraction, draws: Sequence[Fraction], condition: Any) -> dict[str, Any]:
    low, high = (percentile_exact(draws, q) for q in ORACLE_CI_QS)
    return {
        "l12": _frac(l12),
        "l3": _frac(l3),
        "delta": _frac(l3 - l12),
        "interval": [_frac(low), _frac(high)],
        "holds": _judge((low, high), condition),
    }


def manipulation_samples(
    records: Sequence[EpisodeRecord], l3_records: Sequence[EpisodeRecord], s0_skill_id: str
) -> dict[str, tuple[ManipulationSample, ManipulationSample]]:
    """전체(L1/L2 6칸 대 L3 3칸)와 family별(L1·L2 2칸 대 L3 1칸) 표본(§10 해석 1)."""
    maj12 = {
        m: majority(
            repeat_outcomes(
                records,
                skill_id=s0_skill_id,
                split=ORACLE_SPLIT,
                metric=m,
                repeats=S0_ORACLE_REPEATS,
            )
        )
        for m in METRICS
    }
    maj3 = {
        m: majority(
            l3_repeats(l3_records, skill_id=s0_skill_id, metric=m, repeats=S0_ORACLE_REPEATS)
        )
        for m in METRICS
    }

    def sample(
        maj: Mapping[str, Mapping[CaseId, int]], cells: Sequence[Cell]
    ) -> ManipulationSample:
        groups = tuple(
            tuple((maj[CD][c], maj[ES][c]) for c in maj[CD] if (c[0], c[1]) == cell)
            for cell in cells
        )
        return ManipulationSample(groups, sum(len(g) for g in groups))

    out = {"all": (sample(maj12, CELLS), sample(maj3, L3_CELLS))}
    for family in sorted(f.value for f in Family):
        l12_cells = [c for c in CELLS if c[0] == family]
        l3_cells = [c for c in L3_CELLS if c[0] == family]
        out[family] = (sample(maj12, l12_cells), sample(maj3, l3_cells))
    return out


# ---------------------------------------------------------------------------
# RQ-L3-1~3
# ---------------------------------------------------------------------------


def rq_l3_1(l12: Mapping[str, Rate], l3: Mapping[str, Rate]) -> Verdict:
    """게이트 C1·C2·C4마다 Harmful(L3) > Harmful(L1/L2). 모두 성립 지지, 일부 부분 지지."""
    comparisons = [
        _compare(f"harmful {g}: L3 > L1/L2", l3[g].value, l12[g].value, _gt) for g in RQ_L3_1_GATES
    ]
    return _verdict(comparisons, (SUPPORTED, PARTIAL, NOT_SUPPORTED))


def rq_l3_2(
    l12: Mapping[str, Mapping[str, Mapping[str, Any]]],
    l3: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    sentences: Sequence[str] = RQ_L3_2_SENTENCES,
    metrics: Sequence[str] = METRICS,
) -> dict[str, dict[str, dict[str, str]]]:
    """문장 x metric마다 판정 범주가 같으면 보존(판정 불가끼리도 같은 범주, §10 해석 10)."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for s in sentences:
        out[s] = {}
        for m in metrics:
            a, b = l12[s][m]["verdict"], l3[s][m]["verdict"]
            out[s][m] = {"l12": a, "l3": b, "status": "preserved" if a == b else "changed"}
    return out


def rq_l3_3(l12: Mapping[str, str], l3: Mapping[str, str]) -> dict[str, Any]:
    """하위집합 라벨 다섯 개(uncertain 포함)를 그대로 비교한다(§10 해석 11)."""
    names = sorted(l12)
    if sorted(l3) != names:
        raise ValueError("label sets differ")
    same = sum(1 for n in names if l12[n] == l3[n])
    return {
        "same": same,
        "n": len(names),
        "rate": _frac(Fraction(same, len(names))) if names else None,
        "either_uncertain": sum(1 for n in names if "uncertain" in (l12[n], l3[n])),
        "changed": [{"name": n, "l12": l12[n], "l3": l3[n]} for n in names if l12[n] != l3[n]],
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def run_l3_analysis(
    records: Sequence[EpisodeRecord],
    l3_records: Sequence[EpisodeRecord],
    noise_records: Sequence[EpisodeRecord],
    pool: PoolSpec,
    *,
    ticket005: bytes,
    run_info: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    prereg_78 = check_preregistration(root)
    prereg_l3 = check_preregistration_l3(root)
    if not prereg_l3.ok:
        raise ValueError(prereg_l3.detail)

    # 2. Ticket 005 결과 재현(바이트까지)
    t005 = json.loads(ticket005)
    n_resample = int(t005["meta"]["n_resample"])
    again = run_analysis(
        records,
        noise_records,
        pool,
        run_info={
            "analysis_commit": t005["meta"]["analysis_commit"],
            "plan_sha256": t005["meta"]["plan_sha256"],
        },
        n_resample=n_resample,
        root=root,
    )
    if ticket005_bytes(again) != ticket005:
        raise ValueError(
            "the Ticket 005 result does not reproduce from these records; gate decisions are "
            "taken from Ticket 005 unchanged, so the L3 analysis stops"
        )

    # 3. 실행 조건
    check_run_conditions(
        [*records, *l3_records, *noise_records],
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
    )

    # 4~5. Ticket 005와 같은 계산으로 게이트 결정·L1/L2 추정, 그리고 L3 추정
    s0 = pool.s0_skill_id
    deltas = {
        m: {n: null_diff_quantile_exact(noise_records, m, n) for n in (210, *BUDGET_NS)}
        for m in METRICS
    }
    tau12 = {
        m: tau_cells(
            repeat_outcomes(
                records, skill_id=s0, split=ORACLE_SPLIT, metric=m, repeats=S0_ORACLE_REPEATS
            )
        )
        for m in METRICS
    }
    tau3 = {
        m: tau_cells(
            l3_repeats(l3_records, skill_id=s0, metric=m, repeats=S0_ORACLE_REPEATS),
            cells=L3_CELLS,
        )
        for m in METRICS
    }
    tau_family = {
        m: tau_cells(
            repeat_outcomes(
                records, skill_id=s0, split=ORACLE_SPLIT, metric=m, repeats=S0_ORACLE_REPEATS
            ),
            cells=FAMILY_CELLS,
            cell_key=_family_key,
        )
        for m in METRICS
    }

    scores12: dict[str, CandidateScore] = {}
    scores3: dict[str, CandidateScore] = {}
    scores_family: dict[str, CandidateScore] = {}
    budget12: list[BudgetCandidate] = []
    budget3: list[BudgetCandidate] = []
    candidates_out: dict[str, Any] = {}
    for spec in pool.candidates:
        dev_a, dev_b, est12, est3, est_family = {}, {}, {}, {}, {}
        accepted: dict[tuple[str, str], bool] = {}
        for m in METRICS:
            kw = {"candidate_skill_id": spec.skill_id, "s0_skill_id": s0, "metric": m}
            dev_a[m] = paired_gate(records, split="dev_a", **kw)
            dev_b[m] = paired_gate(records, split="dev_b", **kw)
            table12 = paired_oracle(records, **kw)
            est12[m] = estimate(table12, tau12[m])
            est3[m] = l3_estimate(l3_paired(l3_records, **kw), tau3[m])
            est_family[m] = family_estimate(table12, est12[m], tau_family[m])
            for g in decide_gates(dev_a[m], dev_b[m], deltas[m][210]):
                accepted[(g.gate, m)] = g.accepted
        scores12[spec.name] = CandidateScore(spec.name, accepted, est12, spec.expected_effect)
        scores3[spec.name] = CandidateScore(spec.name, accepted, est3, spec.expected_effect)
        scores_family[spec.name] = CandidateScore(
            spec.name, accepted, est_family, spec.expected_effect
        )
        budget12.append(BudgetCandidate(spec.name, dev_a, dev_b, est12, spec.expected_effect))
        budget3.append(BudgetCandidate(spec.name, dev_a, dev_b, est3, spec.expected_effect))
        candidates_out[spec.name] = {
            "group": spec.group,
            "skill_id": spec.skill_id,
            "oracle_l3": {m: _oracle_l3_json(est3[m]) for m in METRICS},
            "subset_l12": {m: subset_label(est12[m]).label for m in METRICS},
            "family_l12": {
                m: {
                    "cells": {_cell(c): _frac(v) for c, v in est_family[m].cell_deltas.items()},
                    "local_harm": local_harm(est_family[m]),
                }
                for m in METRICS
            },
        }

    # 6~7. 지표, 비교 표, Local Harm
    names = {g: [c.name for c in pool.candidates if c.group == g] for g in GROUPS}
    metrics12 = _group_metrics(scores12, names)
    metrics3 = _group_metrics(scores3, names)
    metrics_family = _group_metrics(scores_family, names)
    rq1_cross3 = {
        g: {
            gate: gate_metrics([scores3[n] for n in ns], gate, CD, oracle_metric=ES)
            for gate in GATE_NAMES
        }
        for g, ns in names.items()
    }

    # Budget Curve·민감도: Ticket 005의 재표본 판정 그대로, oracle만 L3
    budget_deltas = {(m, n): deltas[m][n] for m in METRICS for n in BUDGET_NS}
    decisions_by_n = budget_decisions(budget12, budget_deltas, n_resample=n_resample)
    sens = sensitivity_decisions(budget12, n_resample=n_resample)
    group_budget3 = {g: [b for b in budget3 if b.name in ns] for g, ns in names.items()}
    budget_l3 = {
        g: {n: budget_points(decisions_by_n[n], b) for n in BUDGET_NS}
        for g, b in group_budget3.items()
    }
    sensitivity_l3 = {
        g: {m: sensitivity_rates(sens, b, m) for m in METRICS} for g, b in group_budget3.items()
    }

    # 8. RQ
    rq_l3 = _rq(
        metrics3["proposer"],
        rq1_cross3["proposer"],
        budget_l3["proposer"],
        sensitivity_l3["proposer"],
    )
    samples = manipulation_samples(records, l3_records, s0)
    rq: dict[str, Any] = {
        "primary_metric": PRIMARY_METRIC,
        "RQ_L3_0": {scope: manipulation_check(a, b) for scope, (a, b) in samples.items()},
        "RQ_L3_1": {
            m: _verdict_json(
                rq_l3_1(
                    {g: metrics12["proposer"][m][g].harmful for g in RQ_L3_1_GATES},
                    {g: metrics3["proposer"][m][g].harmful for g in RQ_L3_1_GATES},
                )
            )
            for m in METRICS
        },
        "RQ_L3_2": rq_l3_2(t005["rq"], rq_l3),
        "RQ_L3_2_detail": {s: rq_l3[s] for s in RQ_L3_2_SENTENCES},
        "RQ_L3_3": {
            g: {
                m: rq_l3_3(
                    {n: subset_label(scores12[n].oracle[m]).label for n in ns},
                    {n: subset_label(scores3[n].oracle[m]).label for n in ns},
                )
                for m in METRICS
            }
            for g, ns in names.items()
        },
    }

    skill_names = {c.skill_id: c.name for c in pool.candidates}
    by_skill = Counter(
        skill_names.get(r.key.skill_id, r.key.skill_id)
        for r in l3_records
        if r.trace.format_errors > 0
    )
    return {
        "meta": {
            "analysis_commit": run_info.get("analysis_commit"),
            "plan_sha256": run_info.get("plan_sha256"),
            "preregistration_l3_sha256": prereg_l3.detail.removeprefix("L3 sha256 "),
            "preregistration_7_8_sha256": prereg_78,
            "ticket005_analysis_sha256": hashlib.sha256(ticket005).hexdigest(),
            "section10_version": SECTION10_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_resample": n_resample,
            "n_boot": N_BOOT,
            "seed": SEED,
            "run_conditions": {
                "policy_id": POLICY_ID,
                "config_id": CONFIG_ID,
                "tool_spec_id": TOOL_SPEC_ID,
            },
            "s0_skill_id": s0,
            "primary_metric": PRIMARY_METRIC,
        },
        "episodes": {
            "total": len(l3_records),
            "format_error_episodes": dict(sorted(by_skill.items())),
            "no_disposition": sum(1 for r in l3_records if r.action is None),
        },
        "tau": {
            m: {
                "L1/L2": {_cell(c): _frac(tau12[m][c]) for c in CELLS},
                "L3": {_cell(c): _frac(tau3[m][c]) for c in L3_CELLS},
                "family_L1L2": {_cell(c): _frac(tau_family[m][c]) for c in FAMILY_CELLS},
            }
            for m in METRICS
        },
        "candidates": candidates_out,
        "metrics": {
            g: {
                m: {
                    gate: {
                        "l12": _metrics_json(metrics12[g][m][gate]),
                        "l3": _metrics_json(metrics3[g][m][gate]),
                    }
                    for gate in GATE_NAMES
                }
                for m in METRICS
            }
            for g in GROUPS
        },
        "comparison": {
            g: {
                m: {
                    gate: _comparison(metrics12[g][m][gate], metrics3[g][m][gate])
                    for gate in GATE_NAMES
                }
                for m in METRICS
            }
            for g in GROUPS
        },
        "local_harm": {
            g: {
                m: {
                    gate: {
                        "l3": _rate_json(metrics3[g][m][gate].local_harm),
                        "family_l12": _rate_json(metrics_family[g][m][gate].local_harm),
                    }
                    for gate in GATE_NAMES
                }
                for m in METRICS
            }
            for g in GROUPS
        },
        "budget_curve_l3": {
            g: {
                str(n): {
                    f"{gate}/{m}": {
                        "acceptance_rate": _summary_json(p.acceptance_rate),
                        "metrics": {k: _summary_json(v) for k, v in p.metrics.items()},
                    }
                    for (gate, m), p in points.items()
                }
                for n, points in by_n.items()
            }
            for g, by_n in budget_l3.items()
        },
        "sensitivity_l3": {
            g: {m: {k: _summary_json(v) for k, v in r.items()} for m, r in by_m.items()}
            for g, by_m in sensitivity_l3.items()
        },
        "fitness": "not_applicable",
        "rq": rq,
    }


def _group_metrics(
    scores: Mapping[str, CandidateScore], names: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, dict[str, GateMetrics]]]:
    return {
        g: {
            m: {gate: gate_metrics([scores[n] for n in ns], gate, m) for gate in GATE_NAMES}
            for m in METRICS
        }
        for g, ns in names.items()
    }


def _indicator(m: GateMetrics, name: str) -> Fraction | None:
    if name == "gain":
        return m.gain.mean
    rate: Rate = getattr(m, name)
    return rate.value


def _comparison(l12: GateMetrics, l3: GateMetrics) -> dict[str, Any]:
    out = {}
    for name in INDICATORS:
        a, b = _indicator(l12, name), _indicator(l3, name)
        out[name] = {
            "l12": _frac(a),
            "l3": _frac(b),
            "diff": None if a is None or b is None else _frac(b - a),
        }
    return out


def _oracle_l3_json(e: OracleEstimate) -> dict[str, Any]:
    subset = subset_label(e)
    return {
        "delta": _frac(e.delta),
        "interval": [_frac(e.interval[0]), _frac(e.interval[1])],
        "cells": {_cell(c): _frac(v) for c, v in e.cell_deltas.items()},
        "useful": useful(e),
        "harmful": harmful(e),
        "unsupported": unsupported(e),
        "local_harm": local_harm(e),
        "effect": classify_effect(e),
        "subset": subset.label,
        "big_cells": [_cell(c) for c in subset.big_cells],
    }


def load_l3_records(runs_dir: Path, pool: PoolSpec) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for skill in (pool.s0_skill_id, *(c.skill_id for c in pool.candidates)):
        store = EpisodeStore.for_run(
            runs_dir,
            suite=L3_ORACLE_SPLIT,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id=skill,
        )
        records += EpisodeStore(store.path, repair=False).records()
    return records


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

VERDICT_KO = {
    "supported": "지지",
    "partially_supported": "부분 지지",
    "not_supported": "지지 안 됨",
    "undecidable": "판정 불가",
    "grows": "커진다",
    "partly_grows": "일부 커진다",
    "does_not_grow": "커지지 않는다",
    "trend_supported": "추세 지지",
    "partial_trend_support": "부분 추세 지지",
    "trend_not_supported": "추세 지지 안 됨",
    CONFIRMED: "조작 확인",
    NOT_CONFIRMED: "조작 미달",
    UNDECIDABLE: "판정 불가",
    "preserved": "보존",
    "changed": "바뀜",
}
METRIC_KO = {CD: "CD", ES: "ES"}
SENTENCE_KO = {
    "RQ2_first": "RQ2 첫째",
    "RQ2_second": "RQ2 둘째",
    "RQ3_first": "RQ3 첫째",
    "RQ3_trend": "RQ3 추세",
}
NOTES_L3 = (
    "게이트 결정은 Ticket 005의 결과 그대로다(본 판정, Budget Curve 재표본 판정, C1 민감도 판정). "
    "L3는 사후 채점에만 쓴다(고정 결정 3).",
    "L1/L2 Oracle은 family x level 6칸(칸당 50), L3 Oracle은 family 3칸(칸당 100)이라 칸 정의가 "
    "다르다. Local Harm과 하위집합 라벨(RQ-L3-3)은 이 차이를 함께 읽어야 한다. 보조 지표로 L1/L2를 "
    "family 3칸으로 다시 묶은 Local Harm을 나란히 둔다.",
    "traffic_spike L3는 이동 없는 대조 칸이다(T1: L2 extra_deployment와 같은 구조).",
    "Oracle 적합성(F)은 L3에 적용하지 않는다. F는 한 family 안의 L1·L2 두 칸을 비교하는 정의다.",
    "후보 수가 12개라 모든 비교는 검정 없이 기술적으로 한다. 주 결론은 proposer 후보로만 낸다.",
    "RQ-L3-2는 판정 범주가 같은지(보존 / 바뀜)만 보고하고 종합하지 않는다.",
)
LIMITS_L3 = (
    "ES 바닥 효과: S0의 bad_deployment ES가 거의 0이라, ES 기준 지표는 그 칸의 처분 회귀를 거의 "
    "보지 못한다(Ticket 005 결과 보고서 한계).",
    "L3는 L2 방해 요인의 조합이라 이동의 크기가 크지 않을 수 있다. dep L3만 새 술어 조합을 만든다.",
    "Oracle에서 후보는 S0 반복 0과 LLM seed를 공유하고, 그 반복 0이 다수결 다섯 표 중 하나다"
    "(Ticket 005 §8-5).",
)


def render_markdown_l3(result: Mapping[str, Any]) -> str:
    meta, rq = result["meta"], result["rq"]
    lines = [
        "# Ticket 006 L3 분석 결과",
        "",
        f"- Ticket 006 사전 등록 sha256: `{meta['preregistration_l3_sha256']}`",
        f"- 7.8 사전 등록 sha256: `{meta['preregistration_7_8_sha256']}`",
        f"- Ticket 005 결과 sha256: `{meta['ticket005_analysis_sha256']}` "
        "(같은 레코드로 재현 확인)",
        f"- 분석 코드 커밋: `{meta['analysis_commit']}`, "
        f"L3 실행 계획 sha256: `{meta['plan_sha256']}`",
        f"- §10 해석: {meta['section10_version']}",
        f"- Python {meta['python']}, numpy {meta['numpy']}, 재표본 {meta['n_resample']}회, "
        f"bootstrap {meta['n_boot']}회, seed {meta['seed']}",
        f"- L3 에피소드 {result['episodes']['total']}개, 처분 없이 끝남 "
        f"{result['episodes']['no_disposition']}개, 형식 오류 에피소드 "
        f"{result['episodes']['format_error_episodes'] or '없음'}",
        "",
        "## RQ-L3-0 조작 확인 (S0 다수결)",
        "",
        "| 범위 | CD L1/L2 | CD L3 | Δ₀ [95% 구간] | Δ₀ < -ε | g L1/L2 | g L3 | g 차이 [95% 구간] "
        "| 차이 > ε | 종합 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scope, v in rq["RQ_L3_0"].items():
        cd, gap = v["cd"], v["gap"]
        lines.append(
            f"| {scope} | {_f(cd['l12'])} | {_f(cd['l3'])} | {_f(cd['delta'])} {_iv(cd)} | "
            f"{_b(cd['holds'])} | {_f(gap['l12'])} | {_f(gap['l3'])} | {_f(gap['delta'])} "
            f"{_iv(gap)} | {_b(gap['holds'])} | **{VERDICT_KO[v['verdict']]}** |"
        )

    lines += ["", "## τ_cell", "", "| metric | 칸 | τ |", "| --- | --- | --- |"]
    for m in METRICS:
        for kind in ("L1/L2", "L3", "family_L1L2"):
            for cell, value in result["tau"][m][kind].items():
                lines.append(f"| {METRIC_KO[m]} | {cell} | {_f(value)} |")

    lines += ["", "## RQ-L3-1 Harmful의 이동 (proposer, 주 metric CD)", ""]
    for m in METRICS:
        v = rq["RQ_L3_1"][m]
        lines.append(f"- {METRIC_KO[m]}: **{VERDICT_KO[v['verdict']]}** {_cmp(v)}")

    for group in GROUPS:
        lines += ["", f"## 비교 표 ({group})", ""]
        lines.append("칸: L1/L2 → L3 (차이).")
        lines.append("")
        lines.append("| metric | 게이트 | " + " | ".join(INDICATORS) + " |")
        lines.append("| --- | --- |" + " --- |" * len(INDICATORS))
        for m in METRICS:
            for gate in GATE_NAMES:
                row = result["comparison"][group][m][gate]
                cells = [
                    f"{_f(row[k]['l12'])} → {_f(row[k]['l3'])} ({_f(row[k]['diff'])})"
                    for k in INDICATORS
                ]
                lines.append(f"| {METRIC_KO[m]} | {gate} | " + " | ".join(cells) + " |")

    lines += ["", "## Local Harm (L3 3칸 대 L1/L2 family 3칸)", ""]
    lines.append("| 집합 | metric | 게이트 | L3 | L1/L2 family |")
    lines.append("| --- | --- | --- | --- | --- |")
    for group in GROUPS:
        for m in METRICS:
            for gate in GATE_NAMES:
                lh = result["local_harm"][group][m][gate]
                lines.append(
                    f"| {group} | {METRIC_KO[m]} | {gate} | {_rate(lh['l3'])} | "
                    f"{_rate(lh['family_l12'])} |"
                )

    lines += ["", "## RQ-L3-2 판정 범주의 보존 (proposer)", ""]
    lines.append("| 문장 | metric | L1/L2 | L3 | 결과 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for s, by_m in rq["RQ_L3_2"].items():
        for m, v in by_m.items():
            lines.append(
                f"| {SENTENCE_KO[s]} | {METRIC_KO[m]} | {VERDICT_KO[v['l12']]} | "
                f"{VERDICT_KO[v['l3']]} | **{VERDICT_KO[v['status']]}** |"
            )

    lines += ["", "## RQ-L3-3 하위집합 라벨의 전이 (기술용)", ""]
    for group in GROUPS:
        for m in METRICS:
            v = rq["RQ_L3_3"][group][m]
            changed = ", ".join(f"{c['name']} {c['l12']}→{c['l3']}" for c in v["changed"]) or "없음"
            lines.append(
                f"- {group} {METRIC_KO[m]}: 같음 {v['same']}/{v['n']} (한쪽 이상 uncertain "
                f"{v['either_uncertain']}). 바뀐 후보: {changed}"
            )

    lines += ["", "## 후보별 L3 Oracle", ""]
    lines.append("| 후보 | 집합 | metric | L3 Δ [95% 구간] | 칸 Δ | L3 하위집합 | L1/L2 하위집합 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, c in result["candidates"].items():
        for m in METRICS:
            o = c["oracle_l3"][m]
            cell_text = ", ".join(f"{k.split('/')[0]} {_f(v)}" for k, v in o["cells"].items())
            lines.append(
                f"| {name} | {c['group']} | {METRIC_KO[m]} | {_f(o['delta'])} {_iv(o)} | "
                f"{cell_text} | "
                f"{o['subset']} | {c['subset_l12'][m]} |"
            )
    lines += ["", "## 해석 주의", ""] + [f"- {n}" for n in NOTES_L3]
    lines += ["", "## 한계", ""] + [f"- {n}" for n in LIMITS_L3]
    return "\n".join(lines) + "\n"


def _f(x: Mapping[str, Any] | None) -> str:
    return "판정 불가" if x is None else f"{x['float']:.3f}"


def _iv(v: Mapping[str, Any]) -> str:
    low, high = v["interval"]
    return f"[{_f(low)}, {_f(high)}]"


def _b(b: bool | None) -> str:
    return "판정 불가" if b is None else ("예" if b else "아니오")


def _rate(r: Mapping[str, Any]) -> str:
    if r["n"] == 0:
        return f"0/0 판정 불가 (제외 {r['excluded']})"
    return f"{r['successes']}/{r['n']}"


def _cmp(v: Mapping[str, Any]) -> str:
    return "(" + "; ".join(f"{c['name']}: {_b(c['holds'])}" for c in v["comparisons"]) + ")"


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-l3-analysis")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ticket005", type=Path, default=TICKET005_RESULT)
    parser.add_argument("--pool-spec", type=Path, default=None, help="default: the frozen pool")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--analysis-commit", default=None)
    parser.add_argument("--plan-sha256", default=None)
    args = parser.parse_args(argv)

    json_path = args.out_dir / "analysis.json"
    if json_path.exists():
        print(f"error: {json_path} already exists; the analysis runs once", file=sys.stderr)
        return 1
    t005 = args.ticket005.read_bytes()
    digest = hashlib.sha256(t005).hexdigest()
    sha_file = args.ticket005.with_name(args.ticket005.name + ".sha256")
    recorded = sha_file.read_text().split()[0] if sha_file.exists() else None
    if digest != recorded:
        print(f"error: {args.ticket005} sha256 {digest} does not match {recorded}", file=sys.stderr)
        return 1
    if args.pool_spec is None and digest != TICKET005_ANALYSIS_SHA256:
        print(
            f"error: {args.ticket005} sha256 {digest} is not the Ticket 005 result "
            f"{TICKET005_ANALYSIS_SHA256}",
            file=sys.stderr,
        )
        return 1
    pool = load_pool_spec(args.pool_spec) if args.pool_spec else load_pool(args.root)
    result = run_l3_analysis(
        load_records(args.runs_dir, pool),
        load_l3_records(args.runs_dir, pool),
        EpisodeStore(args.noise, repair=False).records(),
        pool,
        ticket005=t005,
        run_info={"analysis_commit": args.analysis_commit, "plan_sha256": args.plan_sha256},
        root=args.root,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = ticket005_bytes(result)
    json_path.write_bytes(data)
    out_digest = hashlib.sha256(data).hexdigest()
    (args.out_dir / "analysis.json.sha256").write_text(f"{out_digest}  analysis.json\n")
    (args.out_dir / "analysis.md").write_text(render_markdown_l3(result), encoding="utf-8")
    print(f"analysis.json sha256 {out_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
