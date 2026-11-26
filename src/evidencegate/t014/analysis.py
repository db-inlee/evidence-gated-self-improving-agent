"""Ticket 014 분석 정의(사전 등록 4~7절). 기존 정의 함수를 그대로 부른다.

사전 등록 항목과 함수(괄호는 쓰는 기존 함수):
- 5절 h(150), δ(900): `thresholds`(`t009.rq9.cell_threshold_h`, `exact.null_diff_quantile_exact`)
- 4절 C4·C5·C6-IUT·SAGE λ1: `gate_row`(`gates.gate_c4`, `t009.rq9.cell_condition`,
  `t013.sim.tango_lower`, `t013.sim.sage_accept`)
- 6절 Oracle 대상: `oracle_targets`
- 6절 대칭 mean 라벨: `oracle_label`(`exact.stratified_bootstrap_indices`,
  `gates._percentile_of_totals`)
- 7.1절 RQ14-1: `rq14_1`(`t009.rq9.rq9_1`)
- 7.2절 RQ14-2: `rq14_2`(`exact.clopper_pearson`)
- 7.3절 RQ14-3: `rq14_3`

- 판정 비교는 유리수로 한다. Tango 하한만 이분법 float이고, -m에서 `TANGO_TIE` 안이면 통과로 보지
  않는다(`tango_pass`).
- Oracle 케이스 차이 d_i = Σ_r 대상_r - Σ_r S0_r(정수, 반복 0~2)이고 Δ는 d / 3의 평균이다. 칸이
  모두 100 케이스라 전체 평균이 칸 동일 가중 평균과 같다. 구간은 칸 층화 paired bootstrap(2,000회,
  seed 0) 2.5·97.5 백분위수이고, 칸 구간은 같은 재표본의 칸 합에서 낸다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from evidencegate.eval_constants import N_BOOT, ORACLE_CI_QS, SEED
from evidencegate.eval_data import CELLS, CaseId, PairedTable, cell_of, outcomes
from evidencegate.exact import (
    clopper_pearson,
    mean_fraction,
    null_diff_quantile_exact,
    stratified_bootstrap_indices,
)
from evidencegate.gates import _percentile_of_totals, bootstrap_quantiles, gate_c4
from evidencegate.metrics import METRICS
from evidencegate.runner import EpisodeRecord
from evidencegate.t009.rq9 import cell_condition, cell_threshold_h, fisher_one_sided, rq9_1
from evidencegate.t013.sim import sage_accept, tango_lower
from evidencegate.t014.settings import (
    C6_ALPHA,
    CALIBRATION_REPEATS,
    MARGIN,
    N_C,
    N_DEV_B,
    ORACLE_REPEATS,
    PREDICTED_LOSS,
    SAGE_ALPHA,
    SAGE_LAMBDA,
    SAGE_TAU,
    TANGO_TIE,
    USEFUL_EPS,
)

CD = "correct_disposition"
GATES = ("C4", "C5", "C6_IUT", "SAGE_l1")


# 5절 보정 -----------------------------------------------------------------------------


def thresholds(
    records: Sequence[EpisodeRecord],
    cases: Sequence[CaseId],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, dict[str, Fraction]]:
    """metric마다 h(N_C), 달성 거짓 기각률, δ(N_DEV_B). 보정 케이스 x 반복 0~4가 모두 필요하다."""
    out = {}
    for metric in METRICS:
        table = {
            r: outcomes(
                records, skill_id=_skill(records), split="-", metric=metric, repeat=r, cases=cases
            )
            for r in CALIBRATION_REPEATS
        }
        h, achieved = cell_threshold_h(table, N_C, n_boot=n_boot, seed=seed)
        delta = null_diff_quantile_exact(records, metric, N_DEV_B, n_boot=n_boot, seed=seed)
        out[metric] = {"h": h, "achieved": achieved, "delta": delta}
    return out


def _skill(records: Sequence[EpisodeRecord]) -> str:
    skills = {r.key.skill_id for r in records}
    if len(skills) != 1:
        raise ValueError(f"calibration needs one skill, got {sorted(skills)}")
    return skills.pop()


# 4절 게이트 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRow:
    """proposer 후보 하나·metric 하나의 Dev-B 게이트(후보 반복 0 대 S0 반복 0)."""

    name: str
    metric: str
    mean: Fraction
    lower: Fraction  # C4 bootstrap 5번째 백분위수
    cell_deltas: tuple[Fraction, ...]  # CELLS 순서
    w: int  # 후보 성공·S0 실패
    loss: int  # 후보 실패·S0 성공
    n_s: int  # S0 성공
    tango_lowers: tuple[float, ...]  # 칸별 Tango 한쪽 하한
    c4: bool
    c5: bool
    c6: bool
    sage: bool

    def accepted(self, gate: str) -> bool:
        return {"C4": self.c4, "C5": self.c5, "C6_IUT": self.c6, "SAGE_l1": self.sage}[gate]


def tango_pass(lower: float) -> bool:
    return lower > float(-MARGIN) + TANGO_TIE


def gate_row(
    name: str,
    metric: str,
    cand: Mapping[CaseId, int],
    s0: Mapping[CaseId, int],
    *,
    h: Fraction,
    delta: Fraction,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> GateRow:
    if set(cand) != set(s0):
        raise ValueError(f"{name}: candidate and S0 cases differ")
    cases = tuple(sorted(cand))
    table = PairedTable(
        split="dev_b",
        metric=metric,
        cases=cases,
        cells=tuple(cell_of(c) for c in cases),
        candidate=tuple(cand[c] for c in cases),
        baseline=tuple(s0[c] for c in cases),
    )
    c4 = gate_c4(table, delta, n_boot=n_boot, seed=seed)
    groups = table.cell_indices()
    cell_deltas = tuple(mean_fraction([table.deltas[i] for i in g]) for g in groups)
    lowers = []
    for g in groups:
        w_c = sum(1 for i in g if table.deltas[i] == 1)
        l_c = sum(1 for i in g if table.deltas[i] == -1)
        lowers.append(tango_lower(w_c, l_c, len(g), C6_ALPHA))
    w = sum(1 for d in table.deltas if d == 1)
    loss = sum(1 for d in table.deltas if d == -1)
    n_s = sum(table.baseline)
    sage = sage_accept(w, loss, n_s, lam=SAGE_LAMBDA, tau=float(SAGE_TAU), alpha=SAGE_ALPHA)
    return GateRow(
        name=name,
        metric=metric,
        mean=c4.mean_delta,
        lower=c4.lower if c4.lower is not None else Fraction(0),
        cell_deltas=cell_deltas,
        w=w,
        loss=loss,
        n_s=n_s,
        tango_lowers=tuple(lowers),
        c4=c4.accepted,
        c5=c4.accepted and cell_condition(cell_deltas, h),
        c6=c4.accepted and all(tango_pass(x) for x in lowers),
        sage=sage,
    )


def oracle_targets(rows: Sequence[GateRow]) -> list[str]:
    """CD 기준 C4 채택 또는 SAGE λ1 채택(이름 순)."""
    return sorted(r.name for r in rows if r.metric == CD and (r.c4 or r.sage))


# 6절 Oracle ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleLabel:
    mean: Fraction
    lower: Fraction
    upper: Fraction
    cells: tuple[tuple[Fraction, Fraction, Fraction], ...]  # 칸마다 (점추정, 하한, 상한)
    m: bool  # 어떤 칸 상한 < -m
    m_point: bool  # 어떤 칸 점추정 < -m(기술용)
    useful: bool  # 전체 하한 > ε
    clean_useful: bool  # Useful 그리고 M 아님
    m_uncertain: bool  # M 아님 그리고 어떤 칸 하한 < -m
    useful_uncertain: bool  # 전체 하한 <= ε < 전체 상한


def oracle_label(
    cand: Mapping[int, Mapping[CaseId, int]],
    s0: Mapping[int, Mapping[CaseId, int]],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> OracleLabel:
    """반복 → 케이스 → 0/1(대상, S0). 반복 0~2, 같은 케이스."""
    if sorted(cand) != list(ORACLE_REPEATS) or sorted(s0) != list(ORACLE_REPEATS):
        raise ValueError(f"the symmetric Oracle needs repeats {ORACLE_REPEATS} for both")
    cases = tuple(sorted(cand[0]))
    if any(set(t) != set(cases) for t in (*cand.values(), *s0.values())):
        raise ValueError("every repeat needs the same Oracle cases")
    scale = len(ORACLE_REPEATS)
    table = PairedTable(
        split="oracle",
        metric="-",
        cases=cases,
        cells=tuple(cell_of(c) for c in cases),
        candidate=tuple(sum(cand[r][c] for r in ORACLE_REPEATS) for c in cases),
        baseline=tuple(sum(s0[r][c] for r in ORACLE_REPEATS) for c in cases),
    )
    groups = table.cell_indices()
    blocks = stratified_bootstrap_indices([len(g) for g in groups], n_boot=n_boot, seed=seed)
    lower, upper = (
        q / scale for q in bootstrap_quantiles(table, ORACLE_CI_QS, n_boot, seed, blocks=blocks)
    )
    deltas = np.asarray(table.deltas, dtype=np.int64)
    cells = []
    for group, block in zip(groups, blocks, strict=True):
        totals = sorted(
            int(t) for t in deltas[np.asarray(group, dtype=np.int64)][block].sum(axis=1)
        )
        size = len(group) * scale
        cells.append(
            (
                Fraction(int(deltas[group].sum()), size),
                _percentile_of_totals(totals, ORACLE_CI_QS[0]) / size,
                _percentile_of_totals(totals, ORACLE_CI_QS[1]) / size,
            )
        )
    m = any(hi < -MARGIN for _, _, hi in cells)
    useful = lower > USEFUL_EPS
    return OracleLabel(
        mean=Fraction(int(deltas.sum()), len(cases) * scale),
        lower=lower,
        upper=upper,
        cells=tuple(cells),
        m=m,
        m_point=any(mid < -MARGIN for mid, _, _ in cells),
        useful=useful,
        clean_useful=useful and not m,
        m_uncertain=not m and any(lo < -MARGIN for _, lo, _ in cells),
        useful_uncertain=lower <= USEFUL_EPS < upper,
    )


# 7절 RQ -------------------------------------------------------------------------------


def _label(labels: Mapping[str, OracleLabel], name: str) -> OracleLabel:
    if name not in labels:
        raise ValueError(f"{name} is accepted but has no Oracle label")
    return labels[name]


def rq14_1(rows: Sequence[GateRow], labels: Mapping[str, OracleLabel]) -> dict[str, Any]:
    """C4 채택 후보마다 (C6-IUT 채택, M). 판정은 `rq9_1` 규칙 그대로."""
    c4 = [r for r in rows if r.c4]
    main = [(r.c6, _label(labels, r.name).m) for r in c4]
    uncertain = [labels[r.name].m_uncertain for r in c4]
    as_m = [(c6, m or u) for (c6, m), u in zip(main, uncertain, strict=True)]
    result = rq9_1(main)
    n, f, k = result["n"], result["f"], result["k"]
    return {
        **result,
        # 기술용: 주어진 n, f, k에서 달성 가능한 최소 단측 p(a = min(f, k)). 판정에는 쓰지 않는다.
        "p_min": fisher_one_sided(a=min(f, k), f=f, k=k, n=n),
        "m_uncertain": sum(uncertain),
        "m_point": sum(1 for r in c4 if labels[r.name].m_point),
        "table": _two_by_two(main),
        "sensitivity": {"uncertain_as_not_m": rq9_1(main), "uncertain_as_m": rq9_1(as_m)},
    }


def _two_by_two(rows: Sequence[tuple[bool, bool]]) -> dict[str, int]:
    return {
        f"{'kept' if kept else 'filtered'}_{'m' if m else 'not_m'}": sum(
            1 for k, x in rows if k == kept and x == m
        )
        for kept in (False, True)
        for m in (True, False)
    }


def rq14_2(rows: Sequence[GateRow], labels: Mapping[str, OracleLabel]) -> dict[str, Any]:
    """C4 채택 Clean Useful 중 C6-IUT가 거른 비율과 정확 구간. 판정 문구 없음."""
    clean = [r for r in rows if r.c4 and _label(labels, r.name).clean_useful]
    u_4, u = len(clean), sum(1 for r in clean if not r.c6)
    interval = clopper_pearson(u, u_4)
    contains: bool | str = (
        "undefined" if interval is None else interval[0] <= float(PREDICTED_LOSS) <= interval[1]
    )
    return {
        "u_4": u_4,
        "u": u,
        "loss": Fraction(u, u_4) if u_4 else None,
        "interval": interval,
        "predicted_loss": PREDICTED_LOSS,
        "contains_prediction": contains,
    }


def rq14_3(rows: Sequence[GateRow], labels: Mapping[str, OracleLabel]) -> dict[str, Any]:
    """게이트마다 채택 수, M 채택 수, M_point 채택 수, Clean Useful 채택 수(기술용)."""
    out = {}
    for gate in GATES:
        accepted = [_label(labels, r.name) for r in rows if r.accepted(gate)]
        out[gate] = {
            "accepted": len(accepted),
            "m": sum(1 for x in accepted if x.m),
            "m_point": sum(1 for x in accepted if x.m_point),
            "clean_useful": sum(1 for x in accepted if x.clean_useful),
        }
    return out


__all__ = [
    "CD",
    "CELLS",
    "GATES",
    "GateRow",
    "OracleLabel",
    "gate_row",
    "oracle_label",
    "oracle_targets",
    "rq14_1",
    "rq14_2",
    "rq14_3",
    "tango_pass",
    "thresholds",
]
