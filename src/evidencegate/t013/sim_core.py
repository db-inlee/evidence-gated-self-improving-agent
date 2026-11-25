"""Ticket 013 본 모의의 계산부: 시나리오, 풀 하나의 판정, 집계와 Go 판정.

동결 계획(`plans/ticket-013-sim-plan.json`)과 부록(`plans/ticket-013-sim-plan-addendum.json`)을
따른다. 입출력과 체크포인트는 `evidencegate.t013.sim`에 있다.

- 참 효과: 기준선 케이스 집단 전체에서 clip(p + s) - p의 정확한 평균(부록 true_effect).
- 이동량: 목표 δ가 clip 아래에서 정확히 맞도록 구간별 선형식으로 푼다(부록 shift_solving).
- 난수: 풀마다 SeedSequence([0, 기준선 번호, n_c, 풀 번호]). 같은 풀의 정규 난수를 시나리오·margin·
  게이트·결합 설정이 공유하고, 결합은 L2 칸의 정규 난수만 바꾼다(부록 conventions, seed_coupling).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from evidencegate.eval_constants import C4_Q, EPSILON, ORACLE_CI_QS, SEED
from evidencegate.eval_data import CELLS, CaseId
from evidencegate.exact import stratified_bootstrap_indices
from evidencegate.t009 import power_sim as ps
from evidencegate.t009.rq9 import cell_threshold_h
from evidencegate.t013 import sim

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# 계획 값(tests/t013/test_sim_run.py가 동결 계획과 대조한다)
ALPHA = Fraction(1, 20)
ROOM_AT_MOST = Fraction(9, 10)
USEFUL_SHIFT = Fraction(1, 10)
GLOBAL_UP = Fraction(1, 10)  # +2 epsilon
SAGE_LAMBDAS = (1, 2)
SAGE_TAU = 0.05
CREDIT_EPS = 0.005
COUPLING_RHO = 0.5
ORACLE_PER_CELL = 50
S0_REPEATS = 5
FALSE_ACCEPT_AT_MOST = 0.05
USEFUL_POWER_AT_LEAST = 0.7
EPISODES_AT_MOST = 75000
BORDERLINE_SE = 2
DEMO_M = Fraction(1, 20)  # percentile 시연 조건의 margin
CELL_NAMES = tuple(f"{f}/{lv}" for f, lv in CELLS)
PAIRS = ((0, 1), (2, 3), (4, 5))  # 같은 family의 (L1, L2)
INADMISSIBLE_SCENARIOS = (
    "null_all_cells",
    "no_global_one_cell_minus_m",
    "global_up_one_cell_minus_m",
    "global_up_one_cell_minus_2m",
)
PLACEMENTS = ("realistic", "worst")
GO_GATES = ("C4", "C5", "C6_IUT", "SAGE_l1", "SAGE_l2", "SAGE_cell_l1", "SAGE_cell_l2", "credit")
DEPENDENCE = ("main", "s0_independent", "idea_groups")


# 기준선과 이동량 ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    name: str
    cells: tuple[tuple[Fraction, ...], ...]  # 칸마다 케이스 성공률(집단)

    @property
    def means(self) -> tuple[Fraction, ...]:
        return tuple(sum(c, Fraction(0)) / len(c) for c in self.cells)

    def arrays(self) -> list[FloatArray]:
        return [np.asarray([float(p) for p in c], dtype=float) for c in self.cells]


def realized(rates: Sequence[Fraction], shift: Fraction) -> Fraction:
    """집단 평균 clip(p + s, 0, 1) - p(정확)."""
    total = sum((min(max(p + shift, Fraction(0)), Fraction(1)) - p for p in rates), Fraction(0))
    return total / len(rates)


def solve_shift(cells: Sequence[Sequence[Fraction]], target: Fraction) -> Fraction | None:
    """Σ_c realized(cell, s) = target인 s(구간별 선형, 정확). 해가 없으면 None."""
    if target == 0:
        return Fraction(0)

    def total(s: Fraction) -> Fraction:
        return sum((realized(c, s) for c in cells), Fraction(0))

    sign = 1 if target > 0 else -1
    points = sorted(
        {Fraction(0)} | {(1 - p) if sign > 0 else -p for c in cells for p in c}, key=abs
    )
    points = [x for x in points if x * sign >= 0]
    prev, prev_v = Fraction(0), Fraction(0)
    for x in points[1:] if points and points[0] == 0 else points:
        v = total(x)
        if (v - target) * sign >= 0:
            if v == prev_v:
                return x
            return prev + (target - prev_v) * (x - prev) / (v - prev_v)
        prev, prev_v = x, v
    return None


def room_cells(b: Baseline) -> list[int]:
    return [i for i, m in enumerate(b.means) if m <= ROOM_AT_MOST]


def loss_cell(b: Baseline, placement: str) -> int:
    """realistic: 평균 최고, worst: p(1 - p) 최대. 동점이면 앞 칸(CELLS 순서)."""
    means = b.means
    scores = [m if placement == "realistic" else m * (1 - m) for m in means]
    return scores.index(max(scores))


@dataclass(frozen=True)
class Scenario:
    name: str
    placement: str | None
    m: Fraction | None  # margin에 기대지 않는 시나리오는 None
    feasible: bool
    shifts: tuple[Fraction, ...]
    deltas: tuple[Fraction, ...]

    @property
    def delta(self) -> Fraction:
        return sum(self.deltas, Fraction(0)) / len(self.deltas)

    @property
    def key(self) -> tuple[str, str]:
        return self.name, self.placement or "-"


def _scenario(name: str, placement: str | None, m: Fraction | None, b: Baseline,
              shifts: Sequence[Fraction | None]) -> Scenario:  # fmt: skip
    if any(s is None for s in shifts):
        zero = (Fraction(0),) * len(b.cells)
        return Scenario(name, placement, m, False, zero, zero)
    fixed = tuple(Fraction(s) for s in shifts if s is not None)
    deltas = tuple(realized(c, s) for c, s in zip(b.cells, fixed, strict=True))
    return Scenario(name, placement, m, True, fixed, deltas)


def build_scenarios(b: Baseline, m: Fraction) -> list[Scenario]:
    """margin m에 대한 시나리오: 전 칸 0, 손실 셋 x 배치 둘, 대표 Useful(margin 무관)."""
    k = len(b.cells)
    out = [_scenario("null_all_cells", None, None, b, [Fraction(0)] * k)]
    room = room_cells(b)
    for name, loss, target in (
        ("no_global_one_cell_minus_m", -m, Fraction(0)),
        ("global_up_one_cell_minus_m", -m, GLOBAL_UP),
        ("global_up_one_cell_minus_2m", -2 * m, GLOBAL_UP),
    ):
        for placement in PLACEMENTS:
            lc = loss_cell(b, placement)
            shifts: list[Fraction | None] = [Fraction(0)] * k
            shifts[lc] = solve_shift([b.cells[lc]], loss)
            others = [c for c in room if c != lc]
            common = (
                solve_shift([b.cells[c] for c in others], k * target - loss) if others else None
            )
            for c in others:
                shifts[c] = common
            if not others:
                shifts[lc] = None
            out.append(_scenario(name, placement, m, b, shifts))
    useful: list[Fraction | None] = [Fraction(0)] * k
    for c in room:
        useful[c] = solve_shift([b.cells[c]], USEFUL_SHIFT)
    if not room:
        useful[0] = None
    out.append(_scenario("useful_representative", None, None, b, useful))
    return out


def inadmissible(deltas: Sequence[Fraction | int], delta: Fraction, m: Fraction) -> bool:
    """δ <= 0 또는 어떤 δ_c <= -m(경계는 귀무 쪽)."""
    return delta <= 0 or any(d <= -m for d in deltas)


def useful(deltas: Sequence[Fraction | int], delta: Fraction, m: Fraction) -> bool:
    return delta > 0 and all(d > -m for d in deltas)


# 난수 ------------------------------------------------------------------------------------------


_ERF = np.vectorize(math.erf, otypes=[float])


def uniforms(normals: FloatArray, *, coupled: bool) -> FloatArray:
    """칸 축이 0번인 정규 난수 → 균등 난수. 결합이면 L2 칸 = rho x L1 + sqrt(1 - rho^2) x L2."""
    z = normals.copy()
    if coupled:
        for l1, l2 in PAIRS:
            z[l2] = COUPLING_RHO * normals[l1] + math.sqrt(1 - COUPLING_RHO**2) * normals[l2]
    out: FloatArray = 0.5 * (1.0 + _ERF(z / math.sqrt(2.0)))
    return out


def phi(counts: Mapping[str, int]) -> float | None:
    n11, n10, n01, n00 = counts["n11"], counts["n10"], counts["n01"], counts["n00"]
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return None if den == 0 else (n11 * n00 - n10 * n01) / math.sqrt(den)


def _pair_counts(a: BoolArray, b: BoolArray) -> dict[str, int]:
    return {
        "n11": int((a & b).sum()),
        "n10": int((a & ~b).sum()),
        "n01": int((~a & b).sum()),
        "n00": int((~a & ~b).sum()),
    }


# 빠른 판정 부품 -----------------------------------------------------------------------------------


def block_counts(blocks: Sequence[IntArray]) -> list[IntArray]:
    return [ps._counts(b) for b in blocks]


def _totals(d: IntArray, counts: Sequence[IntArray]) -> list[IntArray]:
    out, start = [], 0
    for c in counts:
        size = c.shape[1]
        out.append(c @ d[start : start + size])
        start += size
    return out


def c4_fast(d: IntArray, counts: Sequence[IntArray], delta: Fraction) -> BoolArray:
    """`power_sim.c4_parts`와 같은 판정(개수 행렬을 미리 만든 판)."""
    n = d.shape[0]
    totals = np.asarray(sum(_totals(d, counts)))
    v, _den = ps._quantile_scaled(totals, C4_Q)
    sums = d.sum(axis=0)
    return np.asarray((v > 0) & (sums * delta.denominator > delta.numerator * n))


def percentile_lowers(
    d: IntArray, counts: Sequence[IntArray], n_c: int, alpha: Fraction
) -> FloatArray:
    """칸마다 paired bootstrap 평균의 alpha 백분위수(선형 보간), (6, 후보)."""
    out = []
    for totals in _totals(d, counts):
        means = np.sort(totals / n_c, axis=0)
        pos = float(alpha) * (means.shape[0] - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, means.shape[0] - 1)
        out.append(means[lo] + (means[hi] - means[lo]) * (pos - lo))
    return np.asarray(out)


def _oracle_uncertain(d: IntArray, counts: Sequence[IntArray]) -> int:
    """Oracle 구간이 +ε 또는 -ε를 가로지르는 후보 수(`oracle.classify_effect`가 None)."""
    n = d.shape[0]
    totals = np.asarray(sum(_totals(d, counts)))
    ends = [ps._quantile_scaled(totals, q) for q in ORACLE_CI_QS]
    up = [v * EPSILON.denominator > EPSILON.numerator * n * den for v, den in ends]
    down = [v * EPSILON.denominator < -EPSILON.numerator * n * den for v, den in ends]
    return int(((up[0] != up[1]) | (down[0] != down[1])).sum())


# 풀 하나 ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    baseline: Baseline
    baseline_index: int
    n_c: int
    coupled: bool
    n_boot: int
    candidates: int
    margins: tuple[Fraction, ...]
    v1_effects: tuple[tuple[Fraction, ...], ...]
    group_sizes: tuple[int, ...]
    demo: bool  # percentile 시연(기준선 a, 전 칸 0, n_c 70, m 0.05, main)


_COUNTS: dict[tuple[int, int], list[IntArray]] = {}
_TRUE: dict[tuple[str, tuple[Fraction, ...]], tuple[Fraction, ...]] = {}


def _true_effect(b: Baseline, shifts: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """경험 분포 후보의 참 효과(부록 true_effect). 기준선과 이동량마다 한 번만 계산한다."""
    key = (b.name, shifts)
    if key not in _TRUE:
        _TRUE[key] = tuple(realized(c, e) for c, e in zip(b.cells, shifts, strict=True))
    return _TRUE[key]


def _counts_for(n: int, n_boot: int) -> list[IntArray]:
    key = (n, n_boot)
    if key not in _COUNTS:
        _COUNTS[key] = block_counts(stratified_bootstrap_indices([n] * 6, n_boot, SEED))
    return _COUNTS[key]


def _draw_cases(rng: np.random.Generator, pops: Sequence[FloatArray], n: int) -> FloatArray:
    return np.stack([p[rng.integers(0, len(p), n)] for p in pops])


def _calibration(rng: np.random.Generator, s: Setup, pops: Sequence[FloatArray]) -> dict[str, Any]:
    rates = _draw_cases(rng, pops, s.n_c)  # (6, n)
    normals = rng.standard_normal((S0_REPEATS, 6, s.n_c))
    runs = np.stack([uniforms(normals[r], coupled=s.coupled) < rates for r in range(S0_REPEATS)])
    cal_h: dict[int, dict[CaseId, int]] = {r: {} for r in range(S0_REPEATS)}
    cal_d: dict[int, dict[tuple[str, int, str], int]] = {r: {} for r in range(S0_REPEATS)}
    for k, (f, lv) in enumerate(CELLS):
        for r in range(S0_REPEATS):
            for i in range(s.n_c):
                cal_h[r][(f, lv, 900_000 + i)] = int(runs[r, k, i])
                cal_d[r][(f, 900_000 + i, lv)] = int(runs[r, k, i])
    h, _achieved = cell_threshold_h(cal_h, s.n_c, n_boot=s.n_boot)
    delta = ps.delta_from_outcomes(cal_d, 6 * s.n_c, n_boot=s.n_boot)
    pairs = {
        CELL_NAMES[l1].split("/")[0]: _pair_counts(runs[:, l1].ravel(), runs[:, l2].ravel())
        for l1, l2 in PAIRS
    }
    return {"h": h, "delta": delta, "pairs": pairs}


def _bounds(w: IntArray, loss: IntArray, n: int) -> tuple[FloatArray, FloatArray]:
    tango = np.vectorize(lambda a, b: sim.tango_lower(int(a), int(b), n, ALPHA), otypes=[float])
    exact = np.vectorize(
        lambda a, b: sim.exact_conditional_lower(int(a), int(b), n, ALPHA), otypes=[float]
    )
    return tango(w, loss), exact(w, loss)


def _gate_rows(
    d: IntArray, s0: BoolArray, s: Setup, counts: Sequence[IntArray], cal: Mapping[str, Any],
    true: Sequence[tuple[Fraction, ...]], with_demo: bool,
) -> tuple[dict[str, dict[Fraction, BoolArray]], dict[Fraction, dict[str, int]]]:  # fmt: skip
    """한 (시나리오, 의존) 조합의 후보별 채택(게이트 -> margin -> bool 배열)과 칸 하한 coverage."""
    n, k = s.n_c, d.shape[1]
    dd = d.reshape(6, n, k)
    w = (dd == 1).sum(axis=1)
    loss = (dd == -1).sum(axis=1)
    s0_cells = s0.reshape(6, n, -1).sum(axis=1)  # (6, 1) 또는 (6, k)
    n_s_cells = np.broadcast_to(s0_cells, (6, k))
    c4 = c4_fast(d, counts, cal["delta"])
    h: Fraction = cal["h"]
    c5 = c4 & (dd.sum(axis=1) * h.denominator >= -h.numerator * n).all(axis=0)
    tango, exact = _bounds(w, loss, n)
    pct = percentile_lowers(d, counts, n, ALPHA) if with_demo else None
    big_w, big_l, big_s = w.sum(axis=0), loss.sum(axis=0), n_s_cells.sum(axis=0)
    sage = {
        lam: np.asarray(
            [
                sim.sage_accept(
                    int(big_w[j]), int(big_l[j]), int(big_s[j]), lam=lam, tau=SAGE_TAU, alpha=ALPHA
                )
                for j in range(k)
            ]
        )
        for lam in SAGE_LAMBDAS
    }
    cells_ok = np.asarray(
        [
            sim.sage_cells_ok(loss[:, j].tolist(), n_s_cells[:, j].tolist(), tau=SAGE_TAU)
            for j in range(k)
        ]
    )
    credit = np.asarray(
        [sim.credit_accept(int(big_w[j]), int(big_l[j]), 6 * n, int(big_s[j]), eps=CREDIT_EPS)
         for j in range(k)]
    )  # fmt: skip
    gates: dict[str, dict[Fraction, BoolArray]] = {}
    cover: dict[Fraction, dict[str, int]] = {}
    true_arr = np.asarray([[float(x) for x in t] for t in true]).T  # (6, k)
    for m in s.margins:
        fm = float(m)
        per = {
            "C4": c4,
            "C5": c5,
            "C6_IUT": c4 & (tango > -fm).all(axis=0),
            "C6_IUT_exact": c4 & (exact > -fm).all(axis=0),
            **{f"SAGE_l{lam}": sage[lam] for lam in SAGE_LAMBDAS},
            **{f"SAGE_cell_l{lam}": sage[lam] & cells_ok for lam in SAGE_LAMBDAS},
            "credit": credit,
        }
        if pct is not None and m == DEMO_M:
            per["C6_IUT_percentile"] = c4 & (pct > -fm).all(axis=0)
        for g, acc in per.items():
            gates.setdefault(g, {})[m] = np.asarray(acc, dtype=bool)
        cover[m] = {
            "cells": int(tango.size),
            "tango_covered": int((tango <= true_arr + 1e-12).sum()),
            "exact_covered": int((exact <= true_arr + 1e-12).sum()),
            "exact_d_zero": int(((w + loss) == 0).sum()),
        }
    return gates, cover


def _shifted(rates: FloatArray, shifts: FloatArray) -> FloatArray:
    """rates (6, n), shifts (6,) 또는 (6, k) -> (6, n, k) 또는 (6, n, 1)."""
    sh = shifts if shifts.ndim == 2 else shifts[:, None]
    out: FloatArray = np.clip(rates[:, :, None] + sh[:, None, :], 0.0, 1.0)
    return out


def evaluate_pool(s: Setup, pool: int) -> dict[str, Any]:
    """풀 하나의 모든 조건. 반환은 칸 키 (m, 시나리오, 배치, 의존, 게이트)별 개수."""
    rng = np.random.default_rng(np.random.SeedSequence([0, s.baseline_index, s.n_c, pool]))
    pops = s.baseline.arrays()
    k = s.candidates
    cal = _calibration(rng, s, pops)
    dev = _draw_cases(rng, pops, s.n_c)
    s0_dev = uniforms(rng.standard_normal((6, s.n_c)), coupled=s.coupled) < dev
    s0_ind = uniforms(rng.standard_normal((6, s.n_c, k)), coupled=s.coupled) < dev[:, :, None]
    u_dev = uniforms(rng.standard_normal((6, s.n_c, k)), coupled=s.coupled)
    orc = _draw_cases(rng, pops, ORACLE_PER_CELL)
    or_s0 = np.stack(
        [uniforms(rng.standard_normal((6, ORACLE_PER_CELL)), coupled=s.coupled) < orc
         for _ in range(S0_REPEATS)]
    )  # fmt: skip
    u_or = uniforms(rng.standard_normal((6, ORACLE_PER_CELL, k)), coupled=s.coupled)
    picks = rng.integers(0, len(s.v1_effects), k)
    group_picks = rng.integers(0, len(s.v1_effects), len(s.group_sizes))
    counts = _counts_for(s.n_c, s.n_boot)
    or_counts = _counts_for(ORACLE_PER_CELL, s.n_boot)
    maj = (or_s0.sum(axis=0) >= 3).astype(np.int64).reshape(-1)

    def per_candidate(effects: Sequence[tuple[Fraction, ...]]) -> list[tuple[Fraction, ...]]:
        return [_true_effect(s.baseline, eff) for eff in effects]

    combos: list[
        tuple[str, str, Fraction | None, list[tuple[Fraction, ...]], list[tuple[Fraction, ...]]]
    ] = []
    seen: set[tuple[str, str]] = set()
    for m in s.margins:
        for sc in build_scenarios(s.baseline, m):
            if not sc.feasible:
                continue
            if sc.m is None and sc.key in seen:
                continue
            seen.add(sc.key)
            combos.append((sc.name, sc.placement or "-", sc.m, [sc.shifts] * k, [sc.deltas] * k))
    emp = [s.v1_effects[i] for i in picks]
    combos.append(("empirical", "-", None, emp, per_candidate(emp)))
    grouped = [
        s.v1_effects[g]
        for g, size in zip(group_picks, s.group_sizes, strict=True)
        for _ in range(size)
    ]
    combos.append(("empirical_groups", "-", None, grouped, per_candidate(grouped)))

    gates_out: dict[tuple[str, ...], dict[str, Any]] = {}
    cover_out: dict[tuple[str, ...], dict[str, int]] = {}
    oracle_out: dict[tuple[str, ...], dict[str, int]] = {}
    for name, placement, sc_m, shifts, true in combos:
        sh = np.asarray([[float(x) for x in e] for e in shifts]).T  # (6, k)
        cand_dev = (u_dev < _shifted(dev, sh)).astype(np.int64)
        cand_or = (u_or < _shifted(orc, sh)).astype(np.int64)
        d_or = cand_or.reshape(-1, k) - maj[:, None]
        unc = _oracle_uncertain(d_or, or_counts)
        deps = ["idea_groups"] if name == "empirical_groups" else ["main", "s0_independent"]
        scen = "empirical" if name == "empirical_groups" else name
        for dep in deps:
            s0 = s0_ind if dep == "s0_independent" else s0_dev[:, :, None]
            d = (cand_dev - s0.astype(np.int64)).reshape(-1, k)
            demo = s.demo and scen == "null_all_cells" and dep == "main"
            gates, cover = _gate_rows(d, s0, s, counts, cal, true, demo)
            for m in s.margins if sc_m is None else (sc_m,):
                ms = str(m)
                bad = np.asarray([inadmissible(t, sum(t, Fraction(0)) / 6, m) for t in true])
                good = np.asarray([useful(t, sum(t, Fraction(0)) / 6, m) for t in true])
                gain = np.asarray([float(sum(t, Fraction(0)) / 6) for t in true])
                for g, by_m in gates.items():
                    if m not in by_m:
                        continue
                    acc = by_m[m]
                    gates_out[(ms, scen, placement, dep, g)] = {
                        "n": k,
                        "inadmissible": int(bad.sum()),
                        "accepted": int(acc.sum()),
                        "accepted_inadmissible": int((acc & bad).sum()),
                        "useful": int(good.sum()),
                        "accepted_useful": int((acc & good).sum()),
                        "gain": float((acc * gain).sum()),
                    }
                cover_out[(ms, scen, placement, dep)] = cover[m]
                oracle_out[(ms, scen, placement, dep)] = {"uncertain": unc, "n": k}
    return {
        "gates": gates_out,
        "coverage": cover_out,
        "oracle": oracle_out,
        "calibration": {"h": str(cal["h"]), "delta": str(cal["delta"]), "pairs": cal["pairs"]},
    }


# 집계와 판정 ------------------------------------------------------------------------------------


def rate(pools: Sequence[Mapping[str, Any]], num: str, den: str) -> dict[str, Any]:
    """풀 단위 비율. 분자는 개수 또는 이득 합(실수)이라 자르지 않는다(2026-11-26 수정)."""
    events = [p[num] for p in pools]
    eligible = [int(p[den]) for p in pools]
    estimate, se = sim.mc_se(events, eligible)
    return {"estimate": estimate, "se": se, "events": sum(events), "eligible": sum(eligible)}


def judge(rows: Mapping[str, Any], *, n_c: int, candidates: int = 40) -> dict[str, Any]:
    """점추정 판정과 borderline(어느 기준이든 경계에서 2 MC SE 안)."""
    fa, power = rows["false_accept"], rows["useful_power"]
    episodes = (candidates + 1) * 6 * n_c

    def near(r: Mapping[str, Any], threshold: float) -> bool:
        return bool(abs(r["estimate"] - threshold) < BORDERLINE_SE * r["se"])

    meets = (
        fa["estimate"] <= FALSE_ACCEPT_AT_MOST
        and power["estimate"] >= USEFUL_POWER_AT_LEAST
        and episodes <= EPISODES_AT_MOST
    )
    borderline = near(fa, FALSE_ACCEPT_AT_MOST) or near(power, USEFUL_POWER_AT_LEAST)
    return {"episodes": episodes, "meets": bool(meets), "borderline": bool(borderline)}
