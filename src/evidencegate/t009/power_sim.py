"""Ticket 009 §2.5 검정력 점검(계획용, LLM 없음). v1 공개 결과만 쓴다.

    uv run python -m evidencegate.t009.power_sim --pools 1000 --boot 2000 \\
        --out plans/ticket-009-power.json

모의 풀 하나:
- 보정: 칸마다 v1 Oracle 케이스별 성공률 p̂_i(반복 5회 평균)에서 35개를 복원 추출하고,
  S0 x 반복 5를 Bernoulli로 모의한다(실제 보정 기록은 쓰지 않음). 여기서 h(35)·h(70)
  (§2.2, `rq9.cell_threshold_h`)과 δ(210)·δ(420)(v1 알고리즘)을 낸다.
- Dev-B: 칸마다 p̂에서 70개를 복원 추출한다. 칸당 35 대안은 각 칸의 앞 35개다(공통 난수).
- Oracle: 칸마다 p̂에서 50개, S0 x 5 다수결, τ_cell은 `oracle.tau_cells`.
- 후보: v1 proposer 12개의 칸별 oracle Δ_c 중 하나를 복원 추출해 Δ_c x 축소(1 또는 0.5)를 더한
  성공률 clip(p̂_i + Δ_c, 0, 1)로 Bernoulli. 같은 균등 난수를 두 축소에 쓴다(공통 난수).
- 귀무 후보(Δ = 0)로 Local Harm·Severe 귀무 라벨률을 낸다.
- 판정: C4(`gates.gate_c4`와 같음), C5 = C4 그리고 칸 조건, Severe(`rq9`와 같음),
  Local Harm·Useful(`oracle`과 같음). 같음은 tests/t009/test_power_sim.py가 확인한다.
  후보 여러 개를 한 번에 판정하려고 bootstrap 색인의 개수 행렬을 곱한다
  (색인은 실제 함수와 같은 `stratified_bootstrap_indices(…, seed 0)`).
"""

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import numpy.typing as npt

from evidencegate.eval_constants import C4_Q, EPSILON, N_BOOT, NOISE_Q, ORACLE_CI_QS, SEED
from evidencegate.eval_data import CELLS, CaseId, Cell
from evidencegate.exact import IndexArray, percentile_exact, stratified_bootstrap_indices
from evidencegate.oracle import tau_cells
from evidencegate.t009.rq9 import SEVERE_Q, Rq95Row, cell_threshold_h, rq9_1, rq9_2, rq9_3
from evidencegate.t009.rq9 import rq9_5 as rq9_5_rule

IntArray = npt.NDArray[np.int64]
DEV_SIZES = (35, 70)
SHRINKS = (1.0, 0.5)
CAL_PER_CELL = 35  # 기본값(2026-11-04-04 대안 비교). 확정 설정은 70(--cal-per-cell 70)
ORACLE_PER_CELL = 50
SIM_SEED = 20261104  # 모의실험 난수(계획용)


@dataclass(frozen=True, order=True)
class Scenario:
    dev_b_per_cell: int
    shrink: float


# --- 실제 함수와 같은 판정(여러 후보를 한 번에) ------------------------------------------------


def _counts(block: IndexArray) -> IntArray:
    """(n_boot, size) 색인 -> (n_boot, size) 등장 횟수."""
    n_boot, size = block.shape
    flat = (block + np.arange(n_boot)[:, None] * size).ravel()
    return np.bincount(flat, minlength=n_boot * size).reshape(n_boot, size).astype(np.int64)


def _totals(d: IntArray, blocks: Sequence[IndexArray]) -> list[IntArray]:
    """칸마다 재표본 합계 (n_boot, 후보)."""
    out, start = [], 0
    for block in blocks:
        size = block.shape[1]
        out.append(_counts(block) @ d[start : start + size])
        start += size
    return out


def _quantile_scaled(totals: IntArray, q: Fraction) -> tuple[IntArray, int]:
    """정렬된 합계의 선형 보간 백분위수 x den (정수), den. `gates._percentile_of_totals`와 같다."""
    ordered = np.sort(totals, axis=0)
    position = q * (ordered.shape[0] - 1)
    lo = math.floor(position)
    hi = min(lo + 1, ordered.shape[0] - 1)
    frac = position - lo
    return ordered[lo] * (frac.denominator - frac.numerator) + ordered[hi] * frac.numerator, int(
        frac.denominator
    )


def c4_parts(
    d: IntArray, blocks: Sequence[IndexArray], delta: Fraction
) -> tuple[npt.NDArray[np.bool_], IntArray, int, IntArray]:
    """(채택, 하한 x den x N 정수, den, 평균 Δ x N 정수). 하한 = lower_v / (den * N)."""
    n = d.shape[0]
    totals = sum(_totals(d, blocks))
    v, den = _quantile_scaled(np.asarray(totals), C4_Q)
    sums = d.sum(axis=0)
    mean_ok = sums * delta.denominator > delta.numerator * n
    return np.asarray((v > 0) & mean_ok), np.asarray(v), den, np.asarray(sums)


def c4_many(d: IntArray, blocks: Sequence[IndexArray], delta: Fraction) -> npt.NDArray[np.bool_]:
    return c4_parts(d, blocks, delta)[0]


def cell_sums(d: IntArray, n_per_cell: int) -> IntArray:
    sums: IntArray = d.reshape(6, n_per_cell, d.shape[1]).sum(axis=1)
    return sums


def c5_many(
    c4: npt.NDArray[np.bool_], d: IntArray, n_per_cell: int, h: Fraction
) -> npt.NDArray[np.bool_]:
    ok = (cell_sums(d, n_per_cell) * h.denominator >= -h.numerator * n_per_cell).all(axis=0)
    return np.asarray(c4 & ok)


def severe_many(
    d: IntArray, blocks: Sequence[IndexArray], tau: Mapping[Cell, Fraction]
) -> npt.NDArray[np.bool_]:
    severe = np.zeros(d.shape[1], dtype=bool)
    for cell, block, totals in zip(CELLS, blocks, _totals(d, blocks), strict=True):
        size = block.shape[1]
        v, den = _quantile_scaled(totals, SEVERE_Q)
        t = tau[cell]
        severe |= v * t.denominator < -t.numerator * size * den
    return severe


def useful_many(d: IntArray, blocks: Sequence[IndexArray]) -> list[bool | None]:
    n = d.shape[0]
    totals = np.asarray(sum(_totals(d, blocks)))
    ends = []
    for q in ORACLE_CI_QS:
        v, den = _quantile_scaled(totals, q)
        ends.append(v * EPSILON.denominator > EPSILON.numerator * n * den)
    return [bool(lo) if lo == hi else None for lo, hi in zip(ends[0], ends[1], strict=True)]


def local_harm_many(
    d: IntArray, n_per_cell: int, tau: Mapping[Cell, Fraction]
) -> npt.NDArray[np.bool_]:
    sums = cell_sums(d, n_per_cell)
    out = np.zeros(d.shape[1], dtype=bool)
    for k, cell in enumerate(CELLS):
        t = tau[cell]
        out |= sums[k] * t.denominator < -t.numerator * n_per_cell
    return out


def delta_from_outcomes(
    outcomes: Mapping[int, Mapping[tuple[str, int, str], int]],
    n_cases: int,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Fraction:
    """`exact.null_diff_quantile_exact`와 같은 추출 순서(케이스 키 (family, seed, level) 정렬)."""
    repeats = sorted(outcomes)
    cases = sorted(outcomes[repeats[0]])
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample = rng.choices(cases, k=n_cases)
        a, b = rng.sample(repeats, 2)
        values.append(abs(Fraction(sum(outcomes[a][c] - outcomes[b][c] for c in sample), n_cases)))
    return percentile_exact(values, NOISE_Q)


# --- 모의 풀 ---------------------------------------------------------------------------------


def _scenarios() -> list[Scenario]:
    return [Scenario(n, s) for n in DEV_SIZES for s in SHRINKS]


def simulate_pool(
    index: int,
    *,
    effects: Mapping[str, Sequence[float]],
    p_hat: Mapping[Cell, Sequence[float]],
    n_cands: int,
    n_null: int,
    n_boot: int,
    seed: int,
    cal_per_cell: int = CAL_PER_CELL,
) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(index + 1)[index])
    names = sorted(effects)
    p = {cell: np.asarray(p_hat[cell], dtype=float) for cell in CELLS}

    # 보정(칸당 35 x 반복 5)
    cal_h: dict[int, dict[CaseId, int]] = {r: {} for r in range(5)}
    cal_d: dict[int, dict[tuple[str, int, str], int]] = {r: {} for r in range(5)}
    for f, lv in CELLS:
        ps = rng.choice(p[(f, lv)], cal_per_cell)
        draws = rng.random((5, cal_per_cell)) < ps
        for r in range(5):
            for i in range(cal_per_cell):
                cal_h[r][(f, lv, 900_000 + i)] = int(draws[r, i])
                cal_d[r][(f, 900_000 + i, lv)] = int(draws[r, i])
    h, achieved = {}, {}
    for n in DEV_SIZES:
        h[n], achieved[n] = cell_threshold_h(cal_h, n, n_boot=n_boot)
    delta = {6 * n: delta_from_outcomes(cal_d, 6 * n, n_boot=n_boot) for n in DEV_SIZES}

    picks = rng.integers(0, len(names), n_cands)
    eff = np.asarray([effects[names[i]] for i in picks], dtype=float).T  # (6, 후보)

    # Dev-B(칸당 70, 35는 앞부분)
    dev_p = np.stack([rng.choice(p[cell], max(DEV_SIZES)) for cell in CELLS])  # (6, 70)
    dev_s0 = rng.random(dev_p.shape) < dev_p
    dev_u = rng.random((6, max(DEV_SIZES), n_cands))

    # Oracle(칸당 50, S0 x 5 다수결)
    or_p = np.stack([rng.choice(p[cell], ORACLE_PER_CELL) for cell in CELLS])
    or_s0 = rng.random((5, *or_p.shape)) < or_p
    s0_repeats = {
        r: {
            (f, lv, 700_000 + i): int(or_s0[r, k, i])
            for k, (f, lv) in enumerate(CELLS)
            for i in range(ORACLE_PER_CELL)
        }
        for r in range(5)
    }
    tau = tau_cells(s0_repeats)
    or_maj = (or_s0.sum(axis=0) >= 3).astype(np.int64).reshape(-1)
    or_u = rng.random((6, ORACLE_PER_CELL, n_cands))
    null_u = rng.random((6, ORACLE_PER_CELL, n_null))
    or_blocks = stratified_bootstrap_indices([ORACLE_PER_CELL] * 6, n_boot, SEED)

    candidates: dict[Scenario, list[tuple[bool, bool, bool, bool, bool | None]]] = {}
    rank: dict[Scenario, list[tuple[int, int]]] = {}  # (평균 Δ x N, 하한 x den x N): RQ9-5 순위용
    for shrink in SHRINKS:
        shift = shrink * eff[:, None, :]
        dev_c = dev_u < np.clip(dev_p[:, :, None] + shift, 0, 1)
        or_c = (or_u < np.clip(or_p[:, :, None] + shift, 0, 1)).astype(np.int64)
        d_or = or_c.reshape(-1, n_cands) - or_maj[:, None]
        severe = severe_many(d_or, or_blocks, tau)
        lh = local_harm_many(d_or, ORACLE_PER_CELL, tau)
        use = useful_many(d_or, or_blocks)
        for n in DEV_SIZES:
            d_dev = (dev_c[:, :n].astype(np.int64) - dev_s0[:, :n, None]).reshape(-1, n_cands)
            blocks = stratified_bootstrap_indices([n] * 6, n_boot, SEED)
            c4, lower_v, _den, sums = c4_parts(d_dev, blocks, delta[6 * n])
            c5 = c5_many(c4, d_dev, n, h[n])
            rank[Scenario(n, shrink)] = [(int(sums[j]), int(lower_v[j])) for j in range(n_cands)]
            candidates[Scenario(n, shrink)] = [
                (bool(c4[j]), bool(c5[j]), bool(severe[j]), bool(lh[j]), use[j])
                for j in range(n_cands)
            ]
    null_c = (null_u < or_p[:, :, None]).astype(np.int64).reshape(-1, n_null)
    d_null = null_c - or_maj[:, None]
    null = {
        "local_harm": int(local_harm_many(d_null, ORACLE_PER_CELL, tau).sum()),
        "severe": int(severe_many(d_null, or_blocks, tau).sum()),
        "n": n_null,
    }
    return {
        "calibration_per_cell": cal_per_cell,
        "h": h,
        "achieved": achieved,
        "delta": delta,
        "tau": tau,
        "candidates": candidates,
        "rank": rank,
        "null": null,
    }


def simulate(
    *,
    effects: Mapping[str, Sequence[float]],
    p_hat: Mapping[Cell, Sequence[float]],
    n_pools: int,
    n_cands: int = 60,
    n_null: int = 200,
    n_boot: int = N_BOOT,
    seed: int = SIM_SEED,
    workers: int = 1,
    cal_per_cell: int = CAL_PER_CELL,
) -> dict[str, Any]:
    kw = dict(
        effects=effects,
        p_hat=p_hat,
        n_cands=n_cands,
        n_null=n_null,
        n_boot=n_boot,
        seed=seed,
        cal_per_cell=cal_per_cell,
    )
    if workers == 1:
        pools = [simulate_pool(b, **kw) for b in range(n_pools)]  # type: ignore[arg-type]
    else:
        with ProcessPoolExecutor(workers) as ex:
            futures = [ex.submit(simulate_pool, b, **kw) for b in range(n_pools)]  # type: ignore[arg-type]
            pools = [f.result() for f in futures]
    return {"scenarios": _scenarios(), "pools": pools}


# --- 요약 ------------------------------------------------------------------------------------

VERDICTS = ("supported", "direction_only", "not_supported", "undecidable")


def summarize(
    sim: Mapping[str, Any],
    ns: Sequence[int] = (40, 48, 60),
    *,
    clean_useful: bool = False,
    rq9_5: bool = False,
) -> dict[Scenario, Any]:
    """clean_useful이면 RQ9-2를 깨끗한 Useful(확정 Useful 그리고 Severe 아님) 기준으로 판정하고,
    모든 Useful 기준은 `rq9_2_all_useful`로 함께 낸다(2026-11-04 확정 정의). 기본은 모든 Useful."""
    out: dict[Scenario, Any] = {}
    for sc in sim["scenarios"]:
        out[sc] = {}
        for n in ns:
            keys = ("rq9_1", "rq9_2", "rq9_3") + (("rq9_2_all_useful",) if clean_useful else ())
            tallies: dict[str, Counter[str]] = {r: Counter() for r in keys}
            five: dict[str, list[int]] = {k: [] for k in RQ9_5_MEDIANS}
            if rq9_5:
                tallies["rq9_5"] = Counter()
                tallies["rq9_5_with_test"] = Counter()
            c4n, fn, kn = [], [], []
            for pool in sim["pools"]:
                rows = pool["candidates"][sc][:n]
                if rq9_5:
                    r5 = _rq9_5_pool(rows, pool["rank"][sc][:n])
                    # 출력 키는 초안 때 이름을 유지한다(계획 파일 바이트 고정): rq9_5는 개수 규칙,
                    # rq9_5_with_test는 판정(6.5)
                    tallies["rq9_5"][r5["verdict_count"]] += 1
                    tallies["rq9_5_with_test"][r5["verdict"]] += 1
                    for k, src in RQ9_5_MEDIANS.items():
                        five[k].append(r5[src])
                acc = [r for r in rows if r[0]]
                one = rq9_1([(c5, severe) for _, c5, severe, _, _ in acc])
                filtered = any(not r[1] for r in acc)
                useful_c4 = sum(1 for r in acc if r[4] is True)
                u = sum(1 for r in acc if r[4] is True and not r[1])
                all_useful = rq9_2(useful_c4, u, any_filtered=filtered)
                if clean_useful:
                    clean = [r for r in acc if r[4] is True and not r[2]]
                    two = rq9_2(
                        len(clean), sum(1 for r in clean if not r[1]), any_filtered=filtered
                    )
                    tallies["rq9_2_all_useful"][all_useful["verdict"]] += 1
                else:
                    two = all_useful
                three = rq9_3(severe_accepted=sum(1 for r in acc if r[2]), accepted=len(acc))
                tallies["rq9_1"][one["verdict"]] += 1
                tallies["rq9_2"][two["verdict"]] += 1
                tallies["rq9_3"][three["verdict"]] += 1
                c4n.append(len(acc))
                fn.append(one["f"])
                kn.append(one["k"])
            out[sc][n] = {
                **{r: {v: t[v] for v in VERDICTS} for r, t in tallies.items()},
                "median_c4_accepted": median(c4n),
                "median_filtered": median(fn),
                "median_severe_in_c4": median(kn),
                **({k: median(v) for k, v in five.items()} if rq9_5 else {}),
            }
    return out


RQ9_5_MEDIANS = {
    "median_m": "m",
    "median_d": "d",
    "median_severe_c5": "severe_c5",
    "median_severe_c4_star": "severe_c4_star",
    "median_clean_useful_c5": "clean_useful_c5",
    "median_clean_useful_c4_star": "clean_useful_c4_star",
}


def _rq9_5_pool(
    rows: Sequence[tuple[bool, bool, bool, bool, bool | None]], rank: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    acc = [
        Rq95Row(
            f"{j:03d}",
            Fraction(rank[j][0]),
            Fraction(rank[j][1]),
            r[1],
            r[2],
            r[4] is True and not r[2],
        )
        for j, r in enumerate(rows)
        if r[0]
    ]
    return rq9_5_rule(acc)


def h_summary(sim: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in DEV_SIZES:
        hs = sorted(float(pool["h"][n]) for pool in sim["pools"])
        ach = [float(pool["achieved"][n]) for pool in sim["pools"]]
        out[str(n)] = {
            "h_median": median(hs),
            "h_p05": hs[int(0.05 * (len(hs) - 1))],
            "h_p95": hs[int(0.95 * (len(hs) - 1))],
            "achieved_false_rejection_median": median(ach),
        }
    nl = sum(p["null"]["local_harm"] for p in sim["pools"])
    ns = sum(p["null"]["severe"] for p in sim["pools"])
    total = sum(p["null"]["n"] for p in sim["pools"])
    out["null_label_rate"] = {"local_harm": nl / total, "severe": ns / total, "n": total}
    return out


# --- v1 입력 ---------------------------------------------------------------------------------

V1_ORACLE_S0 = Path(
    "runs/oracle/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/bf65e569ae37.jsonl"
)
V1_ANALYSIS = Path("results/ticket-005/analysis.json")


def load_v1_inputs(
    root: Path = Path("."),
) -> tuple[dict[str, tuple[float, ...]], dict[Cell, list[float]]]:
    analysis = json.loads((root / V1_ANALYSIS).read_text())
    effects = {
        name: tuple(
            float(Fraction(c["oracle"]["correct_disposition"]["cells"][f"{f}/{lv}"]["exact"]))
            for f, lv in CELLS
        )
        for name, c in analysis["candidates"].items()
        if c["group"] == "proposer"
    }
    per_case: dict[CaseId, list[int]] = {}
    for line in (root / V1_ORACLE_S0).read_text().splitlines():
        r = json.loads(line)
        k = r["key"]
        per_case.setdefault((k["family"], k["level"], k["incident_seed"]), []).append(
            int(bool(r["correct_disposition"]))
        )
    p_hat = {
        cell: [sum(v) / len(v) for c, v in sorted(per_case.items()) if c[:2] == cell]
        for cell in CELLS
    }
    return effects, p_hat


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-power-009")
    parser.add_argument("--pools", type=int, default=1000)
    parser.add_argument("--boot", type=int, default=N_BOOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cal-per-cell", type=int, default=CAL_PER_CELL)
    parser.add_argument("--clean-useful", action="store_true", help="RQ9-2 on clean Useful")
    parser.add_argument("--ns", default="40,48,60", help="comma-separated candidate counts")
    parser.add_argument("--rq9-5", action="store_true", help="add the RQ9-5 draft summary")
    args = parser.parse_args(argv)
    ns = tuple(int(x) for x in args.ns.split(","))
    effects, p_hat = load_v1_inputs()
    sim = simulate(
        effects=effects,
        p_hat=p_hat,
        n_pools=args.pools,
        n_boot=args.boot,
        workers=args.workers,
        cal_per_cell=args.cal_per_cell,
    )
    table = summarize(sim, ns, clean_useful=args.clean_useful, rq9_5=args.rq9_5)
    # 기본값이 아닌 설정만 적는다(기본 명령의 출력은 이전 파일과 바이트가 같다)
    extra: dict[str, Any] = {}
    if args.cal_per_cell != CAL_PER_CELL:
        extra["calibration_per_cell"] = args.cal_per_cell
    if args.clean_useful:
        extra["rq9_2_basis"] = "clean useful (definite Useful and not Severe); all Useful reported"
    if ns != (40, 48, 60):
        extra["ns"] = list(ns)
    if args.rq9_5:
        extra["rq9_5"] = "draft: C5 vs the size-matched strict mean gate C4* (report 2026-11-05-02)"
    result = {
        "settings": {
            "pools": args.pools,
            "boot": args.boot,
            "candidates_per_pool": 60,
            "null_candidates_per_pool": 200,
            "sim_seed": SIM_SEED,
            "inputs": {"analysis": str(V1_ANALYSIS), "oracle_s0": str(V1_ORACLE_S0)},
            **extra,
        },
        "h": h_summary(sim),
        "table": {
            f"dev_b_{sc.dev_b_per_cell}_shrink_{sc.shrink}": {str(n): v for n, v in rows.items()}
            for sc, rows in table.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["h"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
