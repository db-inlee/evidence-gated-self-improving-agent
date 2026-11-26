"""Ticket 014 계획: 대칭 Oracle의 규모(셀당 n_o, 반복 R)별 라벨 불확실성. LLM 없음.

    uv run python -m evidencegate.t014.oracle_scale --pools 1000 --workers 6 \
        --out plans/ticket-014-oracle-scale.json

- Ticket 013 모의의 부품(`t013.sim_core`: 기준선 a의 케이스 집단, 시나리오, 이동량, 개수
  행렬)을 쓴다.
- 대칭 Oracle: 후보와 S0가 같은 반복 수 R과 같은 집계를 쓴다. 집계는 mean(반복 평균, 단일
  실행 정책의 기대 차이) 또는 majority(R이 홀수일 때 다수결). 짝은 independent(후보와 S0의
  난수 독립) 또는 same_seed(반복 r에서 후보가 S0와 같은 균등 난수를 씀, seed 공유의 상한 가정).
- 라벨(셀 층화 paired bootstrap, 기존 Oracle과 같은 백분위수):
  - useful_uncertain: 전체 구간이 +ε를 가로지름(`oracle.useful`가 None).
  - effect_uncertain: 전체 구간이 +ε 또는 -ε를 가로지름(`oracle.classify_effect`가 None).
  - severe: 어떤 칸의 상한(39/40)이 -τ보다 작음. τ = ε(대칭 R = 1에서는 반복 잡음을 잴 수 없음).
  - severe_uncertain: severe가 아니고, 어떤 칸의 95% 구간이 -τ를 가로지름.
- 결과는 계획용 표다(판정에 쓰지 않음). 규모는 David가 정한다.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import EPSILON, ORACLE_CI_QS
from evidencegate.t009 import power_sim as ps
from evidencegate.t009.rq9 import SEVERE_Q
from evidencegate.t013 import sim
from evidencegate.t013 import sim_core as sc

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

M_D = Fraction(1, 20)  # Ticket 014의 m_c(David 결정, 2026-11-26)
TAU = EPSILON
CANDIDATES, CONTROLS = 40, 7
# 009b 본 실행 처리량(writeup/inputs/run_summary_009b.json, 테스트로 대조)
EPISODES_009B = 32820
WALL_S_009B = 275636.257
ACTIVE_S_009B = 218799.339
WALL_S_PER_EPISODE = WALL_S_009B / EPISODES_009B
ACTIVE_S_PER_EPISODE = ACTIVE_S_009B / EPISODES_009B
GRID_N_O = (50, 100, 150)
GRID_R = (1, 3)


@dataclass(frozen=True)
class Condition:
    n_o: int
    repeats: int
    aggregation: str  # mean / majority
    pairing: str  # independent / same_seed

    @property
    def key(self) -> str:
        return f"n{self.n_o}_r{self.repeats}_{self.aggregation}_{self.pairing}"


def oracle_episodes(*, n_o: int, repeats: int) -> int:
    """후보 40 + 대조 7 + S0, 셀 6, 셀당 n_o, 반복 R."""
    return (CANDIDATES + CONTROLS + 1) * 6 * n_o * repeats


def hours(episodes: int) -> dict[str, float]:
    return {
        "active": episodes * ACTIVE_S_PER_EPISODE / 3600,
        "wall": episodes * WALL_S_PER_EPISODE / 3600,
    }


def draw_uniforms(
    rng: np.random.Generator, *, repeats: int, cases: int, candidates: int, pairing: str
) -> tuple[FloatArray, FloatArray]:
    """(S0 (R, cases), 후보 (R, cases, K)). same_seed이면 후보가 S0의 난수를 그대로 쓴다."""
    u_s0 = rng.random((repeats, cases))
    if pairing == "same_seed":
        u_c = np.broadcast_to(u_s0[:, :, None], (repeats, cases, candidates)).copy()
    elif pairing == "independent":
        u_c = rng.random((repeats, cases, candidates))
    else:
        raise ValueError(f"unknown pairing {pairing!r}")
    return u_s0, u_c


def case_differences(cand: IntArray, s0: IntArray, *, aggregation: str) -> tuple[IntArray, int]:
    """cand (R, cases, K), s0 (R, cases) -> (케이스별 정수 차이 (cases, K), 배율)."""
    r = cand.shape[0]
    if aggregation == "mean":
        return cand.sum(axis=0) - s0.sum(axis=0)[:, None], r
    if aggregation == "majority":
        if r % 2 == 0:
            raise ValueError(f"majority needs an odd number of repeats, got {r}")
        c = (2 * cand.sum(axis=0) > r).astype(np.int64)
        s = (2 * s0.sum(axis=0) > r).astype(np.int64)
        return c - s[:, None], 1
    raise ValueError(f"unknown aggregation {aggregation!r}")


def labels(d: IntArray, *, scale: int, n_o: int, n_boot: int) -> dict[str, BoolArray]:
    """d (6 n_o, K) 정수 차이(실제 차이 = d / scale)의 라벨."""
    counts = sc._counts_for(n_o, n_boot)
    per_cell = sc._totals(d, counts)  # 칸마다 (n_boot, K) 합
    totals = np.asarray(sum(per_cell))
    n = d.shape[0]
    eps = EPSILON

    def above(total_v: IntArray, den: int, size: int, x: Fraction) -> BoolArray:
        # total_v / (den * size * scale) > x
        return np.asarray(total_v * x.denominator > x.numerator * size * scale * den)

    ends = [ps._quantile_scaled(totals, q) for q in ORACLE_CI_QS]
    up = [above(v, den, n, eps) for v, den in ends]
    down = [~above(v, den, n, -eps) & ~_equal(v, den, n, scale, -eps) for v, den in ends]
    useful_unc = up[0] != up[1]
    effect_unc = useful_unc | (down[0] != down[1])
    severe = np.zeros(d.shape[1], dtype=bool)
    crosses = np.zeros(d.shape[1], dtype=bool)
    for cell_totals in per_cell:
        hi_v, hi_den = ps._quantile_scaled(cell_totals, SEVERE_Q)
        lo_v, lo_den = ps._quantile_scaled(cell_totals, ORACLE_CI_QS[0])
        upper_below = ~above(hi_v, hi_den, n_o, -TAU) & ~_equal(hi_v, hi_den, n_o, scale, -TAU)
        lower_below = ~above(lo_v, lo_den, n_o, -TAU) & ~_equal(lo_v, lo_den, n_o, scale, -TAU)
        severe |= upper_below
        crosses |= lower_below & ~upper_below
    return {
        "useful_uncertain": np.asarray(useful_unc, dtype=bool),
        "effect_uncertain": np.asarray(effect_unc, dtype=bool),
        "severe": severe,
        "severe_uncertain": crosses & ~severe,
    }


def _equal(v: IntArray, den: int, size: int, scale: int, x: Fraction) -> BoolArray:
    return np.asarray(v * x.denominator == x.numerator * size * scale * den)


def evaluate(
    baseline: sc.Baseline,
    cond: Condition,
    *,
    v1_effects: Sequence[tuple[Fraction, ...]],
    pools: int,
    candidates: int = CANDIDATES,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    pops = baseline.arrays()
    scenarios = [s for s in sc.build_scenarios(baseline, M_D) if s.feasible]
    keys = [f"{s.name}|{s.placement or '-'}" for s in scenarios] + ["empirical|-"]
    tallies: dict[str, dict[str, list[int]]] = {k: {} for k in keys}
    ss = np.random.SeedSequence([seed, cond.n_o, cond.repeats])
    for pool_seed in ss.spawn(pools):
        rng = np.random.default_rng(pool_seed)
        rates = sc._draw_cases(rng, pops, cond.n_o)  # (6, n_o)
        flat = rates.reshape(-1)
        u_s0, u_c = draw_uniforms(
            rng, repeats=cond.repeats, cases=flat.size, candidates=candidates, pairing=cond.pairing
        )
        picks = rng.integers(0, len(v1_effects), candidates)
        s0 = (u_s0 < flat[None, :]).astype(np.int64)
        shifts_by_key = [np.asarray([float(x) for x in s.shifts]) for s in scenarios]
        emp = np.asarray([[float(x) for x in v1_effects[i]] for i in picks]).T  # (6, K)
        for key, shift in zip(keys, [*shifts_by_key, emp], strict=True):
            sh = shift if shift.ndim == 2 else np.broadcast_to(shift[:, None], (6, candidates))
            cand_rate = np.clip(rates[:, :, None] + sh[:, None, :], 0.0, 1.0).reshape(
                -1, candidates
            )
            cand = (u_c < cand_rate[None, :, :]).astype(np.int64)
            d, scale = case_differences(cand, s0, aggregation=cond.aggregation)
            got = labels(d, scale=scale, n_o=cond.n_o, n_boot=n_boot)
            for name, arr in got.items():
                tallies[key].setdefault(name, []).append(int(arr.sum()))
    out = {k: {name: _rate(v, candidates) for name, v in by.items()} for k, by in tallies.items()}
    episodes = oracle_episodes(n_o=cond.n_o, repeats=cond.repeats)
    return {"condition": cond.key, "scenarios": out, "oracle_episodes": episodes}


def _rate(events: Sequence[int], candidates: int) -> dict[str, float]:
    est, se = sim.mc_se(list(events), [candidates] * len(events))
    return {"estimate": est, "se": se}


def _one(
    baseline: sc.Baseline,
    v1_effects: Sequence[tuple[Fraction, ...]],
    pools: int,
    n_boot: int,
    cond: Condition,
) -> dict[str, Any]:
    return evaluate(baseline, cond, v1_effects=v1_effects, pools=pools, n_boot=n_boot)


def merge_parts(paths: Sequence[Path]) -> dict[str, Any]:
    """n_o별로 나눠 돌린 결과를 합친다. 행 외의 설정이 모두 같아야 한다."""
    parts = [json.loads(Path(p).read_text()) for p in paths]
    settings = [
        {k: v for k, v in part.items() if k not in ("rows", "n_o_filter")} for part in parts
    ]
    if any(x != settings[0] for x in settings):
        raise ValueError("parts were made with different settings")
    return settings[0] | {"rows": [row for part in parts for row in part["rows"]]}


def conditions() -> list[Condition]:
    out = []
    for n_o in GRID_N_O:
        for r in GRID_R:
            aggs = ("mean",) if r == 1 else ("mean", "majority")
            for agg in aggs:
                for pairing in ("independent", "same_seed"):
                    out.append(Condition(n_o, r, agg, pairing))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t014-oracle-scale")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--pools", type=int, default=1000)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-o", type=int, default=None, help="only this n_o (a part to merge)")
    parser.add_argument("--repeats", type=int, default=None, help="only this R (a part to merge)")
    parser.add_argument("--merge", type=Path, nargs="*", default=None, help="parts to merge")
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} exists", file=sys.stderr)
        return 1
    if args.merge:
        merged = merge_parts(args.merge)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
        return 0
    inputs = sim.load_inputs(args.root, gate={})
    baseline = inputs.baselines[0]
    conds = [
        c
        for c in conditions()
        if (args.n_o is None or c.n_o == args.n_o)
        and (args.repeats is None or c.repeats == args.repeats)
    ]
    if args.workers <= 1:
        rows = [_one(baseline, inputs.v1_effects, args.pools, args.boot, c) for c in conds]
    else:
        from concurrent.futures import ProcessPoolExecutor

        n = len(conds)
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(
                ex.map(_one, [baseline] * n, [inputs.v1_effects] * n, [args.pools] * n,
                       [args.boot] * n, conds)
            )  # fmt: skip
    result = {
        "status": "planning (not a result; no verdict uses it)",
        "baseline": "a (S0 calibration case rates)",
        "m_c": str(M_D),
        "tau": str(TAU),
        "pools": args.pools,
        "n_boot": args.boot,
        "candidates": CANDIDATES,
        "controls": CONTROLS,
        "throughput_009b": {
            "episodes": EPISODES_009B,
            "wall_s": WALL_S_009B,
            "active_s": ACTIVE_S_009B,
        },
        "rows": [r | {"hours": hours(r["oracle_episodes"])} for r in rows],
    }
    if args.n_o is not None or args.repeats is not None:
        result["n_o_filter"] = [args.n_o, args.repeats]
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
