"""Ticket 013 3단계 게이트 모의(동결 계획 `plans/ticket-013-sim-plan.json`, 태그 t013-sim-plan).

    uv run python -m evidencegate.t013.sim gate     # 첫 관문: 009b 사전 등록 9.1 귀무 라벨률 재현
    uv run python -m evidencegate.t013.sim timing   # 본 실행 소요 시간 추정용 소규모 측정

- 입력은 동결 계획 하나다. 실행 시작 때 계획 sha256을 고정값(`PLAN_SHA256`)과 옆 `.sha256`으로
  확인하고, 다르면 멈춘다. 계획은 고치지 않는다.
- 판정 부품: Tango score 하한(주), 정확 조건부 하한(보조), 칸 percentile bootstrap 하한(시연),
  SAGE(원문 규칙의 한 라운드판), 칸별 SAGE 상한, 신용형 게이트, 풀 단위 MC SE. C4·C5·Oracle
  라벨은 `evidencegate.t009.power_sim`의 함수(실제 함수와 같음이 테스트로 확인됨)를 그대로 쓴다.
- 본 모의(조건 반복과 `results/ticket-013/sim.json` 한 번 쓰기)는 계획에 없는 선택이 정해진 뒤에
  붙인다(보고서 2026-11-25-02).
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import N_BOOT, SEED
from evidencegate.eval_data import CELLS, CaseId, Cell
from evidencegate.exact import clopper_pearson, stratified_bootstrap_indices
from evidencegate.oracle import tau_cells
from evidencegate.t009 import power_sim as ps

PLAN_PATH = Path("plans/ticket-013-sim-plan.json")
PLAN_SHA256 = "8c97f64145d8f9b79cb7cb7591fafbd4e1d4895f57c8f3c05e7d797db3db43a0"
POWER_009_FINAL = Path("plans/ticket-009-power-final.json")
REGISTERED_NULL = {"local_harm": 0.167205, "severe": 0.000475}  # 9.1, POWER_009_FINAL
V1_ORACLE_S0 = ps.V1_ORACLE_S0
V1_ORACLE_S0_SHA256 = "737365eab4eb99403caf945ca82095f35c4ea5900a9b9408cea1bcf654dcdab5"
V1_RUN_SUMMARY = Path("writeup/inputs/run_summary.json")  # V1_ORACLE_S0의 sha256을 담은 커밋 파일
GATE_POOLS, GATE_NULL = 1000, 200  # 9.1과 같은 귀무 후보 200,000개
ORACLE_PER_CELL = ps.ORACLE_PER_CELL
GATE_SE_MULTIPLE = 2 * math.sqrt(2)  # 보고서 2026-11-25-02 §0: 차이의 SE = √2 x SE_new, 2 SE

IntArray = npt.NDArray[np.int64]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(root: Path) -> dict[str, Any]:
    """동결 계획을 읽는다. 고정 sha256과 옆 `.sha256` 둘 다 맞아야 한다."""
    path = root / PLAN_PATH
    actual = _sha(path)
    recorded = path.with_name(path.name + ".sha256").read_text().split()[0]
    if actual != PLAN_SHA256 or recorded != PLAN_SHA256:
        raise ValueError(f"{PLAN_PATH} sha256 {actual} does not match the frozen {PLAN_SHA256}")
    plan: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if plan["status"] != "frozen" or plan["open_before_freeze"]:
        raise ValueError(f"{PLAN_PATH} is not frozen")
    return plan


# 칸 하한 ---------------------------------------------------------------------------------------


def _z(alpha: Fraction) -> float:
    return NormalDist().inv_cdf(1 - float(alpha))


def tango_z(w: int, loss: int, n: int, delta0: Fraction | float) -> float:
    """Tango(1998) score 통계량. w = 후보 성공·S0 실패, loss = 후보 실패·S0 성공, Δ0 = 귀무 차이."""
    d0 = float(delta0)
    a = 2 * n
    b = -w - loss + (2 * n - w + loss) * d0
    c = -loss * d0 * (1 - d0)
    q21 = (math.sqrt(max(b * b - 4 * a * c, 0.0)) - b) / (2 * a)
    var = n * (2 * q21 + d0 * (1 - d0))
    if var <= 0:
        return math.inf if w - loss - n * d0 > 0 else (0.0 if w - loss - n * d0 == 0 else -math.inf)
    return (w - loss - n * d0) / math.sqrt(var)


@cache
def tango_lower(w: int, loss: int, n: int, alpha: Fraction) -> float:
    """한쪽 수준 alpha의 Tango score 하한: Z(Δ0) = z_(1-alpha)인 Δ0(Z는 Δ0에 대해 감소)."""
    z, hi, lo = _z(alpha), (w - loss) / n, -1.0 + 1e-12
    if tango_z(w, loss, n, lo) <= z:
        return -1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if tango_z(w, loss, n, mid) > z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@cache
def exact_conditional_lower(w: int, loss: int, n: int, alpha: Fraction) -> float:
    """불일치 d = w + l을 조건으로 q = w / d의 한쪽 정확 하한, Δ 하한 = (d / n)(2 q_lower - 1)."""
    d = w + loss
    if d == 0:
        return 0.0
    interval = clopper_pearson(w, d, 2 * alpha)
    assert interval is not None
    return (d / n) * (2 * interval[0] - 1)


def percentile_cell_lower(
    d_cell: IntArray, blocks: npt.NDArray[np.int64], alpha: Fraction
) -> float:
    """칸 paired bootstrap 평균의 alpha 백분위수(선형 보간). 시연 조건에만 쓴다."""
    means = np.sort(d_cell[blocks].mean(axis=1))
    pos = float(alpha) * (len(means) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(means) - 1)
    return float(means[lo] + (means[hi] - means[lo]) * (pos - lo))


# SAGE와 신용형 ----------------------------------------------------------------------------------


@cache
def sage_p(w: int, loss: int, lam: int) -> Fraction:
    """P[Bin(w + loss, λ / (1 + λ)) >= w], 정확한 분수. w + loss = 0이면 1."""
    m = w + loss
    if m == 0:
        return Fraction(1)
    q = Fraction(lam, 1 + lam)
    return sum(
        (Fraction(math.comb(m, k)) * q**k * (1 - q) ** (m - k) for k in range(w, m + 1)),
        Fraction(0),
    )


def sage_accept(w: int, loss: int, n_s: int, *, lam: int, tau: float, alpha: Fraction) -> bool:
    """Δ_λ = w - λ loss > 0, p_λ < alpha, rho = loss / n_S <= τ(n_S = 0이면 rho = 0)."""
    rho = loss / n_s if n_s else 0.0
    return w - lam * loss > 0 and sage_p(w, loss, lam) < alpha and rho <= tau


def sage_cells_ok(l_cells: Sequence[int], n_s_cells: Sequence[int], *, tau: float) -> bool:
    """칸마다 rho_c = l_c / n_S,c <= τ(n_S,c = 0이면 rho_c = 0)."""
    return all((loss / n if n else 0.0) <= tau for loss, n in zip(l_cells, n_s_cells, strict=True))


def credit_accept(w: int, loss: int, n: int, n_s: int, *, eps: float) -> bool:
    """Wald 95% 하한 > 0 그리고 C(S0 성공) 안의 새 오류 비율 <= ε(|C| = 0이면 0)."""
    g = (w - loss) / n
    var = (w + loss) / n**2 - (w - loss) ** 2 / n**3
    bound = g - NormalDist().inv_cdf(0.975) * math.sqrt(max(var, 0.0))
    return bound > 0 and (loss / n_s if n_s else 0.0) <= eps


# MC SE -----------------------------------------------------------------------------------------


def mc_se(events: Sequence[int], eligible: Sequence[int]) -> tuple[float, float]:
    """풀 단위 비율 추정량 p = Σa / Σn과 SE = sqrt(R/(R-1) Σ(a - p n)^2) / Σn(계획 monte_carlo)."""
    r, total = len(events), sum(eligible)
    if total == 0:
        return math.nan, math.nan
    p = sum(events) / total
    if r < 2:
        return p, math.nan
    ss = sum((a - p * n) ** 2 for a, n in zip(events, eligible, strict=True))
    return p, math.sqrt(r / (r - 1) * ss) / total


# 첫 관문: 9.1 귀무 라벨률 ---------------------------------------------------------------------


def _pool_rngs(seed: int, pools: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(pools)]


def null_label_gate(
    p_hat: Mapping[Cell, Sequence[float]],
    *,
    pools: int,
    n_null: int,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """확장 코드 경로의 Oracle 모의(셀당 50, S0 5회 다수결).

    효과 0 후보의 Local Harm·Severe 비율과 풀 단위 MC SE를 낸다.
    """
    p = {cell: np.asarray(p_hat[cell], dtype=float) for cell in CELLS}
    blocks = stratified_bootstrap_indices([ORACLE_PER_CELL] * 6, n_boot, SEED)
    lh, sv = [], []
    for rng in _pool_rngs(seed, pools):
        or_p = np.stack([rng.choice(p[cell], ORACLE_PER_CELL) for cell in CELLS])
        or_s0 = rng.random((5, *or_p.shape)) < or_p
        repeats: dict[int, dict[CaseId, int]] = {
            r: {
                (f, lv, 700_000 + i): int(or_s0[r, k, i])
                for k, (f, lv) in enumerate(CELLS)
                for i in range(ORACLE_PER_CELL)
            }
            for r in range(5)
        }
        tau = tau_cells(repeats)
        maj = (or_s0.sum(axis=0) >= 3).astype(np.int64).reshape(-1)
        null = (rng.random((6, ORACLE_PER_CELL, n_null)) < or_p[:, :, None]).astype(np.int64)
        d = null.reshape(-1, n_null) - maj[:, None]
        lh.append(int(ps.local_harm_many(d, ORACLE_PER_CELL, tau).sum()))
        sv.append(int(ps.severe_many(d, blocks, tau).sum()))
    eligible = [n_null] * pools
    out: dict[str, Any] = {"pools": pools, "n_null": n_null, "n": pools * n_null, "seed": seed}
    for name, counts in (("local_harm", lh), ("severe", sv)):
        rate, se = mc_se(counts, eligible)
        out[name] = {"count": sum(counts), "rate": rate, "se": se}
    out["per_pool"] = {"local_harm": lh, "severe": sv}
    return out


def gate_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """보고서 §0 기준: |p_new - p_9.1| <= 2 √2 SE_new이면 재현."""
    out: dict[str, Any] = {}
    for name, registered in REGISTERED_NULL.items():
        rate, se = result[name]["rate"], result[name]["se"]
        bound = GATE_SE_MULTIPLE * se
        out[name] = {
            "registered": registered,
            "rate": rate,
            "se": se,
            "difference": rate - registered,
            "bound": bound,
            "reproduced": abs(rate - registered) <= bound,
        }
    out["reproduced"] = all(out[name]["reproduced"] for name in REGISTERED_NULL)
    return out


def load_v1_p_hat(root: Path) -> dict[Cell, list[float]]:
    """v1 Oracle S0 기록(반복 5회)의 케이스별 성공률. sha256은 커밋된 v1 실행 요약과 대조한다."""
    pinned = json.loads((root / V1_RUN_SUMMARY).read_text())["inputs"][str(V1_ORACLE_S0)]
    actual = _sha(root / V1_ORACLE_S0)
    if pinned != V1_ORACLE_S0_SHA256 or actual != pinned:
        raise ValueError(f"{V1_ORACLE_S0} sha256 {actual} does not match the pin {pinned}")
    _effects, p_hat = ps.load_v1_inputs(root)
    return p_hat


def _gate(root: Path, workers_note: str = "") -> int:
    load_plan(root)
    start = time.perf_counter()
    result = null_label_gate(load_v1_p_hat(root), pools=GATE_POOLS, n_null=GATE_NULL, seed=0)
    verdict = gate_verdict(result)
    result.pop("per_pool")
    print(
        json.dumps(
            {"gate": result, "verdict": verdict, "wall_s": round(time.perf_counter() - start, 1)},
            indent=1,
        )
    )
    return 0 if verdict["reproduced"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t013-sim")
    parser.add_argument("command", choices=("gate",))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        return _gate(args.root)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
