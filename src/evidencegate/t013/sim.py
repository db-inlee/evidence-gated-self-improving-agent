"""Ticket 013 3단계 게이트 모의(동결 계획 `plans/ticket-013-sim-plan.json`, 태그 t013-sim-plan).

    uv run python -m evidencegate.t013.sim gate                 # 첫 관문(9.1 귀무 라벨률 재현)
    uv run python -m evidencegate.t013.sim run --workers 6      # 본 실행(결과는 한 번만 쓴다)
    uv run python -m evidencegate.t013.sim run --trial --pools 4 --boot 100   # 시험(runs/ 아래만)

- 입력은 동결 계획과 부록이다. 실행 시작 때 두 파일의 sha256을 고정값(`PLAN_SHA256`,
  `ADDENDUM_SHA256`)과 옆 `.sha256`으로 확인하고, 다르면 멈춘다. 두 파일은 고치지 않는다.
- 판정 부품: Tango score 하한(주), 정확 조건부 하한(보조), 칸 percentile bootstrap 하한(시연),
  SAGE(원문 규칙의 한 라운드판), 칸별 SAGE 상한, 신용형 게이트, 풀 단위 MC SE. C4·C5·Oracle
  라벨은 `evidencegate.t009.power_sim`의 함수(실제 함수와 같음이 테스트로 확인됨)를 그대로 쓴다.
- 본 모의: 블록(기준선, n_c, 결합 설정)마다 풀 2,000개를 `sim_core.evaluate_pool`로 판정하고,
  끝난 블록을 `runs/ticket013_sim/blocks/`(git 밖)에 쓴다. 다시 시작하면 해시가 같은 블록은
  그대로 쓰고, 다르면 거부한다. 모든 블록이 끝났을 때만 `results/ticket-013/sim.json`
  (+ `.sha256`)을 한 번 쓴다.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import N_BOOT, SEED
from evidencegate.eval_data import CELLS, CaseId, Cell
from evidencegate.exact import stratified_bootstrap_indices
from evidencegate.oracle import tau_cells
from evidencegate.t009 import power_sim as ps

if TYPE_CHECKING:
    from evidencegate.t013.sim_core import Baseline

PLAN_PATH = Path("plans/ticket-013-sim-plan.json")
PLAN_SHA256 = "8c97f64145d8f9b79cb7cb7591fafbd4e1d4895f57c8f3c05e7d797db3db43a0"
ADDENDUM_PATH = Path("plans/ticket-013-sim-plan-addendum.json")
ADDENDUM_SHA256 = "3ccd2b03beb17942dea5196f9f83504584a0584a5bf44b61b657fad99d7dcebe"
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
    return (d / n) * (2 * _cp_lower(w, d, float(alpha)) - 1)


def _upper_tail(k: int, n: int, q: float) -> float:
    """P[Bin(n, q) >= k], 로그 공간 합(정확한 이항 합과 같은 값을 float로)."""
    if k <= 0:
        return 1.0
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lq, lr = math.log(q), math.log1p(-q)
    base = math.lgamma(n + 1)
    return math.fsum(
        math.exp(base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lq + (n - i) * lr)
        for i in range(k, n + 1)
    )


def _cp_lower(k: int, n: int, alpha: float) -> float:
    """한쪽 Clopper-Pearson 하한: P[Bin(n, q) >= k] = alpha인 q(k = 0이면 0).

    `exact.clopper_pearson(k, n, 2 alpha)[0]`과 같은 값이다(테스트로 대조). 하한만 필요해 위쪽 끝과
    정확한 정수 합을 건너뛴다.
    """
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _upper_tail(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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
    # q = λ / (1 + λ)이므로 Σ C(m, k) q^k (1 - q)^(m - k) = Σ C(m, k) λ^k / (1 + λ)^m (정수 합)
    num = sum(math.comb(m, k) * lam**k for k in range(w, m + 1))
    return Fraction(num, (1 + lam) ** m)


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


# 부록 -----------------------------------------------------------------------------------------


def load_addendum(root: Path) -> dict[str, Any]:
    """동결 부록을 읽는다. 원 계획도 확인하고, 부록이 가리키는 계획 sha256이 같아야 한다."""
    load_plan(root)
    path = root / ADDENDUM_PATH
    actual = _sha(path)
    recorded = path.with_name(path.name + ".sha256").read_text().split()[0]
    if actual != ADDENDUM_SHA256 or recorded != ADDENDUM_SHA256:
        raise ValueError(
            f"{ADDENDUM_PATH} sha256 {actual} does not match the frozen {ADDENDUM_SHA256}"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data["status"] != "frozen" or data["addendum_to"]["sha256"] != PLAN_SHA256:
        raise ValueError(f"{ADDENDUM_PATH} is not frozen against {PLAN_PATH}")
    return data


# 본 모의: 입력과 설정 ---------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    baselines: tuple["Baseline", ...]  # a, b 순서
    v1_effects: tuple[tuple[Fraction, ...], ...]
    group_sizes: tuple[int, ...]
    calibration_pairs_009b: Mapping[str, Mapping[str, int]]  # 기술용(부록 seed_coupling.report)
    gate: Mapping[str, Any]


@dataclass(frozen=True)
class RunSettings:
    n_cs: tuple[int, ...]
    margins: tuple[Fraction, ...]
    pools: int
    n_boot: int
    candidates: int
    workers: int


def main_settings(plan: Mapping[str, Any], workers: int = 1) -> RunSettings:
    grid, mc = plan["grid"], plan["monte_carlo"]
    return RunSettings(
        n_cs=tuple(grid["n_c"]),
        margins=tuple(Fraction(str(m)) for m in grid["m_c"]),
        pools=mc["pools_per_condition"],
        n_boot=mc["bootstrap_draws"],
        candidates=grid["candidates"],
        workers=workers,
    )


T005_ANALYSIS = Path("results/ticket-005/analysis.json")
T005_ANALYSIS_SHA256 = "51d51c9b01e5471e72c7927133d9f9c5732a46036d7867ba82359601e89fc982"
POOL_GROUPS = Path("plans/ticket-009b-pool-groups.json")
POOL_GROUPS_SHA256 = "c3893a7737fe9e2cc76d3b4e1c2f5ffe2f1c8804ecc95fc68798e70a14d8bcc9"
BLOCKS_DIR = Path("runs/ticket013_sim/blocks")
TRIAL_DIR = Path("runs/ticket013_sim/trial")
FINAL = Path("results/ticket-013/sim.json")


def _pinned_json(root: Path, rel: Path, digest: str) -> Any:
    if _sha(root / rel) != digest:
        raise ValueError(f"{rel} sha256 does not match the pin {digest}")
    return json.loads((root / rel).read_text(encoding="utf-8"))


def load_inputs(root: Path, *, gate: Mapping[str, Any]) -> Inputs:
    """기준선 a(S0 보정 기록), b(S0-prime 0단계 기록), v1 효과, 묶음 크기, 009b 보정 L1-L2 짝."""
    from evidencegate.eval_constants import S0_SKILL_ID
    from evidencegate.t013 import audit
    from evidencegate.t013.sim_core import CELL_NAMES, Baseline

    runs = root / "runs"
    pins = audit.record_pins(root)
    used: dict[str, str] = {}
    calibration = []
    for split in ("calibration", "t009_calibration"):
        calibration += audit._read(audit._store(runs, split, S0_SKILL_ID), root, pins, used)
    stage0 = [
        audit._read(audit._store(runs, split, audit.S0_PRIME_SKILL_ID), root, pins, used)
        for split in audit.T012_SPLITS
    ]

    def population(records: Sequence[Any], name: str) -> "Baseline":
        per: dict[CaseId, list[int]] = {}
        for r in records:
            per.setdefault(audit.case_key(r), []).append(int(r.correct_disposition))
        cells = tuple(
            tuple(Fraction(sum(v), len(v)) for c, v in sorted(per.items()) if c[:2] == cell)
            for cell in CELLS
        )
        return Baseline(name, cells)

    a = population(calibration, "a")
    b = population([r for run in stage0 for r in run], "b")
    t005 = _pinned_json(root, T005_ANALYSIS, T005_ANALYSIS_SHA256)
    effects = tuple(
        tuple(
            Fraction(c["oracle"]["correct_disposition"]["cells"][name]["exact"])
            for name in CELL_NAMES
        )
        for _, c in sorted(t005["candidates"].items())
        if c["group"] == "proposer"
    )
    groups = _pinned_json(root, POOL_GROUPS, POOL_GROUPS_SHA256)["groups"]
    sizes = tuple(sorted((len(g["calls"]) for g in groups.values()), reverse=True))
    by_rep: dict[tuple[str, int, int], dict[str, int]] = {}
    for r in calibration:
        by_rep.setdefault((r.key.family, r.key.incident_seed, r.key.repeat), {})[r.key.level] = int(
            r.correct_disposition
        )
    pairs: dict[str, dict[str, int]] = {}
    for (family, _seed, _rep), levels in sorted(by_rep.items()):
        if {"L1", "L2"} <= set(levels):
            row = pairs.setdefault(family, {"n11": 0, "n10": 0, "n01": 0, "n00": 0})
            row["n" + str(levels["L1"]) + str(levels["L2"])] += 1
    return Inputs((a, b), effects, sizes, pairs, gate)


# 본 모의: 블록과 체크포인트 ---------------------------------------------------------------


def _block_id(baseline: str, n_c: int, coupled: bool) -> str:
    return f"{baseline}_{n_c}_{'coupled' if coupled else 'none'}"


def _setups(inputs: Inputs, settings: RunSettings) -> list[tuple[str, Any]]:
    from evidencegate.t013.sim_core import Setup

    out = []
    for index, baseline in enumerate(inputs.baselines):
        for n_c in settings.n_cs:
            for coupled in (False, True):
                setup = Setup(
                    baseline=baseline,
                    baseline_index=index,
                    n_c=n_c,
                    coupled=coupled,
                    n_boot=settings.n_boot,
                    candidates=settings.candidates,
                    margins=settings.margins,
                    v1_effects=inputs.v1_effects,
                    group_sizes=inputs.group_sizes,
                    demo=index == 0 and n_c == 70 and not coupled,
                )
                out.append((_block_id(baseline.name, n_c, coupled), setup))
    return out


def _chunk(setup: Any, pools: Sequence[int]) -> list[dict[str, Any]]:
    from evidencegate.t013 import sim_core

    return [_stringify(sim_core.evaluate_pool(setup, pool)) for pool in pools]


def _stringify(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        part: (
            {"|".join(k): v for k, v in result[part].items()}
            if part != "calibration"
            else result[part]
        )
        for part in result
    }


def _checkpoint_header(settings: RunSettings) -> dict[str, Any]:
    return {
        "plan_sha256": PLAN_SHA256,
        "addendum_sha256": ADDENDUM_SHA256,
        "pools": settings.pools,
        "n_boot": settings.n_boot,
        "candidates": settings.candidates,
    }


def run_blocks(inputs: Inputs, settings: RunSettings, *, blocks_dir: Path) -> list[str]:
    """끝나지 않은 블록만 계산해 체크포인트로 쓴다. 이미 있는 체크포인트는 머리말이 같아야 한다."""
    from concurrent.futures import ProcessPoolExecutor

    blocks_dir.mkdir(parents=True, exist_ok=True)
    header = _checkpoint_header(settings)
    done = []
    for block_id, setup in _setups(inputs, settings):
        path = blocks_dir / f"{block_id}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if {k: existing.get(k) for k in header} != header:
                raise ValueError(f"checkpoint {path} was made with other settings or hashes")
            done.append(block_id)
            continue
        start = time.perf_counter()
        pools = list(range(settings.pools))
        if settings.workers <= 1:
            results = _chunk(setup, pools)
        else:
            size = max(1, math.ceil(len(pools) / (settings.workers * 4)))
            chunks = [pools[i : i + size] for i in range(0, len(pools), size)]
            with ProcessPoolExecutor(settings.workers) as ex:
                results = [
                    r for part in ex.map(_chunk, [setup] * len(chunks), chunks) for r in part
                ]
        payload = header | {
            "block": block_id,
            "wall_s": round(time.perf_counter() - start, 1),
            "results": results,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        tmp.replace(path)
        print(f"block {block_id}: {payload['wall_s']} s", flush=True)
        done.append(block_id)
    return done


# 본 모의: 집계와 판정 ---------------------------------------------------------------------


def worst_false_accept(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """부적합 시나리오 x 배치 중 P(accept | inadmissible)의 최댓값(부적합 후보가 있는 것만)."""
    best: dict[str, Any] | None = None
    for key, row in sorted(rows.items()):
        if not row["eligible"] or row["estimate"] is None:
            continue
        if best is None or row["estimate"] > best["estimate"]:
            best = {"estimate": row["estimate"], "se": row["se"], "scenario": key}
    return best


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[int(q * (len(ordered) - 1))]

    return {"median": pick(0.5), "p05": pick(0.05), "p95": pick(0.95)}


def assemble(inputs: Inputs, settings: RunSettings, *, blocks_dir: Path) -> dict[str, Any]:
    """모든 블록 체크포인트에서 결과를 만든다. 빠진 블록이 있으면 거부한다."""
    from evidencegate.t013 import sim_core as sc

    header = _checkpoint_header(settings)
    paths = []
    for block_id, _setup in _setups(inputs, settings):
        path = blocks_dir / f"{block_id}.json"
        if not path.exists():
            raise ValueError(f"block {block_id} is missing; run every block first")
        paths.append((block_id, path))

    metrics: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    oracle: dict[str, dict[str, Any]] = {}
    calibration: dict[str, Any] = {}
    simulated: dict[str, Any] = {}
    for block_id, path in paths:  # 블록 하나씩 읽고 요약한 뒤 버린다(메모리)
        data = json.loads(path.read_text())
        if {k: data.get(k) for k in header} != header:
            raise ValueError(f"checkpoint {path} was made with other settings or hashes")
        pools: list[dict[str, Any]] = data.pop("results")
        del data
        metrics[block_id] = {}
        for key in pools[0]["gates"]:
            rows = [p["gates"][key] for p in pools]
            metrics[block_id][key] = {
                "acceptance": sc.rate(rows, "accepted", "n"),
                "false_accept": sc.rate(rows, "accepted_inadmissible", "inadmissible"),
                "inadmissible_among_accepted": sc.rate(rows, "accepted_inadmissible", "accepted"),
                "useful_acceptance": sc.rate(rows, "accepted_useful", "useful"),
                "gain_per_proposal": sc.rate(rows, "gain", "n"),
            }
        coverage[block_id] = {
            key: {
                "tango": sc.rate([p["coverage"][key] for p in pools], "tango_covered", "cells"),
                "exact_conditional": sc.rate(
                    [p["coverage"][key] for p in pools], "exact_covered", "cells"
                ),
                "exact_d_zero": sc.rate(
                    [p["coverage"][key] for p in pools], "exact_d_zero", "cells"
                ),
            }
            for key in pools[0]["coverage"]
        }
        oracle[block_id] = {
            key: sc.rate([p["oracle"][key] for p in pools], "uncertain", "n")
            for key in pools[0]["oracle"]
        }
        calibration[block_id] = {
            "h": _summary([float(Fraction(p["calibration"]["h"])) for p in pools]),
            "delta": _summary([float(Fraction(p["calibration"]["delta"])) for p in pools]),
        }
        families = pools[0]["calibration"]["pairs"]
        simulated[block_id] = {
            fam: sc.phi(
                {
                    c: sum(p["calibration"]["pairs"][fam][c] for p in pools)
                    for c in ("n11", "n10", "n01", "n00")
                }
            )
            for fam in families
        }

    baselines: dict[str, Any] = {}
    for b in inputs.baselines:
        scen: dict[str, dict[str, Any]] = {}
        for m in settings.margins:
            scen[str(m)] = {
                f"{x.name}|{x.placement or '-'}": {
                    "feasible": x.feasible,
                    "shifts": {n: str(v) for n, v in zip(sc.CELL_NAMES, x.shifts, strict=True)},
                    "deltas": {n: str(v) for n, v in zip(sc.CELL_NAMES, x.deltas, strict=True)},
                    "delta": str(x.delta),
                }
                for x in sc.build_scenarios(b, m)
            }
        baselines[b.name] = {
            "means": {n: str(v) for n, v in zip(sc.CELL_NAMES, b.means, strict=True)},
            "room_cells": [sc.CELL_NAMES[i] for i in sc.room_cells(b)],
            "loss_cell": {pl: sc.CELL_NAMES[sc.loss_cell(b, pl)] for pl in sc.PLACEMENTS},
            "scenarios": scen,
        }

    decision: dict[str, Any] = {}
    go = False
    for b in inputs.baselines:
        combos = []
        for n_c in settings.n_cs:
            block = metrics[_block_id(b.name, n_c, False)]
            for m in settings.margins:
                for gate in sc.GO_GATES:
                    fa_rows: dict[str, dict[str, Any]] = {}
                    for name in sc.INADMISSIBLE_SCENARIOS:
                        for pl in ("-",) if name == "null_all_cells" else sc.PLACEMENTS:
                            key = f"{m}|{name}|{pl}|main|{gate}"
                            if key in block:
                                fa_rows[f"{name}|{pl}"] = block[key]["false_accept"]
                    fa = worst_false_accept(fa_rows)
                    power_key = f"{m}|useful_representative|-|main|{gate}"
                    power = block[power_key]["acceptance"] if power_key in block else None
                    entry: dict[str, Any] = {
                        "gate": gate,
                        "n_c": n_c,
                        "m_c": str(m),
                        "false_accept": fa,
                        "useful_power": power,
                    }
                    if fa is None or power is None:
                        entry |= {
                            "episodes": (settings.candidates + 1) * 6 * n_c,
                            "meets": False,
                            "borderline": False,
                            "note": "a scenario is infeasible or empty",
                        }
                    else:
                        rows_fa = {"false_accept": fa, "useful_power": power}
                        entry |= sc.judge(rows_fa, n_c=n_c, candidates=settings.candidates)
                    entry["counts_toward_go"] = bool(entry["meets"] and not entry["borderline"])
                    go = go or entry["counts_toward_go"]
                    combos.append(entry)
        decision[b.name] = combos
    decision["go"] = go
    return {
        "settings": _checkpoint_header(settings)
        | {"n_c": list(settings.n_cs), "m_c": [str(m) for m in settings.margins]},
        "gate": dict(inputs.gate),
        "baselines": baselines,
        "calibration": calibration,
        "correlations": {
            "simulated": simulated,
            "009b_calibration_descriptive": {
                fam: sc.phi(c) for fam, c in sorted(inputs.calibration_pairs_009b.items())
            },
        },
        "metrics": metrics,
        "coverage": coverage,
        "oracle_label_uncertainty": oracle,
        "decision": decision,
    }


def _clean(obj: Any) -> Any:
    """JSON에 넣을 수 없는 NaN을 None으로."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def run_simulation(
    inputs: Inputs, settings: RunSettings, *, blocks_dir: Path, final: Path
) -> dict[str, Any]:
    """블록을 모두 끝내고 최종 파일을 한 번 쓴다. 이미 있으면 거부한다."""
    if final.exists():
        raise FileExistsError(f"{final} exists; results are written once")
    run_blocks(inputs, settings, blocks_dir=blocks_dir)
    result = _clean(assemble(inputs, settings, blocks_dir=blocks_dir))
    payload = (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(payload)
    final.with_name(final.name + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {final.name}\n"
    )
    out: dict[str, Any] = json.loads(payload)
    return out


def _run(root: Path, *, workers: int, trial: bool, pools: int | None, boot: int | None) -> int:
    plan = load_plan(root)
    load_addendum(root)
    settings = main_settings(plan, workers)
    if trial:
        settings = RunSettings(settings.n_cs, settings.margins, pools or 4, boot or 100,
                               settings.candidates, workers)  # fmt: skip
    elif pools is not None or boot is not None:
        raise ValueError(
            "--pools and --boot are for --trial only; the main run uses the frozen plan"
        )
    gate_pools = settings.pools if trial else GATE_POOLS
    gate = null_label_gate(load_v1_p_hat(root), pools=gate_pools, n_null=GATE_NULL, seed=0)
    gate.pop("per_pool")
    verdict = gate_verdict(gate)
    if not trial and not verdict["reproduced"]:
        print(json.dumps(verdict, indent=1))
        print("error: the null label rates of 9.1 were not reproduced; stopping", file=sys.stderr)
        return 2
    inputs = load_inputs(root, gate={"result": gate, "verdict": verdict})
    base = root / (TRIAL_DIR if trial else Path("."))
    blocks_dir = base / "blocks" if trial else root / BLOCKS_DIR
    final = base / "sim.json" if trial else root / FINAL
    result = run_simulation(inputs, settings, blocks_dir=blocks_dir, final=final)
    print(json.dumps({"final": str(final), "go": result["decision"]["go"]}, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t013-sim")
    parser.add_argument("command", choices=("gate", "run"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--pools", type=int, default=None)
    parser.add_argument("--boot", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "gate":
            return _gate(args.root)
        return _run(
            args.root, workers=args.workers, trial=args.trial, pools=args.pools, boot=args.boot
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
