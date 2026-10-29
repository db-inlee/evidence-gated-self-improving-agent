"""원고용 값 층 (Ticket 007 §1·§2). 읽기 전용이다.

    uv run python -m evidencegate.writeup --out writeup/generated/values.json

- 출처는 세 갈래다: 결과 파일 두 개(`result`), 설계 상수(`design`, `eval_constants`·`splits`에서
  import), 실행 요약(`run`, `writeup/inputs/run_summary.json`). 여기에 결과 값의 산술만으로 얻는
  `derived`(S0 ES율 = CD율 - g), 탐색 계산 `exploratory`(E1), 출처 표시 `provenance`(해시·커밋)가
  더해진다.
- 결과 파일 두 개, E1 정의 파일의 sha256이 고정값과 다르면 멈춘다. 실행 요약은 자기 sha256 기록과
  같아야 한다.
- 출력은 평평한 키 → {"exact", "display", "kind"}다. 표시 형식 규칙은 이 모듈의 `display` 한
  곳에만 둔다(소수 3자리 half-up, 비율 k/n, 구간 [a, b], 판정 범주는 영어 문구).
- E1은 `writeup/inputs/e1_definition.txt`(커밋 185d69c)의 정의 그대로다. L2 세 칸 Δ의 평균이고,
  판정·구간·임계값은 쓰지 않는다.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate import eval_constants as c
from evidencegate import splits
from evidencegate.analysis import REPO_ROOT
from evidencegate.run_summary import summary_bytes

T005_PATH = Path("results/ticket-005/analysis.json")
T006_PATH = Path("results/ticket-006/analysis.json")
RUN_SUMMARY_PATH = Path("writeup/inputs/run_summary.json")
E1_DEFINITION_PATH = Path("writeup/inputs/e1_definition.txt")
DEFAULT_OUT = Path("writeup/generated/values.json")
T005_PLAN = Path("plans/ticket-005-plan.json")

KINDS = ("result", "derived", "exploratory", "design", "run", "provenance")
DECIMALS = 3
METRIC_KEYS = {"correct_disposition": "cd", "evidence_safe": "es"}
FAMILIES = ("bad_deployment", "dependency_failure", "traffic_spike")
RATE_KEYS = {"successes", "n", "excluded", "value", "interval"}

# 결과 파일에 들어 있지 않은 고정 절차 커밋(README §3-4). git에 있는지는 테스트가 확인한다.
COMMITS = {
    "preregistration": "aa1c809",
    "candidate_pool": "7c48a6d",
    "l3_preregistration": "49ce87d",
    "t005_results": "38ea84e",
    "t006_results": "528c054",
    "e1_definition": "185d69c",
}

VERDICT_EN = {
    "supported": "supported",
    "partially_supported": "partially supported",
    "not_supported": "not supported",
    "undecidable": "undecidable",
    "grows": "grows",
    "partly_grows": "partly grows",
    "does_not_grow": "does not grow",
    "trend_supported": "trend supported",
    "partial_trend_support": "partial trend support",
    "trend_not_supported": "trend not supported",
    "confirmed": "manipulation confirmed",
    "not_confirmed": "manipulation not confirmed",
    "preserved": "preserved",
    "changed": "changed",
    "as_expected": "as expected",
    "unexpected": "unexpected",
}

Value = dict[str, str]


# ---------------------------------------------------------------------------
# 표시 형식
# ---------------------------------------------------------------------------


def _decimal(x: Fraction | float) -> str:
    """소수 DECIMALS자리. 정확한 값에서 half-up(0에서 먼 쪽)으로 반올림한다.

    float는 repr이 나타내는 십진값을 정확한 값으로 본다.
    """
    exact = Fraction(x) if isinstance(x, Fraction) else Fraction(Decimal(repr(x)))
    scaled = abs(exact) * 10**DECIMALS
    units = int(scaled) + (1 if scaled - int(scaled) >= Fraction(1, 2) else 0)
    sign = "-" if exact < 0 and units else ""
    return f"{sign}{units // 10**DECIMALS}.{units % 10**DECIMALS:0{DECIMALS}d}"


def _is_fraction(obj: object) -> bool:
    return isinstance(obj, dict) and set(obj) == {"exact", "float"}


def display(obj: object) -> tuple[str, str]:
    """값 하나 → (exact, display)."""
    if _is_fraction(obj):
        assert isinstance(obj, dict)
        return str(obj["exact"]), _decimal(Fraction(obj["exact"]))
    if isinstance(obj, Fraction):
        return str(obj), _decimal(obj)
    if isinstance(obj, bool):
        return ("true", "yes") if obj else ("false", "no")
    if isinstance(obj, int):
        return str(obj), str(obj)
    if isinstance(obj, float):
        return repr(obj), _decimal(obj)
    if obj is None:
        return "null", "n/a"
    if isinstance(obj, str):
        return obj, VERDICT_EN.get(obj, obj)
    raise TypeError(f"no display rule for {type(obj).__name__}")


def _leaf(obj: object, kind: str) -> Value:
    exact, shown = display(obj)
    return {"exact": exact, "display": shown, "kind": kind}


def _short(name: str) -> str:
    """metric 이름을 cd/es로 줄인다. "C2/correct_disposition" 같은 묶음 키 안에서도."""
    for long, short in METRIC_KEYS.items():
        name = name.replace(long, short)
    return name


def flatten_result(prefix: str, obj: object, kind: str = "result") -> dict[str, Value]:
    """JSON → 평평한 키. metric 이름은 cd/es로 줄인다. 비율과 구간은 묶음 값도 낸다."""
    out: dict[str, Value] = {}

    def walk(key: str, node: object) -> None:
        if _is_fraction(node) or not isinstance(node, dict | list):
            out[key] = _leaf(node, kind)
            return
        if isinstance(node, dict):
            if not node:
                out[key] = {"exact": "{}", "display": "none", "kind": kind}
                return
            if set(node) >= RATE_KEYS:
                s, n = node["successes"], node["n"]
                shown = f"{s}/{n}" if n else "0/0 (undefined)"
                out[key] = {"exact": f"{s}/{n}", "display": shown, "kind": kind}
                interval = node["interval"]
                if n and interval:
                    lo, hi = (display(x) for x in interval)
                    out[f"{key}.ci"] = {
                        "exact": f"{s}/{n} [{lo[0]}, {hi[0]}]",
                        "display": f"{shown} [{lo[1]}, {hi[1]}]",
                        "kind": kind,
                    }
                else:
                    out[f"{key}.ci"] = {"exact": f"{s}/{n}", "display": shown, "kind": kind}
            for child, value in node.items():
                walk(f"{key}.{_short(child)}", value)
            return
        out[f"{key}.count"] = _leaf(len(node), kind)
        if not node:
            out[key] = {"exact": "[]", "display": "none", "kind": kind}
            return
        if all(not isinstance(x, dict | list) or _is_fraction(x) for x in node):
            parts = [display(x) for x in node]
            if len(node) == 2 and key.endswith("interval"):
                shown = f"[{parts[0][1]}, {parts[1][1]}]"
            else:
                shown = ", ".join(p[1] for p in parts)
            out[key] = {
                "exact": "[" + ", ".join(p[0] for p in parts) + "]",
                "display": shown,
                "kind": kind,
            }
        for i, value in enumerate(node):
            walk(f"{key}.{i}", value)

    walk(prefix, obj)
    return out


# ---------------------------------------------------------------------------
# E1 (writeup/inputs/e1_definition.txt)
# ---------------------------------------------------------------------------


def _f(obj: Mapping[str, Any]) -> Fraction:
    return Fraction(obj["exact"])


def e1_rows(t005: Mapping[str, Any], t006: Mapping[str, Any]) -> list[dict[str, Any]]:
    """후보 x metric: Δ_L1/L2, E1 = (1/3)(Δ_bad,L2 + Δ_dep,L2 + Δ_traffic,L2), Δ_L3, 잔차 둘."""
    rows = []
    for name, cand in t005["candidates"].items():
        l3cand = t006["candidates"][name]
        for metric in METRIC_KEYS:
            o12 = cand["oracle"][metric]
            o3 = l3cand["oracle_l3"][metric]
            l2 = {f: _f(o12["cells"][f"{f}/L2"]) for f in FAMILIES}
            e1 = sum(l2.values(), Fraction(0)) / len(FAMILIES)
            l12, l3 = _f(o12["delta"]), _f(o3["delta"])
            rows.append(
                {
                    "name": name,
                    "group": cand["group"],
                    "metric": metric,
                    "l12": l12,
                    "e1": e1,
                    "l3": l3,
                    "residual_e1": l3 - e1,
                    "residual_l12": l3 - l12,
                    "family": {f: (l2[f], _f(o3["cells"][f"{f}/L3"])) for f in FAMILIES},
                }
            )
    return rows


def e1_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """미리 고정한 요약값(proposer, CD): E1이 더 가까운 후보 수(엄격), 잔차 평균 절댓값 둘."""
    chosen = [r for r in rows if r["group"] == "proposer" and r["metric"] == "correct_disposition"]
    if not chosen:
        raise ValueError("no proposer rows for the E1 summary")
    return {
        "closer_count": sum(1 for r in chosen if abs(r["residual_e1"]) < abs(r["residual_l12"])),
        "n": len(chosen),
        "mean_abs_residual_e1": sum((abs(r["residual_e1"]) for r in chosen), Fraction(0))
        / len(chosen),
        "mean_abs_residual_l12": sum((abs(r["residual_l12"]) for r in chosen), Fraction(0))
        / len(chosen),
    }


def _e1_values(t005: Mapping[str, Any], t006: Mapping[str, Any]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    rows = e1_rows(t005, t006)
    for r in rows:
        base = f"e1.{r['name']}.{METRIC_KEYS[r['metric']]}"
        for field in ("l12", "e1", "l3", "residual_e1", "residual_l12"):
            out[f"{base}.{field}"] = _leaf(r[field], "exploratory")
        for f, (l2, l3) in r["family"].items():
            out[f"{base}.family.{f}.l2"] = _leaf(l2, "exploratory")
            out[f"{base}.family.{f}.l3"] = _leaf(l3, "exploratory")
    for field, value in e1_summary(rows).items():
        out[f"e1.summary.proposer.cd.{field}"] = _leaf(value, "exploratory")
    return out


# ---------------------------------------------------------------------------
# derived, design, run, provenance
# ---------------------------------------------------------------------------


def _derived_values(t006: Mapping[str, Any]) -> dict[str, Value]:
    """S0 다수결의 CD율과 ES율(= CD율 - g). RQ-L3-0 값의 산술일 뿐이다."""
    out: dict[str, Value] = {}
    for scope, v in t006["rq"]["RQ_L3_0"].items():
        for side in ("l12", "l3"):
            cd, gap = _f(v["cd"][side]), _f(v["gap"][side])
            out[f"derived.s0_cd_rate.{scope}.{side}"] = _leaf(cd, "derived")
            out[f"derived.s0_es_rate.{scope}.{side}"] = _leaf(cd - gap, "derived")
    return out


def _design_values(root: Path) -> dict[str, Value]:
    plan = json.loads((root / T005_PLAN).read_text(encoding="utf-8"))
    cells = len(c.EVAL_LEVELS) * len(FAMILIES)
    design: dict[str, object] = {
        "epsilon": c.EPSILON,
        "n_boot": c.N_BOOT,
        "seed": c.SEED,
        "n_resample": c.N_RESAMPLE,
        "noise_q": c.NOISE_Q,
        "c4_q": c.C4_Q,
        "tau_q": c.TAU_Q,
        "ci_alpha": c.CI_ALPHA,
        "fitness_size": c.FITNESS_SIZE,
        "s0_oracle_repeats": len(c.S0_ORACLE_REPEATS),
        "cells": cells,
        "dev_a_per_cell": splits.DEV_A_PER_CELL,
        "dev_b_per_cell": splits.DEV_B_PER_CELL,
        "oracle_per_cell": splits.ORACLE_PER_CELL,
        "dev_a_size": splits.DEV_A_PER_CELL * cells,
        "dev_b_size": splits.DEV_B_PER_CELL * cells,
        "oracle_size": splits.ORACLE_PER_CELL * cells,
        "l3_oracle_per_cell": c.L3_SPLIT_SEEDS[c.L3_ORACLE_SPLIT][1],
        "l3_oracle_size": c.L3_SPLIT_SEEDS[c.L3_ORACLE_SPLIT][1] * len(FAMILIES),
        "l3_check_size": c.L3_SPLIT_SEEDS[c.L3_CHECK_SPLIT][1] * len(FAMILIES),
        "t005_planned_episodes": plan["episodes"],
        "t006_planned_episodes": c.L3_MAIN_EPISODES,
        "policy_id": c.POLICY_ID,
        "config_id": c.CONFIG_ID,
        "tool_spec_id": c.TOOL_SPEC_ID,
        "s0_skill_id": c.S0_SKILL_ID,
        "run_workers": c.RUN_WORKERS,
    }
    out = {f"design.{key}": _leaf(value, "design") for key, value in design.items()}
    out["design.budget_ns"] = {
        "exact": "[" + ", ".join(str(n) for n in c.BUDGET_NS) + "]",
        "display": ", ".join(str(n) for n in c.BUDGET_NS),
        "kind": "design",
    }
    return out


def _run_values(run: Mapping[str, Any]) -> dict[str, Value]:
    rename = {"ticket004": "t004", "ticket005": "t005", "ticket006": "t006"}
    rename["ticket006_check"] = "t006_check"
    out: dict[str, Value] = {}
    for key, section in run.items():
        if key in ("inputs", "notes"):
            continue
        out |= flatten_result(f"run.{rename[key]}", section, kind="run")
    return out


def _provenance(
    t005: Mapping[str, Any], t006: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Value]:
    entries = {
        "hash.t005_analysis": c.TICKET005_ANALYSIS_SHA256,
        "hash.t006_analysis": c.TICKET006_ANALYSIS_SHA256,
        "hash.e1_definition": c.E1_DEFINITION_SHA256,
        "hash.run_summary": hashlib.sha256(summary_bytes(run)).hexdigest(),
        "hash.t005_plan": t005["meta"]["plan_sha256"],
        "hash.t006_plan": t006["meta"]["plan_sha256"],
        "hash.preregistration_7_8": t005["meta"]["preregistration_sha256"],
        "hash.preregistration_l3": t006["meta"]["preregistration_l3_sha256"],
        "commit.t005_analysis": t005["meta"]["analysis_commit"],
        "commit.t006_analysis": t006["meta"]["analysis_commit"],
        **{f"commit.{k}": v for k, v in COMMITS.items()},
    }
    return {key: {"exact": v, "display": v, "kind": "provenance"} for key, v in entries.items()}


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------


def check_inputs(t005: Path, t006: Path, e1_definition: Path) -> None:
    for path, expected in (
        (t005, c.TICKET005_ANALYSIS_SHA256),
        (t006, c.TICKET006_ANALYSIS_SHA256),
        (e1_definition, c.E1_DEFINITION_SHA256),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{path} sha256 {actual} does not match the fixed {expected}")


def build_values(
    t005: Mapping[str, Any],
    t006: Mapping[str, Any],
    run: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> dict[str, Value]:
    for group, by_m in t005["metrics"].items():
        for m, by_gate in by_m.items():
            for gate, metrics in by_gate.items():
                if t006["metrics"][group][m][gate]["l12"] != metrics:
                    raise ValueError(
                        f"Ticket 006 l12 metrics differ from Ticket 005 at {group} {m} {gate}"
                    )
    values = {
        **flatten_result("t005", t005),
        **flatten_result("t006", t006),
        **_derived_values(t006),
        **_e1_values(t005, t006),
        **_design_values(root),
        **_run_values(run),
        **_provenance(t005, t006, run),
    }
    return dict(sorted(values.items()))


def values_bytes(values: Mapping[str, Value]) -> bytes:
    return (json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-writeup")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--t005", type=Path, default=None)
    parser.add_argument("--t006", type=Path, default=None)
    parser.add_argument("--run-summary", type=Path, default=None)
    parser.add_argument("--e1-definition", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root
    t005 = args.t005 or root / T005_PATH
    t006 = args.t006 or root / T006_PATH
    run_path = args.run_summary or root / RUN_SUMMARY_PATH
    e1 = args.e1_definition or root / E1_DEFINITION_PATH
    out = args.out or root / DEFAULT_OUT
    try:
        check_inputs(t005, t006, e1)
        recorded = run_path.with_name(run_path.name + ".sha256").read_text().split()[0]
        if hashlib.sha256(run_path.read_bytes()).hexdigest() != recorded:
            raise ValueError(f"{run_path} sha256 does not match its record {recorded}")
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    values = build_values(
        json.loads(t005.read_text(encoding="utf-8")),
        json.loads(t006.read_text(encoding="utf-8")),
        json.loads(run_path.read_text(encoding="utf-8")),
        root,
    )
    data = values_bytes(values)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"values {out}: {len(values)} keys, sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
