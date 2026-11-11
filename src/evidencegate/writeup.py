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
- 후속 연구(Ticket 011 §1, 선택 인자 `follow_up`): 009b 결과(`t009b.`, result), 010 탐침(`t010.`,
  exploratory), 009 h·δ(`t009.thresholds.`, result), 실행 요약 009b(`run.`은 run, `provenance.`은
  provenance, 입력 목록은 v1처럼 펴지 않음), 설계 상수(`design.t009b.`), 묶음 수(`derived.t009b.`),
  해시·커밋. 입력 sha256이 고정값·옆 `.sha256`과 다르거나 009b 결과의 사전 등록·h·δ 해시가 다르면
  멈춘다. `follow_up`이 없으면 v1 값만 낸다(v1 키·값·표시는 그대로).
- p값(키 끝이 `.p`인 분수)은 정확한 분수로 표시한다. v1 값에는 그런 키가 없다.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate import eval_constants as c
from evidencegate import release_009b, splits
from evidencegate.analysis import REPO_ROOT, load_pool
from evidencegate.run_summary import summary_bytes
from evidencegate.t009 import splits as t009_splits
from evidencegate.t009.preflight import recorded_sha256
from evidencegate.t009b import settings as t009b_settings

T005_PATH = Path("results/ticket-005/analysis.json")
T006_PATH = Path("results/ticket-006/analysis.json")
RUN_SUMMARY_PATH = Path("writeup/inputs/run_summary.json")
E1_DEFINITION_PATH = Path("writeup/inputs/e1_definition.txt")
RELEASE_MANIFEST_PATH = Path("writeup/inputs/release_manifest.json")
DEFAULT_OUT = Path("writeup/generated/values.json")
T005_PLAN = Path("plans/ticket-005-plan.json")

# 후속 연구 입력(Ticket 011 §1). 결과 파일은 읽기만 한다.
RELEASE_MANIFESTS_009B = release_009b.MANIFESTS  # 압축본마다 매니페스트(있는 것만 편다)
T009B_PATH = Path("results/ticket-009b/analysis.json")
T009B_SIDECAR = Path("results/ticket-009b/analysis.json.sha256")
T010_PATH = Path("results/ticket-010/probe.json")
T009_THRESHOLDS_PATH = t009b_settings.THRESHOLDS_PATH
RUN_SUMMARY_009B_PATH = Path("writeup/inputs/run_summary_009b.json")
RUN_SUMMARY_009B_SIDECAR = Path("writeup/inputs/run_summary_009b.json.sha256")
PREREG_009B_PATH = t009b_settings.PREREG_FILE
PREREG_009_PATH = Path("preregistration/ticket-009.txt")
PREREG_009_SHA_PATH = Path("preregistration/ticket-009.sha256")
T009B_ANALYSIS_SHA256 = "21c87f2798fa032e3ce627029e8b01c7629cb9b73215aeac1b3742f36d3cdafe"
T010_PROBE_SHA256 = "f907489e277e3e3fd3d237d5e61d71554b2f1176bf81d53d62a0bc254835158c"
T009_THRESHOLDS_SHA256 = t009b_settings.THRESHOLDS_SHA256
RUN_SUMMARY_009B_SHA256 = "ff203e7e926de66210d5f25852fb20909830905537db22f55b4d778aa78a326f"
FOLLOW_UP_INPUTS: dict[str, tuple[Path, str]] = {
    "t009b": (T009B_PATH, T009B_ANALYSIS_SHA256),
    "t010": (T010_PATH, T010_PROBE_SHA256),
    "t009_thresholds": (T009_THRESHOLDS_PATH, T009_THRESHOLDS_SHA256),
    "run_summary_009b": (RUN_SUMMARY_009B_PATH, RUN_SUMMARY_009B_SHA256),
}

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

# 연구 ② 고정 절차 커밋(README 고정 절차 표). git에 있고 그 파일을 바꿨는지는 테스트가 확인한다.
COMMITS_009B = {
    "t009_preregistration": "20de478",  # 태그 t009-prereg-r1
    "t009_thresholds": "4716c65",
    "t009_closed": "32d6560",  # 태그 ticket-009-closed
    "t009b_preregistration": "12a36fd",  # 태그 t009b-prereg
    "t009b_pool": "4c3b532",
    "t009b_groups": "4c3b532",
    "t009b_plan": "7b17c8b",
    "t009b_analysis_code": "94fa123",
    "t009b_results": "adb9939",  # 태그 ticket-009b-closed
    "t010_probe": "cd76a4e",  # 태그 ticket-010-closed
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
    "direction_only": "direction only",
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
        if _is_fraction(node) and key.endswith(".p"):  # p값: 정확한 분수(v1에는 없는 키)
            assert isinstance(node, dict)
            out[key] = {"exact": str(node["exact"]), "display": str(node["exact"]), "kind": kind}
            return
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
    pool = load_pool(root)
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
        "proposer_candidates": sum(1 for x in pool.candidates if x.group == "proposer"),
        "control_candidates": sum(1 for x in pool.candidates if x.group == "control"),
        # 탐색 (b): 한 L2 칸이 L1/L2 Oracle에서 차지하는 비중과
        # 한 family가 L3 Oracle에서 차지하는 비중
        "l2_cell_share_l12_oracle": Fraction(
            splits.ORACLE_PER_CELL, splits.ORACLE_PER_CELL * cells
        ),
        "family_share_l3_oracle": Fraction(1, len(FAMILIES)),
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


def _release_values(release: Mapping[str, Any]) -> dict[str, Value]:
    """재현용 기록 압축본(writeup/inputs/release_manifest.json)의 이름·해시·파일 수·비밀 값 확인."""
    out = {
        "hash.records_archive": {
            "exact": release["archive_sha256"],
            "display": release["archive_sha256"],
            "kind": "provenance",
        },
        "release.archive": {
            "exact": release["archive"],
            "display": release["archive"],
            "kind": "provenance",
        },
    }
    out["release.file_count"] = _leaf(len(release["files"]), "run")
    out["release.secret_hits"] = _leaf(release["secret_scan"]["hits"], "run")
    out["release.secret_patterns"] = _leaf(release["secret_scan"]["patterns"], "run")
    return out


# ---------------------------------------------------------------------------
# 후속 연구(Ticket 011 §1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowUp:
    t009b: dict[str, Any]
    t010: dict[str, Any]
    thresholds: dict[str, Any]
    run_009b: dict[str, Any]
    preregistration_009_sha256: str
    releases: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)  # 있는 매니페스트만


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_t009b_inputs(analysis: Mapping[str, Any], *, root: Path) -> None:
    """009b 결과가 기록한 사전 등록·h·δ 해시가 지금 파일·고정값과 같은가."""
    inputs = analysis["inputs"]
    prereg = _file_sha(root / PREREG_009B_PATH)
    if inputs["preregistration"] != prereg:
        raise ValueError(
            f"009b analysis names preregistration {inputs['preregistration']}, file is {prereg}"
        )
    if inputs["thresholds"] != T009_THRESHOLDS_SHA256:
        raise ValueError(
            f"009b analysis names thresholds {inputs['thresholds']}, fixed {T009_THRESHOLDS_SHA256}"
        )


def load_follow_up(root: Path) -> FollowUp:
    """입력 네 개의 고정 sha256, 옆 .sha256 두 개, 009b 입력 해시, 009 사전 등록 해시를 확인한다."""
    data: dict[str, Any] = {}
    for name, (rel, expected) in FOLLOW_UP_INPUTS.items():
        actual = _file_sha(root / rel)
        if actual != expected:
            raise ValueError(f"{rel} sha256 {actual} does not match the fixed {expected}")
        data[name] = json.loads((root / rel).read_text(encoding="utf-8"))
    for sidecar, expected in (
        (T009B_SIDECAR, T009B_ANALYSIS_SHA256),
        (RUN_SUMMARY_009B_SIDECAR, RUN_SUMMARY_009B_SHA256),
    ):
        recorded = (root / sidecar).read_text().split()[0]
        if recorded != expected:
            raise ValueError(f"{sidecar} sidecar records {recorded}, fixed {expected}")
    check_t009b_inputs(data["t009b"], root=root)
    prereg_009 = _file_sha(root / PREREG_009_PATH)
    if recorded_sha256(root / PREREG_009_SHA_PATH) != prereg_009:
        raise ValueError(f"{PREREG_009_PATH} sha256 {prereg_009} does not match its record")
    releases = {
        key: json.loads((root / rel).read_text(encoding="utf-8"))
        for key, rel in RELEASE_MANIFESTS_009B.items()
        if (root / rel).exists()
    }
    return FollowUp(
        data["t009b"],
        data["t010"],
        data["t009_thresholds"],
        data["run_summary_009b"],
        prereg_009,
        releases,
    )


RELEASE_HASH_KEYS = {
    "t009b": "hash.records_archive_009b",
    "t010_probe": "hash.records_archive_010_probe",
}


def follow_up_release_values(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Value]:
    """009b·010 탐침 압축본 매니페스트(있는 것만)의 이름·해시·크기·파일 수·비밀 검사 결과."""
    out: dict[str, Value] = {}
    for key, m in manifests.items():
        digest = m["archive_sha256"]
        out[RELEASE_HASH_KEYS[key]] = {"exact": digest, "display": digest, "kind": "provenance"}
        out[f"release.{key}.archive"] = {
            "exact": m["archive"],
            "display": m["archive"],
            "kind": "provenance",
        }
        out[f"release.{key}.archive_bytes"] = _leaf(m["archive_bytes"], "run")
        out[f"release.{key}.file_count"] = _leaf(len(m["files"]), "run")
        out[f"release.{key}.secret_hits"] = _leaf(m["secret_scan"]["hits"], "run")
        out[f"release.{key}.secret_patterns"] = _leaf(m["secret_scan"]["patterns"], "run")
    return out


def _design_009b_values() -> dict[str, Value]:
    cells = len(t009_splits.LEVELS) * len(FAMILIES)
    dev_a, dev_b, oracle = (t009_splits.DEV_A[2], t009_splits.DEV_B[2], t009_splits.ORACLE[2])
    design: dict[str, object] = {
        "cells": cells,
        "dev_a_per_cell": dev_a,
        "dev_a_size": dev_a * cells,
        "dev_b_per_cell": dev_b,
        "dev_b_size": dev_b * cells,
        "oracle_per_cell": oracle,
        "oracle_size": oracle * cells,
        "calibration_per_cell": t009_splits.CALIBRATION_V1_PER_CELL
        + t009_splits.CALIBRATION_ADD[2],
        "s0_oracle_repeats": len(c.S0_ORACLE_REPEATS),
        # h(n_c)는 Dev-B 칸 크기에서, δ(N)은 Dev-B 전체 크기에서 쓴다(사전 등록 009b)
        "h_n": dev_b,
        "delta_n": dev_b * cells,
        "pool_target": t009b_settings.POOL_TARGET,
        "n_go": t009b_settings.N_GO,
        "stop_after_idle": t009b_settings.STOP_AFTER_IDLE,
        "max_calls": t009b_settings.MAX_CALLS,
        "budget_usd": t009b_settings.BUDGET_USD,
        "dedupe_threshold": t009b_settings.DEDUPE_THRESHOLD,
    }
    return {f"design.t009b.{key}": _leaf(value, "design") for key, value in design.items()}


def _follow_up_values(f: FollowUp) -> dict[str, Value]:
    run = f.run_009b
    provenance = {k: v for k, v in run["provenance"].items() if k != "inputs"}
    inputs = f.t009b["inputs"]
    hashes = {
        "hash.t009b_analysis": T009B_ANALYSIS_SHA256,
        "hash.t010_probe": T010_PROBE_SHA256,
        "hash.t009_thresholds": T009_THRESHOLDS_SHA256,
        "hash.run_summary_009b": RUN_SUMMARY_009B_SHA256,
        "hash.preregistration_009b": inputs["preregistration"],
        "hash.preregistration_009": f.preregistration_009_sha256,
        "hash.t009b_pool": inputs["pool"],
        "hash.t009b_plan": inputs["plan"],
        "hash.t009b_groups": inputs["groups"],
        **{f"commit.{k}": v for k, v in COMMITS_009B.items()},
    }
    groups = len(f.t009b["cluster_sensitivity"]["representatives"])
    return {
        **flatten_result("t009b", f.t009b),
        **flatten_result("t010", f.t010, kind="exploratory"),
        **flatten_result("t009.thresholds", f.thresholds),
        **flatten_result("run", run["run"], kind="run"),
        **flatten_result("provenance", provenance, kind="provenance"),
        **_design_009b_values(),
        "derived.t009b.cluster_groups": _leaf(groups, "derived"),
        **follow_up_release_values(f.releases),
        **{k: {"exact": v, "display": v, "kind": "provenance"} for k, v in hashes.items()},
    }


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
    *,
    release: Mapping[str, Any],
    follow_up: FollowUp | None = None,
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
        **_release_values(release),
    }
    if follow_up is not None:
        extra = _follow_up_values(follow_up)
        clash = sorted(set(extra) & set(values))
        if clash:
            raise ValueError(f"follow-up keys clash with v1 keys: {clash[:3]}")
        values |= extra
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
        follow_up = load_follow_up(root)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    values = build_values(
        json.loads(t005.read_text(encoding="utf-8")),
        json.loads(t006.read_text(encoding="utf-8")),
        json.loads(run_path.read_text(encoding="utf-8")),
        root,
        release=json.loads((root / RELEASE_MANIFEST_PATH).read_text(encoding="utf-8")),
        follow_up=follow_up,
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
