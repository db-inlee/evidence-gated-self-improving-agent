"""Ticket 005 분석 전체 실행 (설계 문서 7.8, 티켓 §8). 모든 블록이 끝난 뒤 한 번만 돌린다.

    uv run python -m evidencegate.analysis --runs-dir runs \\
        --noise runs/calibration/<policy>__7c66ee5bf13c/e9b4996f7529/bf65e569ae37.jsonl \\
        --out-dir results/ticket-005 --analysis-commit <commit> --plan-sha256 <hash>

순서: 7.8 해시 확인 → 실행 조건 확인 → δ(N)(노이즈 기준) → τ_cell(S0 Oracle 반복 5회) → 후보마다
게이트 C1~C4와 Oracle 추정 → 집합별 지표(proposer와 통제 후보 따로, §8-19) → 결과표(7.8 E) →
Budget Curve·RQ2 민감도 → RQ 판정(proposer, CD가 주 metric, §8-24) → 결과 JSON과 Markdown.

결과 JSON은 입력과 설정이 같으면 바이트까지 같다(시각을 넣지 않는다). 유리수는
{"exact": "p/q", "float": x}로 쓴다. 판정은 언제나 유리수로 했다(§8-4).
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

from evidencegate.budget_curve import (
    BudgetCandidate,
    Summary,
    budget_decisions,
    budget_points,
    sensitivity_decisions,
    sensitivity_rates,
)
from evidencegate.eval_constants import (
    BUDGET_NS,
    CONFIG_ID,
    N_BOOT,
    N_RESAMPLE,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    SECTION8_VERSION,
    SEED,
    TOOL_SPEC_ID,
)
from evidencegate.eval_data import (
    CELLS,
    ORACLE_SPLIT,
    check_run_conditions,
    paired_gate,
    paired_oracle,
    repeat_outcomes,
)
from evidencegate.eval_metrics import (
    GATE_NAMES,
    CandidateScore,
    Gain,
    GateMetrics,
    Rate,
    effect_agreement,
    gate_metrics,
)
from evidencegate.exact import null_diff_quantile_exact
from evidencegate.gates import decide_gates
from evidencegate.metrics import METRICS
from evidencegate.oracle import (
    OracleEstimate,
    check_control_label,
    classify_effect,
    estimate,
    fitness,
    harmful,
    local_harm,
    subset_label,
    tau_cells,
    unsupported,
    useful,
)
from evidencegate.rq import Verdict, rq1, rq2_first, rq2_second, rq3_first, rq3_trend
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG_HASH_FILE = Path("preregistration/ticket-004.sha256")
PREREG_COPY = Path("preregistration/ticket-004-section-7.8.txt")
GROUPS = ("proposer", "control")
CD, ES = METRICS
PRIMARY_METRIC = CD  # §8-24
CONTROL_ROWS = (
    ("Positive controls", "positive"),
    ("Negative controls", "negative"),
    ("Mixed controls", "mixed"),
    ("Neutral controls", "neutral"),
)
SUBSET_LABELS = ("positive", "negative", "mixed", "neutral", "uncertain")
SPLITS_FOR_ANALYSIS = ("dev_a", "dev_b", "oracle")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    skill_id: str
    group: str  # "proposer" / "control"
    prior_label: str | None  # 통제 후보의 사전 라벨(positive/negative/mixed/neutral)
    expected_effect: Mapping[str, str] | None  # proposer 증거 카드("cd"/"es" → up/down/none)


@dataclass(frozen=True)
class PoolSpec:
    s0_skill_id: str
    candidates: tuple[CandidateSpec, ...]


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------


def check_preregistration(root: Path = REPO_ROOT) -> str:
    """7.8절 원문 사본의 sha256이 기록과 같은지 확인하고 그 값을 돌려준다."""
    recorded = None
    for line in (root / PREREG_HASH_FILE).read_text(encoding="utf-8").splitlines():
        if line.startswith("sha256: "):
            recorded = line.removeprefix("sha256: ").strip()
    actual = hashlib.sha256((root / PREREG_COPY).read_bytes()).hexdigest()
    if recorded is None or actual != recorded:
        raise ValueError(f"7.8 copy hash {actual} does not match the registered {recorded}")
    return actual


def load_pool(root: Path = REPO_ROOT) -> PoolSpec:
    """동결된 풀(`skills/candidates/POOL.json`)과 증거 카드에서 후보 목록을 만든다."""
    pool = json.loads((root / "skills" / "candidates" / "POOL.json").read_text(encoding="utf-8"))
    specs = []
    for c in pool["proposer_candidates"]:
        card = json.loads(
            (root / "skills" / "candidates" / c["candidate_id"] / "evidence.json").read_text(
                encoding="utf-8"
            )
        )
        specs.append(
            CandidateSpec(
                c["candidate_id"], c["skill_id"], "proposer", None, card["expected_effect"]
            )
        )
    for c in pool["controls"]:
        specs.append(CandidateSpec(c["name"], c["skill_id"], "control", c["subset"], None))
    return PoolSpec(s0_skill_id=_s0_skill_id(root), candidates=tuple(specs))


def load_pool_spec(path: Path) -> PoolSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PoolSpec(
        s0_skill_id=data["s0_skill_id"],
        candidates=tuple(CandidateSpec(**c) for c in data["candidates"]),
    )


def load_records(runs_dir: Path, pool: PoolSpec) -> list[EpisodeRecord]:
    """runs/<split>/<policy>__<config>/<tool_spec>/<skill>.jsonl을 S0와 모든 후보에 대해 읽는다."""
    records: list[EpisodeRecord] = []
    for split in SPLITS_FOR_ANALYSIS:
        for skill in (pool.s0_skill_id, *(c.skill_id for c in pool.candidates)):
            store = EpisodeStore.for_run(
                runs_dir,
                suite=split,
                policy_id=POLICY_ID,
                config_id=CONFIG_ID,
                tool_spec_id=TOOL_SPEC_ID,
                skill_id=skill,
            )
            records += EpisodeStore(store.path, repair=False).records()
    return records


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def run_analysis(
    records: Sequence[EpisodeRecord],
    noise_records: Sequence[EpisodeRecord],
    pool: PoolSpec,
    *,
    run_info: Mapping[str, Any],
    n_resample: int = N_RESAMPLE,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    prereg = check_preregistration(root)
    check_run_conditions(
        [*records, *noise_records],
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
    )
    names = {c.skill_id: c.name for c in pool.candidates}

    deltas = {
        m: {n: null_diff_quantile_exact(noise_records, m, n) for n in (210, *BUDGET_NS)}
        for m in METRICS
    }
    tau = {
        m: tau_cells(
            repeat_outcomes(
                records,
                skill_id=pool.s0_skill_id,
                split=ORACLE_SPLIT,
                metric=m,
                repeats=S0_ORACLE_REPEATS,
            )
        )
        for m in METRICS
    }

    budget_inputs: list[BudgetCandidate] = []
    scores: dict[str, CandidateScore] = {}
    candidates_out: dict[str, Any] = {}
    for spec in pool.candidates:
        dev_a, dev_b, estimates = {}, {}, {}
        accepted: dict[tuple[str, str], bool] = {}
        gates_out: dict[str, Any] = {}
        oracle_out: dict[str, Any] = {}
        for m in METRICS:
            kw = {"candidate_skill_id": spec.skill_id, "s0_skill_id": pool.s0_skill_id, "metric": m}
            dev_a[m] = paired_gate(records, split="dev_a", **kw)
            dev_b[m] = paired_gate(records, split="dev_b", **kw)
            estimates[m] = estimate(paired_oracle(records, **kw), tau[m])
            decisions = decide_gates(dev_a[m], dev_b[m], deltas[m][210])
            gates_out[m] = {}
            for g in decisions:
                accepted[(g.gate, m)] = g.accepted
                gates_out[m][g.gate] = {
                    "accepted": g.accepted,
                    "mean_delta": _frac(g.mean_delta),
                    "threshold": _frac(g.threshold),
                    "lower": _frac(g.lower),
                }
            oracle_out[m] = _oracle_json(spec, estimates[m])
        candidates_out[spec.name] = {
            "group": spec.group,
            "skill_id": spec.skill_id,
            "prior_label": spec.prior_label,
            "expected_effect": dict(spec.expected_effect) if spec.expected_effect else None,
            "gates": gates_out,
            "oracle": oracle_out,
        }
        scores[spec.name] = CandidateScore(spec.name, accepted, estimates, spec.expected_effect)
        budget_inputs.append(
            BudgetCandidate(spec.name, dev_a, dev_b, estimates, spec.expected_effect)
        )

    groups = {g: [c for c in pool.candidates if c.group == g] for g in GROUPS}
    group_scores = {g: [scores[c.name] for c in specs] for g, specs in groups.items()}
    group_budget = {
        g: [b for b in budget_inputs if b.name in {c.name for c in specs}]
        for g, specs in groups.items()
    }

    metrics = {
        g: {m: {gate: gate_metrics(s, gate, m) for gate in GATE_NAMES} for m in METRICS}
        for g, s in group_scores.items()
    }
    rq1_cross = {
        g: {gate: gate_metrics(s, gate, CD, oracle_metric=ES) for gate in GATE_NAMES}
        for g, s in group_scores.items()
    }

    budget_deltas = {(m, n): deltas[m][n] for m in METRICS for n in BUDGET_NS}
    decisions_by_n = budget_decisions(budget_inputs, budget_deltas, n_resample=n_resample)
    budget = {
        g: {n: budget_points(decisions_by_n[n], b) for n in BUDGET_NS}
        for g, b in group_budget.items()
    }
    sens_decisions = sensitivity_decisions(budget_inputs, n_resample=n_resample)
    sensitivity = {
        g: {m: sensitivity_rates(sens_decisions, b, m) for m in METRICS}
        for g, b in group_budget.items()
    }

    controls = {c.name: c for c in groups["control"]}
    fit = {m: fitness({n: scores[n].oracle[m] for n in controls}) for m in METRICS}

    rq = _rq(
        metrics["proposer"], rq1_cross["proposer"], budget["proposer"], sensitivity["proposer"]
    )

    by_skill = Counter(
        names.get(r.key.skill_id, r.key.skill_id) for r in records if r.trace.format_errors > 0
    )
    return {
        "meta": {
            "analysis_commit": run_info.get("analysis_commit"),
            "plan_sha256": run_info.get("plan_sha256"),
            "preregistration_sha256": prereg,
            "section8_version": SECTION8_VERSION,
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
            "s0_skill_id": pool.s0_skill_id,
            "primary_metric": PRIMARY_METRIC,
        },
        "episodes": {
            "total": len(records),
            "format_error_episodes": dict(sorted(by_skill.items())),
            "no_disposition": sum(1 for r in records if r.action is None),
        },
        "delta": {m: {str(n): _frac(v) for n, v in d.items()} for m, d in deltas.items()},
        "tau": {m: {_cell(c): _frac(t[c]) for c in CELLS} for m, t in tau.items()},
        "candidates": candidates_out,
        "results_table": _results_table(pool, candidates_out),
        "metrics": {
            g: {
                m: {gate: _metrics_json(v) for gate, v in by_gate.items()}
                for m, by_gate in d.items()
            }
            for g, d in metrics.items()
        },
        "rq1_cross": {
            g: {gate: _metrics_json(v) for gate, v in d.items()} for g, d in rq1_cross.items()
        },
        "effect_agreement": {
            m: _rate_json(effect_agreement(group_scores["proposer"], m)) for m in METRICS
        },
        "fitness": {
            m: {
                "fit": f.fit,
                "sign_met": f.sign_met,
                "size_met": f.size_met,
                "sign_hits": [list(h) for h in f.sign_hits],
                "size_hits": [list(h) for h in f.size_hits],
            }
            for m, f in fit.items()
        },
        "budget_curve": {
            g: {
                str(n): {
                    f"{gate}/{m}": {
                        "acceptance": {k: _frac(v) for k, v in p.acceptance.items()},
                        "acceptance_rate": _summary_json(p.acceptance_rate),
                        "metrics": {k: _summary_json(v) for k, v in p.metrics.items()},
                    }
                    for (gate, m), p in points.items()
                }
                for n, points in by_n.items()
            }
            for g, by_n in budget.items()
        },
        "sensitivity": {
            g: {m: {k: _summary_json(v) for k, v in r.items()} for m, r in by_m.items()}
            for g, by_m in sensitivity.items()
        },
        "rq": rq,
    }


def _oracle_json(spec: CandidateSpec, e: OracleEstimate) -> dict[str, Any]:
    subset = subset_label(e)
    out: dict[str, Any] = {
        "delta": _frac(e.delta),
        "interval": [_frac(e.interval[0]), _frac(e.interval[1])],
        "cells": {_cell(c): _frac(e.cell_deltas[c]) for c in CELLS},
        "useful": useful(e),
        "harmful": harmful(e),
        "unsupported": unsupported(e),
        "local_harm": local_harm(e),
        "effect": classify_effect(e),
        "subset": subset.label,
        "big_cells": [_cell(c) for c in subset.big_cells],
    }
    if spec.prior_label is not None:
        out["control_check"] = check_control_label(spec.prior_label, e)
    if spec.expected_effect is not None:
        key = "cd" if e.metric == CD else "es"
        actual = classify_effect(e)
        out["effect_match"] = None if actual is None else actual == spec.expected_effect[key]
    return out


def _results_table(pool: PoolSpec, candidates: Mapping[str, Any]) -> list[dict[str, Any]]:
    """7.8 E의 행. 칸은 게이트 x metric마다 "채택 수 / 후보 수"(§8-20)."""

    def row(name: str, members: Mapping[str, list[str]]) -> dict[str, Any]:
        cells = {}
        for gate in GATE_NAMES:
            for m in METRICS:
                who = members[m]
                k = sum(1 for n in who if candidates[n]["gates"][m][gate]["accepted"])
                cells[f"{gate}/{m}"] = f"{k} / {len(who)}"
        counts = {len(v) for v in members.values()}
        return {"row": name, "candidates": max(counts), "cells": cells}

    rows = []
    for title, label in CONTROL_ROWS:
        names = [c.name for c in pool.candidates if c.group == "control" and c.prior_label == label]
        rows.append(row(title, dict.fromkeys(METRICS, names)))
    proposers = [c.name for c in pool.candidates if c.group == "proposer"]
    rows.append(row("Proposer (all)", dict.fromkeys(METRICS, proposers)))
    for label in SUBSET_LABELS:
        members = {
            m: [n for n in proposers if candidates[n]["oracle"][m]["subset"] == label]
            for m in METRICS
        }
        rows.append(row(f"Proposer {label}", members))
    return rows


def _rq(
    metrics: Mapping[str, Mapping[str, GateMetrics]],
    cross: Mapping[str, GateMetrics],
    budget: Mapping[int, Mapping[tuple[str, str], Any]],
    sensitivity: Mapping[str, Mapping[str, Summary]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"primary_metric": PRIMARY_METRIC}
    out["RQ1"] = _verdict_json(
        rq1({g: (cross[g].harmful, metrics[ES][g].harmful) for g in ("C2", "C4")})
    )
    out["RQ2_first"] = {}
    out["RQ2_second"] = {}
    out["RQ3_first"] = {}
    out["RQ3_trend"] = {}
    for m in METRICS:
        by_gate = metrics[m]
        rates = {
            (r, g): getattr(by_gate[g], r)
            for r in ("harmful", "local_harm")
            for g in ("C1", "C2", "C4")
        }
        out["RQ2_first"][m] = _verdict_json(rq2_first(rates))
        original = {k: v.value for k, v in rates.items()}
        out["RQ2_second"][m] = _verdict_json(
            rq2_second(original, {r: sensitivity[m][r].mean for r in ("harmful", "local_harm")})
        )
        out["RQ3_first"][m] = _verdict_json(
            rq3_first(
                useful={g: by_gate[g].useful for g in ("C2", "C4")},
                harmful={g: by_gate[g].harmful for g in ("C2", "C4")},
            )
        )
        d_u = {n: _diff(budget[n][("C4", m)], budget[n][("C2", m)], "useful") for n in BUDGET_NS}
        d_h = {n: _diff(budget[n][("C2", m)], budget[n][("C4", m)], "harmful") for n in BUDGET_NS}
        trend = rq3_trend(d_u, d_h)
        out["RQ3_trend"][m] = _verdict_json(trend) | {
            "d_useful": {str(n): _frac(v) for n, v in d_u.items()},
            "d_harmful": {str(n): _frac(v) for n, v in d_h.items()},
            "non_increasing": dict(trend.monotone),
        }
    return out


def _diff(a: Any, b: Any, name: str) -> Fraction | None:
    x, y = a.metrics[name].mean, b.metrics[name].mean
    return None if x is None or y is None else x - y


# ---------------------------------------------------------------------------
# 직렬화
# ---------------------------------------------------------------------------


def _frac(x: Fraction | int | None) -> dict[str, Any] | None:
    if x is None:
        return None
    f = Fraction(x)
    return {"exact": str(f), "float": float(f)}


def _cell(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def _rate_json(r: Rate) -> dict[str, Any]:
    interval = r.interval
    return {
        "successes": r.successes,
        "n": r.n,
        "excluded": r.excluded,
        "value": _frac(r.value),
        "interval": list(interval) if interval else None,
    }


def _gain_json(g: Gain) -> dict[str, Any]:
    return {"n": g.n, "mean": _frac(g.mean)}


def _metrics_json(m: GateMetrics) -> dict[str, Any]:
    return {
        "accepted": list(m.accepted),
        "harmful": _rate_json(m.harmful),
        "unsupported": _rate_json(m.unsupported),
        "useful": _rate_json(m.useful),
        "gain": _gain_json(m.gain),
        "false_rejection": _rate_json(m.false_rejection),
        "local_harm": _rate_json(m.local_harm),
    }


def _summary_json(s: Summary) -> dict[str, Any]:
    return {
        "mean": _frac(s.mean),
        "low": _frac(s.low),
        "high": _frac(s.high),
        "undefined": s.undefined,
        "n_resample": s.n_resample,
    }


def _verdict_json(v: Verdict) -> dict[str, Any]:
    return {
        "verdict": v.verdict,
        "comparisons": [
            {"name": c.name, "left": _frac(c.left), "right": _frac(c.right), "holds": c.holds}
            for c in v.comparisons
        ],
    }


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
}
METRIC_KO = {CD: "CD", ES: "ES"}
CHECK_KO = {"as_expected": "예상과 같음", "unexpected": "예상과 다름", "uncertain": "불확정"}
NOTES = (
    "C2는 Δ = 0 후보도 채택하므로 neutral이 많으면 C2의 정밀도가 떨어진다(7.8 G).",
    "RQ2 둘째 문장에서 C2·C4는 민감도 분석에서 고정이라 C2·C4 쌍은 항상 같은 결과를 낸다(§8-33).",
    "후보 수가 12개라 구간이 넓다. 게이트 간 비교는 통계 검정 없이 기술적으로 한다(7.8 D).",
    "주요 결론은 proposer 후보로만 낸다. 통제 후보와 C3는 보조로 보고한다(7.8 G).",
    "후보 주제 분포는 기술용이며 게이트 판정과 하위집합 라벨에 쓰지 않는다(7.8 G).",
)
LIMITS = (
    "δ(N)은 칸 층화 없이 재표본하고, C4의 bootstrap은 칸 층화로 재표본한다(7.8 A).",
    "RQ2 민감도 분석은 proposer 입력의 치우침을 재현하지 않는다(7.8 C).",
    "의도된 조건: proposer는 S0가 맞힌 사례를 보지 않고(Dev-A 실패 중 dependency_failure 0개), "
    "빠진 술어 이름을 보며, C1과 같은 S0 Dev-A 실행(반복 0)을 본다(7.8 G).",
    "S0 Dev-A는 Ticket 004에서 실행했고 후보 실행과 시간 간격이 있다(Ticket 005).",
    "Oracle에서 후보는 S0 반복 0과 LLM seed를 공유하고, 그 반복 0이 다수결 다섯 표 중 "
    "하나다(§8-5).",
)


def render_markdown(result: Mapping[str, Any]) -> str:
    meta = result["meta"]
    lines = [
        "# Ticket 005 분석 결과",
        "",
        f"- 사전 등록 sha256: `{meta['preregistration_sha256']}`",
        f"- 분석 코드 커밋: `{meta['analysis_commit']}`, 실행 계획 sha256: `{meta['plan_sha256']}`",
        f"- §8 해석: {meta['section8_version']}",
        f"- Python {meta['python']}, numpy {meta['numpy']}, 재표본 {meta['n_resample']}회, "
        f"bootstrap {meta['n_boot']}회, seed {meta['seed']}",
        f"- 에피소드 {result['episodes']['total']}개, 처분 없이 끝남 "
        f"{result['episodes']['no_disposition']}개, 형식 오류 에피소드 "
        f"{result['episodes']['format_error_episodes'] or '없음'}",
        "",
        "## 결과표",
        "",
        "칸: 채택 수 / 후보 수.",
        "",
    ]
    header = [f"{g} {METRIC_KO[m]}" for g in GATE_NAMES for m in METRICS]
    lines.append("| 행 | " + " | ".join(header) + " |")
    lines.append("| --- |" + " --- |" * len(header))
    for row in result["results_table"]:
        cells = [row["cells"][f"{g}/{m}"] for g in GATE_NAMES for m in METRICS]
        lines.append(f"| {row['row']} | " + " | ".join(cells) + " |")

    for group in GROUPS:
        lines += ["", f"## 지표 ({group})", ""]
        lines.append(
            "| metric | 게이트 | 채택 | Harmful | Unsupported | Useful | Gain | False Rejection "
            "| Local Harm |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for m in METRICS:
            for gate in GATE_NAMES:
                v = result["metrics"][group][m][gate]
                lines.append(
                    f"| {METRIC_KO[m]} | {gate} | {len(v['accepted'])} | "
                    f"{_rate_md(v['harmful'])} | "
                    f"{_rate_md(v['unsupported'])} | {_rate_md(v['useful'])} | "
                    f"{_frac_md(v['gain']['mean'])} (n={v['gain']['n']}) | "
                    f"{_rate_md(v['false_rejection'])} | {_rate_md(v['local_harm'])} |"
                )

    lines += _candidate_md(result) + _controls_md(result) + _budget_md(result)
    lines += ["", "## RQ 판정 (proposer, 주 metric CD)", ""]
    rq = result["rq"]
    lines.append(f"- RQ1: **{VERDICT_KO[rq['RQ1']['verdict']]}** {_comparisons_md(rq['RQ1'])}")
    for key, title in (
        ("RQ2_first", "RQ2 첫째"),
        ("RQ2_second", "RQ2 둘째"),
        ("RQ3_first", "RQ3 첫째"),
        ("RQ3_trend", "RQ3 추세"),
    ):
        for m in METRICS:
            v = rq[key][m]
            line = (
                f"- {title} ({METRIC_KO[m]}): **{VERDICT_KO[v['verdict']]}** {_comparisons_md(v)}"
            )
            if key == "RQ3_trend":
                mono = ", ".join(f"{k} {_bool_md(b)}" for k, b in v["non_increasing"].items())
                line += f"; 비증가(같은 값 허용): {mono}"
            lines.append(line)

    lines += ["", "## Oracle 적합성", ""]
    for m in METRICS:
        f = result["fitness"][m]
        lines.append(
            f"- {METRIC_KO[m]}: {'적합' if f['fit'] else '미달'} "
            f"(부호 {_bool_md(f['sign_met'])}, 크기 {_bool_md(f['size_met'])})"
        )
    lines += ["", "## 해석 주의", ""] + [f"- {n}" for n in NOTES]
    lines += ["", "## 한계", ""] + [f"- {n}" for n in LIMITS]
    return "\n".join(lines) + "\n"


def _candidate_md(result: Mapping[str, Any]) -> list[str]:
    lines = ["", "## 후보별 표", ""]
    lines.append(
        "| 후보 | 집합 | metric | C1 | C2 | C3 | C4 | Oracle Δ [95% 구간] | 하위집합 | "
        "통제 라벨 / 예상 방향 일치 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, c in result["candidates"].items():
        for m in METRICS:
            o = c["oracle"][m]
            gates = [_bool_md(c["gates"][m][g]["accepted"]) for g in GATE_NAMES]
            low, high = o["interval"]
            check = (
                CHECK_KO[o["control_check"]]
                if c["group"] == "control"
                else _bool_md(o.get("effect_match"))
            )
            subset = o["subset"] + (
                f" (큰 칸 {', '.join(o['big_cells'])})" if o["big_cells"] else ""
            )
            lines.append(
                f"| {name} | {c['group']} | {METRIC_KO[m]} | " + " | ".join(gates) + " | "
                f"{_frac_md(o['delta'])} [{_frac_md(low)}, {_frac_md(high)}] | {subset} | {check} |"
            )
    return lines


def _controls_md(result: Mapping[str, Any]) -> list[str]:
    lines = ["", "## 통제 후보 라벨 검증", ""]
    lines.append("| 통제 후보 | 사전 라벨 | CD | ES |")
    lines.append("| --- | --- | --- | --- |")
    for name, c in result["candidates"].items():
        if c["group"] != "control":
            continue
        checks = [CHECK_KO[c["oracle"][m]["control_check"]] for m in METRICS]
        lines.append(f"| {name} | {c['prior_label']} | " + " | ".join(checks) + " |")
    return lines


def _budget_md(result: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for group in GROUPS:
        lines += ["", f"## Budget Curve ({group})", ""]
        lines.append(
            "칸: 채택 후보 비율 평균 / Useful 평균 / Harmful 평균(분모 0 재표본 수). 재표본 평균."
        )
        lines.append("")
        header = [f"{g} {METRIC_KO[m]}" for g in ("C2", "C3", "C4") for m in METRICS]
        lines.append("| N | " + " | ".join(header) + " |")
        lines.append("| --- |" + " --- |" * len(header))
        for n, points in result["budget_curve"][group].items():
            cells = []
            for g in ("C2", "C3", "C4"):
                for m in METRICS:
                    p = points[f"{g}/{m}"]
                    u, h = p["metrics"]["useful"], p["metrics"]["harmful"]
                    cells.append(
                        f"{_frac_md(p['acceptance_rate']['mean'])} / {_frac_md(u['mean'])} "
                        f"({u['undefined']}) / {_frac_md(h['mean'])} ({h['undefined']})"
                    )
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## RQ2 민감도 (C1, 재표본 평균)", ""]
    lines.append("| 집합 | metric | Harmful | Local Harm |")
    lines.append("| --- | --- | --- | --- |")
    for group in GROUPS:
        for m in METRICS:
            s = result["sensitivity"][group][m]
            lines.append(
                f"| {group} | {METRIC_KO[m]} | {_frac_md(s['harmful']['mean'])} "
                f"({s['harmful']['undefined']}) | {_frac_md(s['local_harm']['mean'])} "
                f"({s['local_harm']['undefined']}) |"
            )
    return lines


def _frac_md(x: Mapping[str, Any] | None) -> str:
    return "판정 불가" if x is None else f"{x['float']:.3f}"


def _rate_md(r: Mapping[str, Any]) -> str:
    if r["n"] == 0:
        return f"0/0 판정 불가 (제외 {r['excluded']})"
    low, high = r["interval"]
    return f"{r['successes']}/{r['n']} [{low:.3f}, {high:.3f}] (제외 {r['excluded']})"


def _bool_md(b: bool | None) -> str:
    return "판정 불가" if b is None else ("예" if b else "아니오")


def _comparisons_md(v: Mapping[str, Any]) -> str:
    parts = [f"{c['name']}: {_bool_md(c['holds'])}" for c in v["comparisons"]]
    return "(" + "; ".join(parts) + ")"


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-analysis")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pool-spec", type=Path, default=None, help="default: the frozen pool")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--analysis-commit", default=None)
    parser.add_argument("--plan-sha256", default=None)
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE)
    args = parser.parse_args(argv)

    json_path = args.out_dir / "analysis.json"
    if json_path.exists():
        print(f"error: {json_path} already exists; the analysis runs once", file=sys.stderr)
        return 1
    pool = load_pool_spec(args.pool_spec) if args.pool_spec else load_pool(args.root)
    records = load_records(args.runs_dir, pool)
    noise = EpisodeStore(args.noise, repair=False).records()
    result = run_analysis(
        records,
        noise,
        pool,
        run_info={"analysis_commit": args.analysis_commit, "plan_sha256": args.plan_sha256},
        n_resample=args.n_resample,
        root=args.root,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    json_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (args.out_dir / "analysis.json.sha256").write_text(f"{digest}  analysis.json\n")
    (args.out_dir / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"analysis.json sha256 {digest}")
    return 0


def _s0_skill_id(root: Path) -> str:
    from evidencegate.skills import load_skill

    return load_skill(root / "skills" / "S0" / "SKILL.md").skill_id


if __name__ == "__main__":
    raise SystemExit(main())
