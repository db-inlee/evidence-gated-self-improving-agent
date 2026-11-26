"""Ticket 013 결과를 문서 값(`t013.`, kind exploratory)으로 바꾼다. `evidencegate.writeup`이 부른다.

- 입력: `results/ticket-013/`의 본 모의(`sim.json`), 이득 보정, 009b 재생 두 개. sha256 확인은
  `writeup.load_follow_up`이 한다.
- 문서가 인용하는 부분만 편다(sim.json 전체는 펴지 않음). 비율은 소수 4자리, 귀무 라벨률은 6자리,
  상관은 3자리로 보이며, 정확한 값(repr)도 함께 둔다. 반올림은 half-up(0에서 먼 쪽)이다.
- 이득(gain_per_proposal)은 **보정 파일에서만** 읽는다. sim.json의 이득 값은 무효다(보고서
  2026-11-26-01 §9).
"""

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Any

from evidencegate.t013 import sim_core as sc

KIND = "exploratory"
GATES = ("C4", "C5", "C6_IUT", "SAGE_l1", "SAGE_l2", "SAGE_cell_l1", "SAGE_cell_l2", "credit")
M_KEYS = {"1/20": "m05", "1/10": "m10"}
REPLAY_GATES = ("C4", "C5", "SAGE_l1", "SAGE_l2", "SAGE_cell_l1", "SAGE_cell_l2", "credit")
C6_KEYS = {
    "C4": "C4",
    "C5": "C5",
    "C6_IUT_tango_m1/20": "tango_m05",
    "C6_IUT_exact_conditional_m1/20": "exact_m05",
    "C6_IUT_tango_m1/10": "tango_m10",
    "C6_IUT_exact_conditional_m1/10": "exact_m10",
}
INADMISSIBLE = (
    "null_all_cells|-",
    *(
        f"{s}|{p}"
        for s in (
            "no_global_one_cell_minus_m",
            "global_up_one_cell_minus_m",
            "global_up_one_cell_minus_2m",
        )
        for p in ("realistic", "worst")
    ),
)

Value = dict[str, str]


def _num(x: float | None, digits: int) -> Value:
    if x is None:
        return {"exact": "null", "display": "n/a", "kind": KIND}
    shown = Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return {"exact": repr(x), "display": f"{shown:.{digits}f}", "kind": KIND}


def _int(x: int) -> Value:
    return {"exact": str(x), "display": str(x), "kind": KIND}


def _text(x: str) -> Value:
    return {"exact": x, "display": x, "kind": KIND}


def _where(key: str | None) -> str:
    if key is None:
        return "n/a"
    name, placement = key.split("|")
    return name if placement == "-" else f"{name}, {placement}"


def _worst(metrics: Mapping[str, Any], m: str, dep: str, gate: str) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    for scen in INADMISSIBLE:
        row = metrics.get(f"{m}|{scen}|{dep}|{gate}")
        if row is None:
            continue
        fa = row["false_accept"]
        if (
            fa["eligible"]
            and fa["estimate"] is not None
            and (best is None or fa["estimate"] > best[0])
        ):
            best = (fa["estimate"], fa["se"])
    return best


def _decision(sim: Mapping[str, Any]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    go_count = {}
    for b in ("a", "b"):
        go_count[b] = 0
        for c in sim["decision"][b]:
            base = f"t013.dec.{b}.{c['gate']}.n{c['n_c']}.{M_KEYS[c['m_c']]}"
            fa, power = c["false_accept"] or {}, c["useful_power"] or {}
            status = "Go" if c["counts_toward_go"] else ("borderline" if c["borderline"] else "-")
            go_count[b] += c["counts_toward_go"]
            out |= {
                f"{base}.false_accept": _num(fa.get("estimate"), 4),
                f"{base}.false_accept_se": _num(fa.get("se"), 4),
                f"{base}.false_accept_where": _text(_where(fa.get("scenario"))),
                f"{base}.useful_power": _num(power.get("estimate"), 4),
                f"{base}.useful_power_se": _num(power.get("se"), 4),
                f"{base}.episodes": _int(c["episodes"]),
                f"{base}.status": _text(status),
            }
    out |= {f"t013.go.{b}": _int(n) for b, n in go_count.items()}
    out["t013.go.any"] = _text("yes" if sim["decision"]["go"] else "no")
    return out


def _coverage(sim: Mapping[str, Any]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    for block, rows in sim["coverage"].items():
        b, n, coupling = block.split("_")
        if coupling != "none":
            continue
        for m, mk in M_KEYS.items():
            keys = [k for k in rows if k.startswith(m + "|") and k.endswith("|main")]
            for part, name in (
                ("tango", "tango"),
                ("exact_conditional", "exact"),
                ("exact_d_zero", "d0"),
            ):
                vals = [rows[k][part]["estimate"] for k in keys]
                out[f"t013.cov.{b}.n{n}.{mk}.{name}.min"] = _num(min(vals), 4)
                out[f"t013.cov.{b}.n{n}.{mk}.{name}.max"] = _num(max(vals), 4)
    for name in ("tango", "exact", "d0"):
        for end, pick in (("min", min), ("max", max)):
            vals = [float(v["exact"]) for k, v in out.items() if k.endswith(f".{name}.{end}")]
            out[f"t013.cov.all.{name}.{end}"] = _num(pick(vals), 4)
    return out


def _sensitivity(sim: Mapping[str, Any]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    for b in ("a", "b"):
        for c in sim["decision"][b]:
            if not c["counts_toward_go"]:
                continue
            g, n, m = c["gate"], c["n_c"], c["m_c"]
            base = f"t013.sens.{b}.{g}.n{n}.{M_KEYS[m]}"
            for setting, block, dep in (
                ("main", f"{b}_{n}_none", "main"),
                ("s0_independent", f"{b}_{n}_none", "s0_independent"),
                ("coupled", f"{b}_{n}_coupled", "main"),
            ):
                metrics = sim["metrics"][block]
                worst = _worst(metrics, m, dep, g)
                power = metrics[f"{m}|useful_representative|-|{dep}|{g}"]["acceptance"]
                out[f"{base}.{setting}.false_accept"] = _num(worst[0] if worst else None, 4)
                out[f"{base}.{setting}.useful_power"] = _num(power["estimate"], 4)
    return out


def _gains(sim: Mapping[str, Any], gain: Mapping[str, Any]) -> dict[str, Value]:
    """이득은 보정 파일에서만. Go 조합의 경험 분포와 대표 Useful 시나리오(주 의존 조건)."""
    out: dict[str, Value] = {}
    table = gain["gain_per_proposal"]
    for b in ("a", "b"):
        for c in sim["decision"][b]:
            if not c["counts_toward_go"]:
                continue
            g, n, m = c["gate"], c["n_c"], c["m_c"]
            base = f"t013.gain.{b}.{g}.n{n}.{M_KEYS[m]}"
            for scen, name in (("empirical", "empirical"), ("useful_representative", "useful")):
                row = table[f"{b}_{n}_none"][f"{m}|{scen}|-|main|{g}"]
                out[f"{base}.{name}"] = _num(row["estimate"], 4)
                out[f"{base}.{name}_se"] = _num(row["se"], 4)
    return out


def _misc(sim: Mapping[str, Any]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    verdict = sim["gate"]["verdict"]
    out["t013.gate.reproduced"] = _text("yes" if verdict["reproduced"] else "no")
    for label in ("local_harm", "severe"):
        for part in ("rate", "registered", "bound"):
            out[f"t013.gate.{label}.{part}"] = _num(verdict[label][part], 6)
    demo = sim["metrics"]["a_70_none"]
    for g, name in (
        ("C6_IUT_percentile", "percentile"),
        ("C6_IUT", "tango"),
        ("C6_IUT_exact", "exact"),
        ("C4", "C4"),
    ):
        row = demo[f"1/20|null_all_cells|-|main|{g}"]["acceptance"]
        out[f"t013.demo.{name}"] = _num(row["estimate"], 4)
        out[f"t013.demo.{name}_se"] = _num(row["se"], 4)
    sims = sim["correlations"]["simulated"]
    for fam in ("dependency_failure", "traffic_spike"):
        for coupling in ("coupled", "none"):
            vals = [v[fam] for k, v in sims.items() if k.endswith(coupling) and v[fam] is not None]
            out[f"t013.phi.{coupling}.{fam}.min"] = _num(min(vals), 3)
            out[f"t013.phi.{coupling}.{fam}.max"] = _num(max(vals), 3)
    for fam, v in sim["correlations"]["009b_calibration_descriptive"].items():
        out[f"t013.phi.009b.{fam}"] = _num(v, 3)
    for b in ("a", "b"):
        orc = sim["oracle_label_uncertainty"][f"{b}_70_none"]
        for scen, name in (
            ("null_all_cells", "null"),
            ("useful_representative", "useful"),
            ("empirical", "empirical"),
        ):
            out[f"t013.orc.{b}.{name}"] = _num(orc[f"1/20|{scen}|-|main"]["estimate"], 4)
    infeasible = [
        f"baseline {b}, m_c {float(Fraction(m)):.2f}, {k.replace('|', ', ')}"
        for b in ("a", "b")
        for m, rows in sim["baselines"][b]["scenarios"].items()
        for k, v in rows.items()
        if not v["feasible"]
    ]
    out["t013.infeasible.count"] = _int(len(infeasible))
    out["t013.infeasible.list"] = _text("; ".join(infeasible) or "none")
    for b in ("a", "b"):
        info = sim["baselines"][b]
        out[f"t013.base.{b}.loss_cell.realistic"] = _text(info["loss_cell"]["realistic"])
        out[f"t013.base.{b}.loss_cell.worst"] = _text(info["loss_cell"]["worst"])
        out[f"t013.base.{b}.room_cells"] = _text(", ".join(info["room_cells"]))
    for key in ("pools", "n_boot", "candidates"):
        out[f"t013.settings.{key}"] = _int(sim["settings"][key])
    out["t013.settings.plan_sha256"] = _text(sim["settings"]["plan_sha256"])
    out["t013.settings.addendum_sha256"] = _text(sim["settings"]["addendum_sha256"])
    for m in sim["settings"]["m_c"]:
        out[f"t013.margin.{M_KEYS[m]}"] = _num(float(Fraction(m)), 2)
    for n in sim["settings"]["n_c"]:
        out[f"t013.nc.{n}"] = _int(n)
    out |= {
        "t013.threshold.false_accept": _num(sc.FALSE_ACCEPT_AT_MOST, 2),
        "t013.threshold.useful_power": _num(sc.USEFUL_POWER_AT_LEAST, 1),
        "t013.threshold.episodes": _int(sc.EPISODES_AT_MOST),
        "t013.threshold.borderline_se": _int(sc.BORDERLINE_SE),
        "t013.param.alpha": _num(float(sc.ALPHA), 2),
        "t013.param.sage_tau": _num(sc.SAGE_TAU, 2),
        "t013.param.credit_eps": _num(sc.CREDIT_EPS, 3),
        "t013.param.coupling_rho": _num(sc.COUPLING_RHO, 1),
        "t013.param.oracle_per_cell": _int(sc.ORACLE_PER_CELL),
        "t013.param.nominal_coverage": _num(1 - float(sc.ALPHA), 2),
    }
    return out


def _replays(replay: Mapping[str, Any], c6: Mapping[str, Any]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    for pool, short in (("all", "all"), ("representatives", "reps")):
        for g in REPLAY_GATES:
            row = replay["pools"][pool][g]
            for part in ("accepted", "severe", "useful", "clean_useful", "candidates"):
                out[f"t013.replay.{short}.{g}.{part}"] = _int(row[part])
        for part, n in replay["labels_in_pool"][pool].items():
            out[f"t013.replay.{short}.labels.{part}"] = _int(n)
        for g, key in C6_KEYS.items():
            row = c6["pools"][pool][g]
            for part in ("accepted", "severe", "useful", "clean_useful"):
                out[f"t013.c6.{short}.{key}.{part}"] = _int(row[part])
    return out


def t013_values(
    sim: Mapping[str, Any],
    gain: Mapping[str, Any],
    replay: Mapping[str, Any],
    c6: Mapping[str, Any],
) -> dict[str, Value]:
    return {
        **_decision(sim),
        **_coverage(sim),
        **_sensitivity(sim),
        **_gains(sim, gain),
        **_misc(sim),
        **_replays(replay, c6),
    }
