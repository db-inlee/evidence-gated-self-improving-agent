"""원고용 표와 그림 (Ticket 007 §2). `writeup/generated/values.json`만 읽는다.

    uv run python -m evidencegate.writeup_artifacts

- 표 T1~T3: 칸은 템플릿 문자열이고 숫자는 자리표시자 `{{키}}`로만 들어간다. 템플릿에 자리표시자
  밖의 숫자는 이름(C1, L2, RQ-L3-0, Ticket 005 등)뿐이다(테스트로 확인). `tables.json`에 템플릿을,
  `t1.tex`~`t3.tex`에 LaTeX 조각을 쓴다. README 표는 렌더 때 `table_md`로 만든다(`.md` 파일을 쓰지
  않음).
- 그림 F2~F4: `figures.json`에 그림 데이터(쓰인 값마다 키와 정확한 값)를 쓰고, README용 SVG와 논문용
  PDF를 그린다. 날짜 메타데이터를 넣지 않고 SVG 해시 소금을 고정해 같은 환경에서는 바이트까지 같다.
  - F2: 후보 19개 x (L1/L2 6칸, L3 3칸) 칸별 oracle effect estimate. Δ_cell < -τ_cell인 칸(사전
    등록한 Local Harm 칸 조건)에 테두리.
  - F3: 후보마다 L1/L2 전체 Δ 대 L3 전체 Δ. 강조는 규칙으로 정한다: L3 Harmful 확정이거나 L3 점추정
    Δ < -ε.
  - F4: Ticket 005 Budget Curve(proposer). N별 채택 비율, Useful, Harmful 평균(C2·C4 x CD·ES)과
    분모 0 재표본 수.
- `artifacts.sha256`에 생성 파일마다 sha256을 적는다.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from evidencegate.analysis import REPO_ROOT

VALUES_PATH = Path("writeup/generated/values.json")
OUT_DIR = Path("writeup/generated")
PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
# 자리표시자 밖에 올 수 있는 숫자 섞인 이름(Ticket 007 §1 허용 목록 중 표에 쓰는 것)
ALLOWED_NAME = re.compile(
    r"\b(?:RQ-L3-[0-3]|RQ[1-3]|C[1-4]|L[1-3]|S0|Ticket 00[4-7]|T[1-3]|F[1-4])\b"
)
METRICS = (("cd", "CD"), ("es", "ES"))
GATES = ("C1", "C2", "C3", "C4")
FAMILIES = ("bad_deployment", "dependency_failure", "traffic_spike")
L12_CELLS = tuple(f"{f}/{lv}" for f in FAMILIES for lv in ("L1", "L2"))
L3_CELLS = tuple(f"{f}/L3" for f in FAMILIES)
SENTENCES = (
    ("RQ2_first", "RQ2 (a)"),
    ("RQ2_second", "RQ2 (b)"),
    ("RQ3_first", "RQ3 (a)"),
    ("RQ3_trend", "RQ3 trend"),
)
SVG_SALT = "evidencegate-ticket-007"
DARK_CELL = 0.6  # F2: 이보다 |Δ|가 큰 칸은 글자를 흰색으로(읽기용, 값과 무관)

Values = Mapping[str, Mapping[str, str]]
Table = dict[str, Any]


def fill(template: str, values: Values) -> str:
    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"unknown value key {key}")
        return values[key]["display"]

    return PLACEHOLDER.sub(sub, template)


# ---------------------------------------------------------------------------
# 표
# ---------------------------------------------------------------------------


def _v(key: str) -> str:
    return "{{" + key + "}}"


def _t1() -> Table:
    rq = "t005.rq"
    rows = [
        ["RQ1", "C2, C4: ES harm after CD vs ES acceptance", _v(f"{rq}.RQ1.verdict"), ""],
        [
            "RQ2 (a)",
            "harmful and local harm: C1 vs C2, C4",
            _v(f"{rq}.RQ2_first.cd.verdict"),
            _v(f"{rq}.RQ2_first.es.verdict"),
        ],
        [
            "RQ2 (b)",
            "C1 gap under sensitivity resampling",
            _v(f"{rq}.RQ2_second.cd.verdict"),
            _v(f"{rq}.RQ2_second.es.verdict"),
        ],
        [
            "RQ3 (a)",
            "C4 vs C2 at N = " + _v("design.dev_b_size"),
            _v(f"{rq}.RQ3_first.cd.verdict"),
            _v(f"{rq}.RQ3_first.es.verdict"),
        ],
        [
            "RQ3 trend",
            "N from small to large",
            _v(f"{rq}.RQ3_trend.cd.verdict"),
            _v(f"{rq}.RQ3_trend.es.verdict"),
        ],
    ]
    for scope, label in (("all", "all families"), *((f, f) for f in FAMILIES)):
        rows.append(["RQ-L3-0", label, _v(f"t006.rq.RQ_L3_0.{scope}.verdict"), ""])
    rows.append(
        [
            "RQ-L3-1",
            "harmful acceptance, C1 C2 C4",
            _v("t006.rq.RQ_L3_1.cd.verdict"),
            _v("t006.rq.RQ_L3_1.es.verdict"),
        ]
    )
    for sentence, label in SENTENCES:
        cells = []
        for m, _ in METRICS:
            base = f"t006.rq.RQ_L3_2.{sentence}.{m}"
            cells.append(f"{_v(base + '.l12')} → {_v(base + '.l3')} ({_v(base + '.status')})")
        rows.append(["RQ-L3-2", label, *cells])
    rows.append(
        [
            "RQ-L3-3",
            "proposer labels unchanged",
            *(
                _v(f"t006.rq.RQ_L3_3.proposer.{m}.same")
                + " / "
                + _v(f"t006.rq.RQ_L3_3.proposer.{m}.n")
                for m, _ in METRICS
            ),
        ]
    )
    return {
        "id": "t1",
        "title": "Pre-registered verdicts (Ticket 005 and Ticket 006, proposer candidates)",
        "note": "Verdict categories as recorded in the result files. RQ1 crosses the two metrics, "
        "and RQ-L3-0 judges the S0 baseline, so each has one verdict. RQ-L3-2 shows the L1/L2 and "
        "L3 categories side by side.",
        "columns": ["Question", "Scope", "CD", "ES"],
        "rows": rows,
    }


def _t2() -> Table:
    rows = []
    for m, label in METRICS:
        for gate in GATES:
            for oracle, base in (
                ("L1/L2", f"t005.metrics.proposer.{m}.{gate}"),
                ("L3", f"t006.metrics.proposer.{m}.{gate}.l3"),
            ):
                rows.append(
                    [
                        label,
                        gate,
                        oracle,
                        _v(f"t005.metrics.proposer.{m}.{gate}.accepted.count"),
                        _v(f"{base}.harmful.ci"),
                        _v(f"{base}.unsupported.ci"),
                        _v(f"{base}.useful.ci"),
                        _v(f"{base}.gain.mean"),
                        _v(f"{base}.false_rejection.ci"),
                        _v(f"{base}.local_harm.ci"),
                    ]
                )
    return {
        "id": "t2",
        "title": "Pre-registered metrics per gate (proposer candidates)",
        "note": "Gate decisions are the Ticket 005 ones; the oracle is L1/L2 (Ticket 005) or L3 "
        "(Ticket 006). Rates are successes / denominator after excluding uncertain candidates, "
        "with Clopper-Pearson intervals; gain is the mean oracle effect estimate of accepted "
        "candidates.",
        "columns": [
            "Metric",
            "Gate",
            "Oracle",
            "Accepted",
            "Harmful",
            "Unsupported",
            "Useful",
            "Gain",
            "False rejection",
            "Local harm",
        ],
        "rows": rows,
    }


def _t3() -> Table:
    rows = []
    for gate in GATES:
        row = [gate]
        for m, _ in METRICS:
            row += [
                _v(f"t005.metrics.proposer.{m}.{gate}.local_harm"),
                _v(f"t006.local_harm.proposer.{m}.{gate}.l3"),
                _v(f"t006.local_harm.proposer.{m}.{gate}.family_l12"),
            ]
        rows.append(row)
    return {
        "id": "t3",
        "title": "Pre-registered metric: Local Harm Acceptance Rate (proposer candidates)",
        "note": "Accepted candidates with some cell below minus tau, over accepted candidates. "
        "L1/L2 uses the six family-by-level cells, L3 the three family cells, and the auxiliary "
        "column regroups L1/L2 by family. The counts are pre-registered; reading them as "
        "mean-based "
        "gates admitting local regressions is exploratory.",
        "columns": [
            "Gate",
            "CD, L1/L2 cells",
            "CD, L3 cells",
            "CD, L1/L2 by family",
            "ES, L1/L2 cells",
            "ES, L3 cells",
            "ES, L1/L2 by family",
        ],
        "rows": rows,
    }


def build_tables(values: Values) -> dict[str, Table]:
    tables = {"t1": _t1(), "t2": _t2(), "t3": _t3()}
    for table in tables.values():  # 모든 자리표시자가 풀리는지 미리 확인
        for text in (table["title"], table["note"], *(c for row in table["rows"] for c in row)):
            fill(text, values)
    return tables


TEX_REPLACE = (
    ("\\", "\\textbackslash{}"),
    ("&", "\\&"),
    ("%", "\\%"),
    ("$", "\\$"),
    ("#", "\\#"),
    ("_", "\\_"),
    ("{", "\\{"),
    ("}", "\\}"),
    ("~", "\\textasciitilde{}"),
    ("^", "\\textasciicircum{}"),
    ("→", "$\\rightarrow$"),
    ("Δ", "$\\Delta$"),
    ("\u2212", "$-$"),  # 빼기 기호
)


def tex_escape(text: str) -> str:
    out = text
    for old, new in TEX_REPLACE[:10]:  # 특수 문자 먼저(이어서 넣는 $ 명령은 다시 바꾸지 않음)
        out = out.replace(old, new)
    for old, new in TEX_REPLACE[10:]:
        out = out.replace(old, new)
    return out


def table_tex(table: Table, values: Values) -> str:
    cols = table["columns"]
    lines = [
        "% generated by evidencegate.writeup_artifacts from writeup/generated/values.json",
        f"% {tex_escape(fill(table['title'], values))}",
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(tex_escape(c) for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in table["rows"]:
        lines.append(" & ".join(tex_escape(fill(c, values)) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_md(table: Table, values: Values) -> str:
    cols = table["columns"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in table["rows"]:
        lines.append("| " + " | ".join(fill(c, values).replace("|", "/") for c in row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 그림 데이터
# ---------------------------------------------------------------------------


def _candidates(values: Values) -> list[tuple[str, str]]:
    names = sorted(
        {
            k.split(".")[2]
            for k in values
            if k.startswith("t005.candidates.") and k.endswith(".group")
        }
    )
    groups = {n: values[f"t005.candidates.{n}.group"]["exact"] for n in names}
    return [(n, groups[n]) for n in names if groups[n] == "proposer"] + [
        (n, groups[n]) for n in names if groups[n] == "control"
    ]


def _cell(values: Values, key: str, tau_key: str) -> dict[str, Any]:
    exact = values[key]["exact"]
    return {
        "key": key,
        "exact": exact,
        "display": values[key]["display"],
        "tau_key": tau_key,
        "below_minus_tau": Fraction(exact) < -Fraction(values[tau_key]["exact"]),
    }


def figure_data(values: Values) -> dict[str, Any]:
    cands = _candidates(values)
    eps = Fraction(values["design.epsilon"]["exact"])
    f2: dict[str, Any] = {}
    f3: dict[str, Any] = {}
    for m, _ in METRICS:
        rows = []
        for name, _group in cands:
            row = [
                _cell(values, f"t005.candidates.{name}.oracle.{m}.cells.{c}", f"t005.tau.{m}.{c}")
                for c in L12_CELLS
            ]
            row += [
                _cell(
                    values,
                    f"t006.candidates.{name}.oracle_l3.{m}.cells.{c}",
                    f"t006.tau.{m}.L3.{c}",
                )
                for c in L3_CELLS
            ]
            rows.append(row)
        f2[m] = {
            "rows": [n for n, _ in cands],
            "groups": [g for _, g in cands],
            "columns": [*L12_CELLS, *L3_CELLS],
            "cells": rows,
        }
        points = []
        for name, group in cands:
            x_key = f"t005.candidates.{name}.oracle.{m}.delta"
            y_key = f"t006.candidates.{name}.oracle_l3.{m}.delta"
            harm_key = f"t006.candidates.{name}.oracle_l3.{m}.harmful"
            y = Fraction(values[y_key]["exact"])
            points.append(
                {
                    "name": name,
                    "group": group,
                    "x_key": x_key,
                    "x_exact": values[x_key]["exact"],
                    "y_key": y_key,
                    "y_exact": values[y_key]["exact"],
                    "harmful_key": harm_key,
                    "highlight": values[harm_key]["exact"] == "true" or y < -eps,
                }
            )
        f3[m] = {"epsilon_key": "design.epsilon", "points": points}
    ns = [int(n) for n in values["design.budget_ns"]["exact"].strip("[]").split(", ")]
    series: dict[str, Any] = {}
    for gate in ("C2", "C4"):
        for m, _ in METRICS:
            entry: dict[str, list[dict[str, Any]]] = {
                "acceptance_rate": [],
                "useful": [],
                "harmful": [],
            }
            for n in ns:
                base = f"t005.budget_curve.proposer.{n}.{gate}/{m}"
                for name, key in (
                    ("acceptance_rate", f"{base}.acceptance_rate"),
                    ("useful", f"{base}.metrics.useful"),
                    ("harmful", f"{base}.metrics.harmful"),
                ):
                    entry[name].append(
                        {
                            "key": f"{key}.mean",
                            "exact": values[f"{key}.mean"]["exact"],
                            "undefined": int(values[f"{key}.undefined"]["exact"]),
                            "n_resample": int(values[f"{key}.n_resample"]["exact"]),
                        }
                    )
            series[f"{gate}/{m}"] = entry
    return {"f2": f2, "f3": f3, "f4": {"ns": ns, "series": series}}


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.hashsalt": SVG_SALT,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: Any, out_dir: Path, stem: str) -> list[str]:
    fig.savefig(out_dir / f"{stem}.svg", format="svg", metadata={"Date": None})
    fig.savefig(
        out_dir / f"{stem}.pdf", format="pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)
    return [f"{stem}.svg", f"{stem}.pdf"]


def _f(exact: str) -> float | None:
    return None if exact == "null" else float(Fraction(exact))


def _draw_f2(data: Mapping[str, Any], m: str, label: str, out_dir: Path) -> list[str]:
    f2 = data["f2"][m]
    n_rows, n_cols = len(f2["rows"]), len(f2["columns"])
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * n_rows + 1.6))
    grid = [[_f(c["exact"]) or 0.0 for c in row] for row in f2["cells"]]
    image = ax.imshow(grid, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i, row in enumerate(f2["cells"]):
        for j, cell in enumerate(row):
            dark = abs(_f(cell["exact"]) or 0.0) > DARK_CELL
            color = "white" if dark else "black"
            ax.text(j, i, cell["display"], ha="center", va="center", fontsize=5.5, color=color)
            if cell["below_minus_tau"]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.4, ec="black"))
    short = {"bad_deployment": "bad", "dependency_failure": "dep", "traffic_spike": "traffic"}
    ax.set_xticks(
        range(n_cols), [f"{short[c.split('/')[0]]}\n{c.split('/')[1]}" for c in f2["columns"]]
    )
    ax.set_yticks(range(n_rows), f2["rows"])
    ax.tick_params(axis="x", labelsize=6)
    ax.axvline(len(L12_CELLS) - 0.5, color="black", lw=1.2)
    proposers = sum(1 for g in f2["groups"] if g == "proposer")
    ax.axhline(proposers - 0.5, color="black", lw=0.8, ls="--")
    ax.set_title(
        f"Per-cell oracle effect estimate ({label}). Left: L1/L2 cells (Ticket 005); right: L3 "
        f"cells (Ticket 006).\nOutlined: below minus tau_cell. Dashed line: proposer above, "
        f"controls below.",
        fontsize=7,
    )
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="oracle effect estimate")
    fig.tight_layout()
    return _save(fig, out_dir, f"f2_{m}")


def _draw_f3(data: Mapping[str, Any], values: Values, out_dir: Path) -> list[str]:
    eps = float(Fraction(values["design.epsilon"]["exact"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (m, label) in zip(axes, METRICS, strict=True):
        pts = data["f3"][m]["points"]
        xs = [float(Fraction(p["x_exact"])) for p in pts]
        ys = [float(Fraction(p["y_exact"])) for p in pts]
        lo, hi = min(xs + ys) - 0.05, max(xs + ys) + 0.05
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls=":")
        ax.axhline(-eps, color="grey", lw=0.8, ls="--")
        highlighted = sorted(
            (x, y, p["name"]) for p, x, y in zip(pts, xs, ys, strict=True) if p["highlight"]
        )
        number = {name: i + 1 for i, (_, _, name) in enumerate(highlighted)}
        for p, x, y in zip(pts, xs, ys, strict=True):
            filled = p["group"] == "proposer"
            color = "tab:red" if p["highlight"] else "tab:blue"
            ax.scatter(
                [x], [y], s=22, facecolors=color if filled else "none", edgecolors=color, lw=1
            )
            if p["highlight"]:
                ax.annotate(
                    str(number[p["name"]]),
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 2),
                    fontsize=6,
                )
        ax.text(
            0.02,
            0.98,
            "\n".join(f"{i}: {name}" for name, i in number.items()),
            transform=ax.transAxes,
            va="top",
            fontsize=6,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("L1/L2 overall oracle effect estimate")
        ax.set_ylabel("L3 overall oracle effect estimate")
        ax.set_title(f"{label}: filled proposer, open control", fontsize=7)
    fig.suptitle(
        "Red: L3 harmful (confirmed) or L3 point estimate below minus epsilon (dashed). "
        "Dotted: no change.",
        fontsize=7,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "f3")


def _draw_f4(data: Mapping[str, Any], out_dir: Path) -> list[str]:
    f4 = data["f4"]
    ns = f4["ns"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    styles = {"C2/cd": ("tab:orange", "-"), "C4/cd": ("tab:blue", "-")}
    styles |= {"C2/es": ("tab:orange", "--"), "C4/es": ("tab:blue", "--")}
    titles = {
        "acceptance_rate": "accepted share",
        "useful": "Useful (mean)",
        "harmful": "Harmful (mean)",
    }
    for ax, name in zip(axes, titles, strict=True):
        for index, (series, (color, ls)) in enumerate(styles.items()):
            points = f4["series"][series][name]
            xs = [n for n, p in zip(ns, points, strict=True) if p["exact"] != "null"]
            ys = [_f(p["exact"]) for p in points if p["exact"] != "null"]
            ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=3, lw=1, label=series.upper())
            for n, p in zip(ns, points, strict=True):
                if p["undefined"] and p["exact"] != "null":
                    ax.annotate(
                        f"u{p['undefined']} {series.split('/')[1].upper()}",
                        (n, _f(p["exact"]) or 0.0),
                        textcoords="offset points",
                        xytext=(2, 3 + 7 * index),
                        fontsize=5,
                        color=color,
                    )
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("Dev-B size N")
        ax.set_title(titles[name], fontsize=7)
        ax.set_ylim(-0.05, 1.25)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(
        "Budget curve (Ticket 005, proposer). uK: K resamples with denominator zero, left out of "
        "the mean.",
        fontsize=7,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "f4")


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def build_artifacts(values: Values, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(values)
    data = figure_data(values)
    (out_dir / "tables.json").write_bytes(_json_bytes(tables))
    (out_dir / "figures.json").write_bytes(_json_bytes(data))
    written = ["tables.json", "figures.json"]
    for table_id, table in tables.items():
        (out_dir / f"{table_id}.tex").write_text(table_tex(table, values), encoding="utf-8")
        written.append(f"{table_id}.tex")
    _style()
    for m, label in METRICS:
        written += _draw_f2(data, m, label, out_dir)
    written += _draw_f3(data, values, out_dir)
    written += _draw_f4(data, out_dir)
    digests = {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest() for name in written}
    (out_dir / "artifacts.sha256").write_text(
        "".join(f"{digests[n]}  {n}\n" for n in sorted(digests))
    )
    return digests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-writeup-artifacts")
    parser.add_argument("--values", type=Path, default=REPO_ROOT / VALUES_PATH)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / OUT_DIR)
    args = parser.parse_args(argv)
    sha_file = args.values.with_name(args.values.name + ".sha256")
    recorded = sha_file.read_text().split()[0] if sha_file.exists() else None
    actual = hashlib.sha256(args.values.read_bytes()).hexdigest()
    if actual != recorded:
        print(f"error: {args.values} sha256 {actual} does not match {recorded}", file=sys.stderr)
        return 1
    values = json.loads(args.values.read_text(encoding="utf-8"))
    digests = build_artifacts(values, args.out_dir)
    print(f"artifacts in {args.out_dir}: {len(digests)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
