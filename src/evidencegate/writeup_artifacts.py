"""원고용 표와 그림 (Ticket 007 §2). `writeup/generated/values.json`만 읽는다.

    uv run python -m evidencegate.writeup_artifacts

- 표 T4~T7(Ticket 011 §1): T4 009b 사전 등록 판정(RQ9-1~5, 기술용 행 표시), T5 게이트별 지표(CD,
  C2·C4·C5·C5p·C4star), T6 6.6 군집 민감도, T7 010 탐침(exploratory 값만). 형식은 T1~T3과 같다.
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
NEGATIVE_NUMBER = re.compile(r"(?<![\w$])-(?=\d)")
# 자리표시자 밖에 올 수 있는 숫자 섞인 이름(Ticket 007 §1, Ticket 011 §1). writeup_render도
# 이 목록을 그대로 쓴다(두 목록이 같은지 테스트).
# Ticket 011, h(70), δ(420), 2x2 표 이름은 넣지 않는다.
NAME_PATTERNS: tuple[str, ...] = (
    r"\bRQ-L3-[0-3]\b",
    r"\bRQ9-[1-5]\b",
    r"\bRQ[1-3]\b",
    r"\bC[1-5]\b",  # C5 prime, C4 star의 C5·C4도 지운다
    r"\bL[1-3]\b",
    r"\bS0\b",
    r"\bE1\b",  # 탐색 계산의 이름
    r"\bT[1-7]\b",
    r"\bF[1-4]\b",
    r"\bP[0-6]\b",  # 010 탐침 변형
    r"\bTicket 00[1-8]\b",  # 008: 닫힌 ALFWorld 파일럿(2026-11-04)
    r"\bTicket 009b?\b",
    r"\bTicket 010\b",
    r"\bticket-00[1-8]\b",
    r"\bticket-009b?\b",
    r"\bticket-010\b",
    r"\bcand-\d{3}\b",
)
ALLOWED_NAME = re.compile("|".join(f"(?:{p})" for p in NAME_PATTERNS))
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
                + "/"
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


E1_FIELDS = ("l12", "e1", "l3", "residual_e1", "residual_l12")
E1_COLUMNS = ["Δ L1/L2", "E1", "Δ L3", "Δ L3 − E1", "Δ L3 − Δ L1/L2"]  # noqa: RUF001 (빼기 기호)
E1_NOTE = (
    "Exploratory. E1 approximates the overall oracle effect estimate if L2-like incidents made up "
    "each whole family (mean of the three L2 cell estimates); definition fixed in "
    "writeup/inputs/e1_definition.txt before computing. No intervals or tests."
)


def _e1_row(name: str, m: str) -> list[str]:
    return [_v(f"e1.{name}.{m}.{field}") for field in E1_FIELDS]


def _e1_tables(values: Values) -> dict[str, Table]:
    cands = _candidates(values)
    proposers = [n for n, g in cands if g == "proposer"]
    controls = [n for n, g in cands if g == "control"]
    short = {"bad_deployment": "bad", "dependency_failure": "dep", "traffic_spike": "traffic"}
    return {
        "e1": {
            "id": "e1",
            "title": "E1 (exploratory): proposer candidates, CD",
            "note": E1_NOTE,
            "columns": ["Candidate", *E1_COLUMNS],
            "rows": [[n, *_e1_row(n, "cd")] for n in proposers],
        },
        "e1_family": {
            "id": "e1_family",
            "title": "E1 (exploratory): per-family L2 cell estimate next to the L3 cell estimate, "
            "proposer candidates, CD",
            "note": E1_NOTE,
            "columns": ["Candidate", *(f"{short[f]}: L2 → L3" for f in FAMILIES)],
            "rows": [
                [
                    n,
                    *(
                        f"{_v(f'e1.{n}.cd.family.{f}.l2')} → {_v(f'e1.{n}.cd.family.{f}.l3')}"
                        for f in FAMILIES
                    ),
                ]
                for n in proposers
            ],
        },
        "e1_appendix": {
            "id": "e1_appendix",
            "title": "E1 (exploratory): controls on CD, and every candidate on ES",
            "note": E1_NOTE,
            "columns": ["Candidate", "Metric", *E1_COLUMNS],
            "rows": [[n, "CD", *_e1_row(n, "cd")] for n in controls]
            + [[n, "ES", *_e1_row(n, "es")] for n, _ in cands],
        },
    }


def _rate(base: str) -> str:
    return _v(f"{base}.successes") + "/" + _v(f"{base}.n")


def _afkn(base: str) -> str:
    return ", ".join(f"{f} {_v(f'{base}.{f}')}" for f in ("a", "f", "k", "n"))


def _rq9_5_counts(base: str) -> str:
    """사전 등록 6.5의 m, d, s1(C5만의 Severe), s2(C4*만의 Severe). 칸에 숫자 이름을 쓰지 않는다."""
    return (
        f"admitted {_v(base + '.m')}, set difference {_v(base + '.d')}, "
        f"Severe in C5 only {_v(base + '.s1')}, in C4* only {_v(base + '.s2')}"
    )


def _t4() -> Table:
    rq = "t009b.rq"
    s1s2 = _rq9_5_counts(f"{rq}.rq9_5")
    rows = [
        [
            "RQ9-1",
            "C4-accepted candidates: Severe Local Harm among those C5 rejects and those it keeps "
            "(one-sided Fisher test)",
            _afkn(f"{rq}.rq9_1"),
            "p = " + _v(f"{rq}.rq9_1.p"),
            _v(f"{rq}.rq9_1.verdict"),
        ],
        [
            "RQ9-1 (descriptive)",
            "the same table with the Local Harm label",
            _afkn(f"{rq}.rq9_1_local_harm"),
            "p = " + _v(f"{rq}.rq9_1_local_harm.p"),
            _v(f"{rq}.rq9_1_local_harm.verdict"),
        ],
        [
            "RQ9-2",
            "clean Useful C4 candidates that C5 rejects",
            _v(f"{rq}.rq9_2.u") + " of " + _v(f"{rq}.rq9_2.useful_c4"),
            "",
            _v(f"{rq}.rq9_2.verdict"),
        ],
        [
            "RQ9-2 (descriptive)",
            "all Useful C4 candidates that C5 rejects",
            _v(f"{rq}.rq9_2_all_useful.u") + " of " + _v(f"{rq}.rq9_2_all_useful.useful_c4"),
            "",
            _v(f"{rq}.rq9_2_all_useful.verdict"),
        ],
        [
            "RQ9-3",
            "Severe among C4-accepted candidates, Clopper\u2013Pearson interval",
            _v(f"{rq}.rq9_3.k") + "/" + _v(f"{rq}.rq9_3.n"),
            _v(f"{rq}.rq9_3.interval"),
            _v(f"{rq}.rq9_3.verdict"),
        ],
        [
            "RQ9-3 (descriptive)",
            "the same for C2",
            _v(f"{rq}.rq9_3_c2.k") + "/" + _v(f"{rq}.rq9_3_c2.n"),
            _v(f"{rq}.rq9_3_c2.interval"),
            _v(f"{rq}.rq9_3_c2.verdict"),
        ],
        [
            "RQ9-5",
            "C5 against the strict mean gate C4* that admits as many candidates "
            "(hypergeometric test on the set difference)",
            s1s2,
            "p = " + _v(f"{rq}.rq9_5.p"),
            _v(f"{rq}.rq9_5.verdict"),
        ],
        [
            "RQ9-5 (descriptive)",
            "count rule: Severe admitted by C5 and by C4*",
            "C5 " + _v(f"{rq}.rq9_5.severe_c5") + ", C4* " + _v(f"{rq}.rq9_5.severe_c4_star"),
            "",
            _v(f"{rq}.rq9_5.verdict_count"),
        ],
    ]
    return {
        "id": "t4",
        "title": "Pre-registered verdicts (Ticket 009b, proposer candidates, CD)",
        "note": "Verdict categories as recorded in the result file. Rows marked descriptive were "
        "pre-registered as descriptive companions and are not verdicts of the study.",
        "columns": ["Question", "Scope", "Counts", "Test", "Verdict"],
        "rows": rows,
    }


T5_GATES = (("C2", "C2"), ("C4", "C4"), ("C5", "C5"), ("C5\u2032", "C5p"), ("C4*", "C4star"))
T5_RATES = (
    ("Severe", "severe"),
    ("Local harm", "local_harm"),
    ("Harmful", "harmful"),
    ("Unsupported", "unsupported"),
    ("Useful", "useful"),
    ("Clean Useful", "clean_useful"),
)


def _t5() -> Table:
    rows = []
    for label, key in T5_GATES:
        base = f"t009b.metrics.{key}.cd"
        rows.append(
            [
                label,
                _v(f"{base}.accepted.count"),
                *(_rate(f"{base}.{m}") for _, m in T5_RATES),
                _v(f"{base}.gain"),
                _rate(f"{base}.false_rejection"),
            ]
        )
    return {
        "id": "t5",
        "title": "Pre-registered metrics per gate (Ticket 009b, proposer candidates, CD)",
        "note": "Counts over the candidates each gate accepted, scored on the Oracle split. "
        "C5\u2032 and C4* are the comparison gates of the pre-registration.",
        "columns": ["Gate", "Accepted", *(name for name, _ in T5_RATES), "Gain", "False rejection"],
        "rows": rows,
    }


def _t6() -> Table:
    cs = "t009b.cluster_sensitivity"
    s1s2 = _rq9_5_counts(f"{cs}.rq9_5")
    rows = [
        ["Representatives (one per idea group)", _v("derived.t009b.cluster_groups"), "", ""],
        [
            "RQ9-1 on the representatives (sensitivity verdict)",
            _afkn(f"{cs}.rq9_1"),
            "p = " + _v(f"{cs}.rq9_1.p"),
            _v(f"{cs}.rq9_1.verdict"),
        ],
        [
            "RQ9-3 on the representatives (descriptive)",
            _v(f"{cs}.rq9_3.k") + "/" + _v(f"{cs}.rq9_3.n"),
            _v(f"{cs}.rq9_3.interval"),
            _v(f"{cs}.rq9_3.verdict"),
        ],
        [
            "RQ9-5 on the representatives (descriptive)",
            s1s2,
            "p = " + _v(f"{cs}.rq9_5.p"),
            _v(f"{cs}.rq9_5.verdict"),
        ],
        [
            "Sensitivity verdict differs from the main verdict",
            _v(f"{cs}.differs_from_main"),
            "",
            "",
        ],
    ]
    return {
        "id": "t6",
        "title": "Pre-registered cluster sensitivity "
        "(Ticket 009b, one representative per idea group)",
        "note": "The main verdicts stay those of the full pool; this table repeats them on the "
        "earliest candidate of each idea group.",
        "columns": ["Quantity", "Counts", "Test", "Verdict"],
        "rows": rows,
    }


def _t7() -> Table:
    rows = []
    for i in range(7):
        b = f"t010.variants.{i}"
        rows.append(
            [
                _v(f"{b}.name"),
                _v(f"{b}.field"),
                _v(f"{b}.different_disposition") + " of " + _v(f"{b}.episodes"),
                _v(f"{b}.moved"),
                _v(f"{b}.different_tool_set"),
                _v(f"{b}.traffic_episodes"),
                _v(f"{b}.mean_tool_calls"),
            ]
        )
    rows.append(
        [
            "S0",
            "baseline (Dev-A records)",
            "",
            "",
            "",
            _v("t010.s0.traffic_episodes"),
            _v("t010.s0.mean_tool_calls"),
        ]
    )
    return {
        "id": "t7",
        "title": "Harness knob probe on Dev-A (Ticket 010, exploratory)",
        "note": "Exploratory. A variant moved if at least "
        + _v("t010.rule.k")
        + " of "
        + _v("t010.rule.episodes")
        + " dispositions differed from S0; the rule fixed before the run asked for "
        + _v("t010.rule.m")
        + " moved variants. P0 caps tool calls at "
        + _v("t010.variants.0.max_tool_calls")
        + " as a wiring check and is not counted; it needed "
        + _v("t010.rule.p0_min")
        + ". Decision: "
        + _v("t010.decision.verdict")
        + ".",
        "columns": [
            "Variant",
            "Field",
            "Dispositions differing from S0",
            "Moved",
            "Tool set differs",
            "Traffic tool called",
            "Mean tool calls",
        ],
        "rows": rows,
    }


def build_tables(values: Values) -> dict[str, Table]:
    tables = {
        "t1": _t1(),
        "t2": _t2(),
        "t3": _t3(),
        **_e1_tables(values),
        "t4": _t4(),
        "t5": _t5(),
        "t6": _t6(),
        "t7": _t7(),
    }
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
    # 수의 음호: 앞이 낱말 글자가 아닌 하이픈 + 숫자(이름 속 하이픈 RQ-L3-0, cand-015는 그대로)
    return NEGATIVE_NUMBER.sub("$-$", out)


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
