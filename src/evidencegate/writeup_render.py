"""원고 렌더러와 검사 도구 (Ticket 007 §1 검사 1~4).

    uv run python -m evidencegate.writeup_render render   # writeup/README.template → README.md
    uv run python -m evidencegate.writeup_render check    # 템플릿·README·paper/*.tex 검사

자리표시자: Markdown 원본은 `{{키}}`와 `{{table:t1}}`, LaTeX 원본은 `\\R{키}`. 키는
`writeup/generated/values.json`에 있어야 한다.

검사(원본 기준):
- number: 자리표시자 밖의 숫자는 허용 목록(이름과 식별자)만. 코드(펜스 블록, 인라인 코드),
  링크 대상, LaTeX 주석과 경로·인용·치수 인자는 보지 않는다.
- key: 자리표시자의 키가 values.json(표는 tables의 id)에 있다.
- placement: kind = exploratory 값은 제목에 Exploratory 또는 Appendix가 들어간 절 안에서만.
- banned: 금지 표현(단어 경계, 대소문자 무시). 용어 허용 목록은 `Evidence-Safe` 하나이고, 큰따옴표
  안 원문 인용은 뺀다. 렌더 결과에도 같은 검사를 한다.
README.md는 템플릿 렌더 결과와 바이트까지 같아야 한다.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.writeup_artifacts import build_tables, fill, table_md

TEMPLATE_PATH = Path("writeup/README.template")
README_PATH = Path("README.md")
VALUES_PATH = Path("writeup/generated/values.json")
TABLES_PATH = Path("writeup/generated/tables.json")
PAPER_DIR = Path("paper")

# Ticket 007 원칙 3. 순서대로 긴 것부터 찾는다.
BANNED = (
    "true effect",
    "demonstrates that",
    "first to",
    "proves",
    "guarantees",
    "first",
    "significant",
    "statistically",
    "robust",
    "safe",
    "always",
    "never",
)
ALLOWED_TERMS = ("Evidence-Safe",)
BANNED_RE = re.compile(
    r"\b(?:" + "|".join(p.replace(" ", r"\s+") for p in BANNED) + r")\b", re.IGNORECASE
)
EXPLORATORY_SECTION = re.compile(r"exploratory|appendix", re.IGNORECASE)

# 자리표시자 밖에 올 수 있는 숫자 섞인 이름과 식별자(Ticket 007 §1 허용 목록)
ALLOWED_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\b10\.\d{4,9}/[^\s,;)]+",  # DOI
        r"\b\d{4}\.\d{4,5}(?:v\d+)?\b",  # arXiv ID
        r"\bRQ-L3-[0-3]\b",
        r"\bRQ[1-3]\b",
        r"\bC[1-4]\b",
        r"\bL[1-3]\b",
        r"\bS0\b",
        r"\bE1\b",  # 탐색 계산의 이름
        r"\bT[1-3]\b",
        r"\bF[1-4]\b",
        r"\bTicket 00[1-7]\b",
        r"\bticket-00[1-7]\b",
        r"\bcand-\d{3}\b",
        r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b",  # 커밋·파일 해시
        r"\b20(?:1[5-9]|2[0-6])\b",  # 연도
        r"§\s*\d+(?:-\d+)*",
        r"\bSection\s+\d+(?:\.\d+)*",
    )
)
MD_LINE_START = (re.compile(r"^\s*\d+[.)]\s"), re.compile(r"^#{1,6}\s+\d+(?:\.\d+)*\.?\s"))
MD_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
TEX_PLACEHOLDER = re.compile(r"\\R\{([^{}]+)\}")
TEX_EXEMPT_ARGS = re.compile(
    r"\\(?:cite[tp]?|ref|eqref|label|input|include|includegraphics|url|href|usepackage|"
    r"documentclass|bibliography|bibliographystyle)\*?(?:\[[^\]]*\])?\{[^{}]*\}"
)
TEX_DIMENSION = re.compile(r"-?\d*\.?\d+\s*(?:\\(?:line|text|column)width|pt|mm|cm|in|em|ex)\b")


@dataclass(frozen=True)
class Violation:
    rule: str  # number / key / placement / banned / readme
    line: int
    text: str


# ---------------------------------------------------------------------------
# 원본 정리
# ---------------------------------------------------------------------------


def _md_lines(text: str) -> list[str]:
    """펜스 코드 블록 줄은 빈 줄로(줄 번호 유지), 인라인 코드와 링크 대상은 지운다."""
    out = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append("")
            continue
        if fenced:
            out.append("")
            continue
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        out.append(line)
    return out


def _tex_lines(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = re.sub(r"(?<!\\)%.*", "", line)
        out.append(TEX_EXEMPT_ARGS.sub("", line))
    return out


def _strip_quotes(line: str) -> str:
    line = re.sub(r"“[^”]*”", "", line)
    line = re.sub(r'"[^"]*"', "", line)
    return re.sub(r"``.*?''", "", line)


# ---------------------------------------------------------------------------
# 검사
# ---------------------------------------------------------------------------


def _banned_in(line: str) -> list[str]:
    for term in ALLOWED_TERMS:
        line = line.replace(term, "")
    return [m.group(0) for m in BANNED_RE.finditer(_strip_quotes(line))]


def check_source(text: str, fmt: str, values: Mapping[str, Mapping[str, str]]) -> list[Violation]:
    if fmt not in ("md", "tex"):
        raise ValueError(f"unknown format {fmt!r}")
    lines = _md_lines(text) if fmt == "md" else _tex_lines(text)
    placeholder = MD_PLACEHOLDER if fmt == "md" else TEX_PLACEHOLDER
    table_ids = set(build_tables(values))
    out: list[Violation] = []
    headings: list[tuple[int, str]] = []  # md: (수준, 제목)
    tex_section = ""
    tex_subsection = ""
    appendix = False
    for number, line in enumerate(lines, start=1):
        if fmt == "md":
            heading = re.match(r"^(#{1,6})\s+(.*)", line)
            if heading:
                level = len(heading.group(1))
                headings = [h for h in headings if h[0] < level] + [(level, heading.group(2))]
            in_exploratory = any(EXPLORATORY_SECTION.search(h[1]) for h in headings)
        else:
            if re.search(r"\\appendix\b", line):
                appendix = True
            section = re.search(r"\\section\*?\{([^{}]*)\}", line)
            if section:
                tex_section, tex_subsection = section.group(1), ""
            subsection = re.search(r"\\subsection\*?\{([^{}]*)\}", line)
            if subsection:
                tex_subsection = subsection.group(1)
            in_exploratory = appendix or bool(
                EXPLORATORY_SECTION.search(tex_section)
                or EXPLORATORY_SECTION.search(tex_subsection)
            )

        for match in placeholder.finditer(line):
            key = match.group(1)
            if key.startswith("table:"):
                if key.removeprefix("table:") not in table_ids:
                    out.append(Violation("key", number, key))
                continue
            if key not in values:
                out.append(Violation("key", number, key))
            elif values[key]["kind"] == "exploratory" and not in_exploratory:
                out.append(Violation("placement", number, key))

        bare = placeholder.sub("", line)
        for word in _banned_in(bare):
            out.append(Violation("banned", number, word))
        if fmt == "tex":
            bare = TEX_DIMENSION.sub("", bare)
        for pattern in ALLOWED_PATTERNS:
            bare = pattern.sub("", bare)
        if fmt == "md":
            for pattern in MD_LINE_START:
                bare = pattern.sub("", bare)
        if re.search(r"\d", bare):
            out.append(Violation("number", number, line.strip()))
    return out


def check_rendered(text: str) -> list[Violation]:
    """렌더 결과의 금지 표현(값의 표시 문구까지 포함). 코드는 보지 않는다."""
    out = []
    for number, line in enumerate(_md_lines(text), start=1):
        for word in _banned_in(line):
            out.append(Violation("banned", number, word))
    return out


# ---------------------------------------------------------------------------
# 렌더
# ---------------------------------------------------------------------------


def render_markdown(
    template: str, values: Mapping[str, Mapping[str, str]], tables: Mapping[str, Any]
) -> str:
    def table(match: re.Match[str]) -> str:
        table_id = match.group(1)
        if table_id not in tables:
            raise KeyError(f"unknown table {table_id}")
        return table_md(tables[table_id], values).rstrip("\n")

    with_tables = re.sub(r"\{\{table:([^{}]+)\}\}", table, template)
    return fill(with_tables, values)


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def _load(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    values_path = root / VALUES_PATH
    recorded = values_path.with_name("values.json.sha256").read_text().split()[0]
    actual = hashlib.sha256(values_path.read_bytes()).hexdigest()
    if actual != recorded:
        raise ValueError(f"{values_path} sha256 {actual} does not match {recorded}")
    values = json.loads(values_path.read_text(encoding="utf-8"))
    tables = json.loads((root / TABLES_PATH).read_text(encoding="utf-8"))
    return values, tables


def _report(path: Path, root: Path, violations: Sequence[Violation]) -> None:
    for v in violations:
        print(f"{path.relative_to(root)}:{v.line}: {v.rule}: {v.text}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-writeup-render")
    parser.add_argument("command", choices=("render", "check"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.root
    try:
        values, tables = _load(root)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    template_path = root / TEMPLATE_PATH
    readme_path = root / README_PATH

    if args.command == "render":
        if not template_path.exists():
            print(f"error: no template at {template_path}", file=sys.stderr)
            return 1
        readme_path.write_text(
            render_markdown(template_path.read_text(encoding="utf-8"), values, tables),
            encoding="utf-8",
        )
        print(f"rendered {readme_path.relative_to(root)}")
        return 0

    failed = False
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        found = check_source(template, "md", values)
        _report(template_path, root, found)
        rendered = render_markdown(template, values, tables)
        found_rendered = check_rendered(rendered)
        _report(readme_path, root, found_rendered)
        mismatch = not readme_path.exists() or readme_path.read_text(encoding="utf-8") != rendered
        if mismatch:
            print(f"{README_PATH}: readme: differs from the rendered template", file=sys.stderr)
        failed = bool(found or found_rendered or mismatch)
    else:
        print(f"note: no template at {TEMPLATE_PATH}; README checks skipped")
    paper = root / PAPER_DIR
    for tex in sorted(paper.rglob("*.tex")) if paper.exists() else []:
        found = check_source(tex.read_text(encoding="utf-8"), "tex", values)
        _report(tex, root, found)
        failed = failed or bool(found)
    print("check failed" if failed else "check ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
