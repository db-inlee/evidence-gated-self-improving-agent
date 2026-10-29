"""렌더러와 원고 검사 (Ticket 007 §1 검사 1~4, §6 순서 4).

일부러 넣은 위반마다 검사가 실패하는지 본다. 규칙:
- 숫자: 원고 원본에서 자리표시자({{키}}, \\R{키}) 밖의 숫자는 허용 목록(이름·식별자)만.
- 키: 자리표시자의 키가 values.json에 있다.
- 위치: kind = exploratory 값은 Exploratory 또는 Appendix 절 안에서만.
- 금지 표현: 단어 경계, 대소문자 무시. 용어 허용 목록은 Evidence-Safe 하나, 인용 부호 안
  원문 인용은 뺀다.
- README.md는 템플릿 렌더 결과와 바이트까지 같다.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from evidencegate.writeup_render import (
    BANNED,
    Violation,
    check_rendered,
    check_source,
    main,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"
TABLES = REPO_ROOT / "writeup" / "generated" / "tables.json"


@pytest.fixture(scope="module")
def values() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(VALUES.read_text())
    return data


def rules(violations: list[Violation]) -> set[str]:
    return {v.rule for v in violations}


CLEAN_MD = """# Title

## Pre-registered results

RQ1 was {{t005.rq.RQ1.verdict}} for C2 and C4 (Ticket 005, `cand-015`, commit aa1c809).
Evidence-Safe is the second metric; RQ-L3-0 judges S0 on L3.

{{table:t1}}

## Exploratory findings

E1 was closer for {{e1.summary.proposer.cd.closer_count}}
of {{e1.summary.proposer.cd.n}} candidates.

```bash
bash scripts/run_l3_analysis.sh 48afcad   # code is not checked for numbers
```
"""

CLEAN_TEX = r"""% comment 0.123 is ignored
\section{Pre-registered Results}
RQ1 was \R{t005.rq.RQ1.verdict} (Ticket 005, arXiv 2610.14224, \cite{dwork2015}).
\input{generated/t1.tex}
\includegraphics[width=0.9\linewidth]{generated/f3.pdf}
\section{Exploratory Findings}
E1: \R{e1.summary.proposer.cd.closer_count} of \R{e1.summary.proposer.cd.n}.
"""


def test_clean_sources_pass(values: dict[str, Any]) -> None:
    assert check_source(CLEAN_MD, "md", values) == []
    assert check_source(CLEAN_TEX, "tex", values) == []


# ---------------------------------------------------------------------------
# 일부러 넣은 위반: 숫자
# ---------------------------------------------------------------------------

NUMBER_VIOLATIONS = {
    "decimal": "The L3 estimate was 0.210 for one candidate.",
    "negative decimal": "It dropped to -0.21 on L3.",
    "fraction": "C4 accepted 3/5 candidates.",
    "percent": "About 12% of episodes.",
    "design constant": "Dev-B has N = 210 incidents.",
    "bootstrap count that looks like a year": "We used 2000 bootstrap draws.",
    "spelled count with digits": "There were 12 proposer candidates.",
}


@pytest.mark.parametrize("name", sorted(NUMBER_VIOLATIONS))
@pytest.mark.parametrize("fmt", ["md", "tex"])
def test_a_bare_number_fails(values: dict[str, Any], name: str, fmt: str) -> None:
    text = NUMBER_VIOLATIONS[name]
    if fmt == "md":
        source = f"## Pre-registered results\n\n{text}\n"
    else:
        source = f"\\section{{Pre-registered Results}}\n{text.replace('%', '\\%')}\n"
    found = check_source(source, fmt, values)
    assert "number" in rules(found), (fmt, name)
    assert found[0].line == 3 if fmt == "md" else found[0].line == 2


ALLOWED = (
    "Gates C1 to C4, levels L1, L2 and L3, RQ1 to RQ3 and RQ-L3-0 to RQ-L3-3.",
    "The initial skill S0, tables T1 to T3 and figures F1 to F4.",
    "Ticket 005 and Ticket 006, candidate cand-015, commit 48afcad.",
    "arXiv 2610.14224, DOI 10.1126/science.aaa9375, published 2015.",
    "See Section 3 and §8-14.",
)


@pytest.mark.parametrize("text", ALLOWED)
def test_names_and_identifiers_are_allowed(values: dict[str, Any], text: str) -> None:
    assert check_source(f"## Design\n\n{text}\n", "md", values) == []


def test_numbered_lists_and_headings_are_allowed(values: dict[str, Any]) -> None:
    source = "## 3. Design\n\n1. First item\n2) Second item\n"
    assert rules(check_source(source, "md", values)) == {"banned"}  # "First"만 걸림


# ---------------------------------------------------------------------------
# 일부러 넣은 위반: 키와 위치
# ---------------------------------------------------------------------------


def test_an_unknown_key_fails(values: dict[str, Any]) -> None:
    assert rules(check_source("{{t005.no.such.key}}\n", "md", values)) == {"key"}
    assert rules(check_source("\\R{t005.no.such.key}\n", "tex", values)) == {"key"}
    assert rules(check_source("{{table:t9}}\n", "md", values)) == {"key"}


def test_an_exploratory_value_outside_the_exploratory_section_fails(
    values: dict[str, Any],
) -> None:
    md = "## Pre-registered results\n\nE1 closer: {{e1.summary.proposer.cd.closer_count}}\n"
    assert rules(check_source(md, "md", values)) == {"placement"}
    tex = "\\section{Pre-registered Results}\n\\R{e1.cand-015.cd.e1}\n"
    assert rules(check_source(tex, "tex", values)) == {"placement"}


def test_exploratory_values_are_allowed_in_the_appendix(values: dict[str, Any]) -> None:
    md = "## Appendix: E1 for controls\n\n{{e1.neg-rollback-any.cd.e1}}\n"
    assert check_source(md, "md", values) == []
    tex = "\\appendix\n\\section{E1 for controls}\n\\R{e1.neg-rollback-any.cd.e1}\n"
    assert check_source(tex, "tex", values) == []


def test_a_subsection_inside_exploratory_still_counts(values: dict[str, Any]) -> None:
    md = (
        "## Exploratory findings\n\n### (b) Cell share\n\n{{e1.cand-015.cd.e1}}\n\n## Limitations\n"
    )
    assert check_source(md, "md", values) == []
    md_after = md + "\n{{e1.cand-015.cd.e1}}\n"
    assert rules(check_source(md_after, "md", values)) == {"placement"}


# ---------------------------------------------------------------------------
# 일부러 넣은 위반: 금지 표현
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "true effect",
        "proves",
        "guarantees",
        "demonstrates that",
        "first",
        "first to",
        "significant",
        "statistically",
        "robust",
        "safe",
        "always",
        "never",
    ],
)
def test_each_banned_phrase_fails(values: dict[str, Any], phrase: str) -> None:
    source = f"## Design\n\nThis {phrase.upper()} something.\n"
    assert rules(check_source(source, "md", values)) == {"banned"}
    assert rules(check_rendered(f"x {phrase} y\n")) == {"banned"}


def test_the_banned_list_matches_the_ticket() -> None:
    assert set(BANNED) == {
        "true effect",
        "proves",
        "guarantees",
        "demonstrates that",
        "first",
        "first to",
        "significant",
        "statistically",
        "robust",
        "safe",
        "always",
        "never",
    }


def test_word_boundaries_and_the_one_allowed_term(values: dict[str, Any]) -> None:
    text = (
        "## Design\n\nEvidence-Safe and safely, firstly-not, unsafe, alwaysx, the significance.\n"
    )
    found = check_source(text, "md", values)
    # safely, unsafe, alwaysx, significance는 단어가 다르다. firstly-not도 first가 아니다.
    assert found == []
    assert rules(check_source("## D\n\nThe Evidence-Safe gate is safe.\n", "md", values)) == {
        "banned"
    }


def test_quoted_original_text_is_exempt(values: dict[str, Any]) -> None:
    assert check_source("## Related work\n\nThey write “the first agent”.\n", "md", values) == []
    assert (
        check_source("\\section{Related Work}\nThey write ``robust'' here.\n", "tex", values) == []
    )


def test_a_banned_word_in_a_figure_caption_fails(values: dict[str, Any]) -> None:
    md = "## Design\n\n![F3](writeup/generated/f3.svg)\n\n*F3. Controls are always shown.*\n"
    assert rules(check_source(md, "md", values)) == {"banned"}
    tex = "\\begin{figure}\\caption{Controls are never hidden.}\\end{figure}\n"
    assert rules(check_source(tex, "tex", values)) == {"banned"}


# ---------------------------------------------------------------------------
# 렌더와 README 일치
# ---------------------------------------------------------------------------


def test_render_substitutes_values_and_tables(values: dict[str, Any]) -> None:
    tables = json.loads(TABLES.read_text())
    out = render_markdown(CLEAN_MD, values, tables)
    assert "RQ1 was not supported for C2 and C4" in out
    assert "| Question | Scope | CD | ES |" in out
    assert "E1 was closer for 10\nof 12 candidates." in out
    assert "{{" not in out


def test_render_refuses_an_unknown_key(values: dict[str, Any]) -> None:
    with pytest.raises(KeyError):
        render_markdown("{{nope}}", values, {})


def write_repo(tmp_path: Path, template: str, readme: str | None = None) -> Path:
    root = tmp_path / "repo"
    (root / "writeup" / "generated").mkdir(parents=True)
    for name in ("values.json", "values.json.sha256", "tables.json"):
        (root / "writeup" / "generated" / name).write_bytes(
            (REPO_ROOT / "writeup" / "generated" / name).read_bytes()
        )
    (root / "writeup" / "README.template").write_text(template)
    if readme is not None:
        (root / "README.md").write_text(readme)
    return root


def test_main_render_then_check_passes(tmp_path: Path) -> None:
    root = write_repo(tmp_path, CLEAN_MD)
    assert main(["render", "--root", str(root)]) == 0
    assert main(["check", "--root", str(root)]) == 0


def test_main_check_fails_on_a_hand_edited_readme(tmp_path: Path) -> None:
    root = write_repo(tmp_path, CLEAN_MD)
    assert main(["render", "--root", str(root)]) == 0
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("not supported", "supported"))
    assert main(["check", "--root", str(root)]) == 1


def test_main_check_fails_on_a_template_violation(tmp_path: Path) -> None:
    root = write_repo(tmp_path, CLEAN_MD + "\nIt was 0.5.\n")
    assert main(["render", "--root", str(root)]) == 0
    assert main(["check", "--root", str(root)]) == 1


def test_main_check_also_reads_paper_sources(tmp_path: Path) -> None:
    root = write_repo(tmp_path, CLEAN_MD)
    assert main(["render", "--root", str(root)]) == 0
    (root / "paper").mkdir()
    (root / "paper" / "main.tex").write_text(CLEAN_TEX)
    assert main(["check", "--root", str(root)]) == 0
    (root / "paper" / "intro.tex").write_text("\\section{Introduction}\nThis proves it.\n")
    assert main(["check", "--root", str(root)]) == 1


def test_inline_code_and_fenced_blocks_are_not_checked_for_numbers(values: dict[str, Any]) -> None:
    """명령과 경로(숫자 포함)는 코드로 쓴다. 코드는 숫자 검사를 하지 않는다."""
    md = "## Reproduce\n\nRun `bash scripts/l3_benchmark.sh run 6` to stop after a few blocks.\n"
    assert check_source(md, "md", values) == []
    assert rules(check_source(md.replace("`", ""), "md", values)) == {"number"}
