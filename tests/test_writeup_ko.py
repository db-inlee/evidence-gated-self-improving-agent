"""한국어 요약판 검사(Ticket 011 §3).

한국어 금지 표현은 부분 문자열로 찾고, 영어 목록도 함께 적용한다.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evidencegate import writeup_render as wr

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"
TICKET_KO = (
    "진짜 효과",
    "참 효과",
    "실제 효과",
    "증명",
    "입증",
    "보장",
    "확증했다",
    "최초",
    "처음으로",
    "유의미",
    "통계적으로 유의",
    "유의하게",
    "강건",
    "견고",
    "안전",
    "항상",
    "결코",
    "절대로",
    "반드시",
    "획기적",
    "압도적",
    "명백히",
    "확실히",
)


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(VALUES.read_text(encoding="utf-8"))
    return data


def rules(found: list[wr.Violation]) -> set[str]:
    return {v.rule for v in found}


def test_the_korean_banned_list_is_the_ticket_list() -> None:
    assert wr.BANNED_KO == TICKET_KO
    assert wr.ALLOWED_TERMS_KO == ("Evidence-Safe", "확증 연구")


@pytest.mark.parametrize("phrase", TICKET_KO)
def test_each_korean_phrase_fails_even_with_a_particle(values: dict[str, Any], phrase: str) -> None:
    source = f"## 설계\n\n이 결과는 {phrase}이라고 본다.\n"
    found = wr.check_source(source, "md", values, lang="ko")
    assert rules(found) == {"banned"}, phrase
    assert wr.check_rendered(source, lang="ko"), phrase


def test_korean_phrases_are_not_checked_in_english_documents(values: dict[str, Any]) -> None:
    assert wr.check_source("## Design\n\n이 결과는 증명이다.\n", "md", values) == []


@pytest.mark.parametrize("phrase", ["proves", "significant", "first", "true effect"])
def test_english_phrases_also_fail_in_korean_documents(values: dict[str, Any], phrase: str) -> None:
    found = wr.check_source(f"## 설계\n\n이 수정은 {phrase} 결과다.\n", "md", values, lang="ko")
    assert rules(found) == {"banned"}, phrase


def test_allowed_terms_and_quotes_pass_in_korean(values: dict[str, Any]) -> None:
    ok = "## 설계\n\n확증 연구의 주 지표는 CD이고 보조 지표는 Evidence-Safe다.\n"
    assert wr.check_source(ok, "md", values, lang="ko") == []
    quoted = '## 관련 연구\n\n그 논문은 "최초의 증명"이라고 썼다.\n'
    assert wr.check_source(quoted, "md", values, lang="ko") == []
    assert rules(wr.check_source("## 설계\n\n확증했다.\n", "md", values, lang="ko")) == {"banned"}


def test_korean_exploratory_headings_use_the_english_word(values: dict[str, Any]) -> None:
    md = "## 하네스 탐침 (Exploratory)\n\n{{table:t7}}\n"
    assert wr.check_source(md, "md", values, lang="ko") == []
    bad = "## 하네스 탐침\n\n{{table:t7}}\n"
    assert rules(wr.check_source(bad, "md", values, lang="ko")) == {"placement"}


def test_a_hangul_particle_right_after_a_name_counts_as_a_number(values: dict[str, Any]) -> None:
    """`\\b`는 한글을 낱말 글자로 본다. 이름 바로 뒤에 조사를 붙이지 않고 띄어 쓴다."""
    assert rules(wr.check_source("## 설계\n\nC5가 걸렀다.\n", "md", values, lang="ko")) == {
        "number"
    }
    assert wr.check_source("## 설계\n\nC5 게이트가 걸렀다.\n", "md", values, lang="ko") == []


def test_documents_end_with_the_korean_summary() -> None:
    assert wr.DOCUMENTS[-1] == wr.Document(
        Path("writeup/README.ko.template"), Path("README.ko.md"), "ko"
    )
    assert [d.lang for d in wr.DOCUMENTS] == ["en", "en", "en", "ko"]


def test_main_check_applies_the_korean_list_only_to_the_korean_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    (root / "writeup" / "generated").mkdir(parents=True)
    for name in ("values.json", "values.json.sha256", "tables.json"):
        (root / "writeup" / "generated" / name).write_bytes(
            (REPO_ROOT / "writeup" / "generated" / name).read_bytes()
        )
    text = "# 제목\n\n## 결과\n\n이 결과는 증명이 아니다.\n"
    (root / "writeup" / "en.template").write_text(text, encoding="utf-8")
    (root / "writeup" / "ko.template").write_text(text, encoding="utf-8")
    docs = (
        wr.Document(Path("writeup/en.template"), Path("EN.md"), "en"),
        wr.Document(Path("writeup/ko.template"), Path("KO.md"), "ko"),
    )
    monkeypatch.setattr(wr, "DOCUMENTS", docs[:1])
    assert wr.main(["render", "--root", str(root)]) == 0
    assert wr.main(["check", "--root", str(root)]) == 0
    monkeypatch.setattr(wr, "DOCUMENTS", docs)
    assert wr.main(["render", "--root", str(root)]) == 0
    capsys.readouterr()
    assert wr.main(["check", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "writeup/ko.template:5: banned: 증명" in err  # 템플릿 검사도 한국어 목록을 쓴다
    assert "KO.md:5: banned: 증명" in err  # 렌더 결과 검사도
    assert "en.template" not in err and "EN.md" not in err


def test_both_readmes_link_to_each_other() -> None:
    en = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    ko = (REPO_ROOT / "README.ko.md").read_text(encoding="utf-8").splitlines()
    assert "English · [한국어](README.ko.md)" in en[:5]
    assert "[English](README.md) · 한국어" in ko[:5]


def test_the_korean_summary_is_about_a_third_of_the_english_readme() -> None:
    en = len((REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines())
    ko = len((REPO_ROOT / "README.ko.md").read_text(encoding="utf-8").splitlines())
    assert en / 4 <= ko <= en / 2, (ko, en)


def test_the_korean_summary_keeps_probe_numbers_in_the_exploratory_section() -> None:
    template = (REPO_ROOT / "writeup" / "README.ko.template").read_text(encoding="utf-8")
    section = ""
    for line in template.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading and line.startswith("## "):
            section = heading.group(1)
        if "{{t010." in line or "{{table:t7}}" in line:
            assert "(Exploratory)" in section, line
