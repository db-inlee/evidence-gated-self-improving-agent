"""인용 확인 기록 (Ticket 007 원칙 5): README가 인용하는 문헌은 모두 원문 확인 기록이 있다."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CITATIONS = REPO_ROOT / "writeup" / "inputs" / "citations.json"
TEMPLATE = REPO_ROOT / "writeup" / "README.template"
FIELDS = {"key", "id", "url", "title", "authors", "year", "venue", "checked_on", "scope"}
FIELDS |= {"evidence", "claim"}


def entries() -> list[dict[str, object]]:
    data: list[dict[str, object]] = json.loads(CITATIONS.read_text())["entries"]
    return data


def cited_ids(text: str) -> set[str]:
    arxiv = {f"arXiv:{m}" for m in re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", text)}
    arxiv |= {f"arXiv:{m}" for m in re.findall(r"arXiv:(\d{4}\.\d{4,5})", text)}
    doi = {f"doi:{m}" for m in re.findall(r"doi\.org/(10\.\d{4,9}/[^\s)\]]+)", text)}
    return arxiv | doi


def test_every_entry_records_what_was_checked() -> None:
    keys = [e["key"] for e in entries()]
    assert len(keys) == len(set(keys))
    for e in entries():
        assert set(e) == FIELDS, e["key"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(e["checked_on"])), e["key"]
        assert str(e["evidence"]).strip() and str(e["claim"]).strip(), e["key"]
        assert str(e["url"]).startswith("https://"), e["key"]


def test_every_work_the_readme_cites_has_a_checked_entry() -> None:
    cited = cited_ids(TEMPLATE.read_text())
    assert cited
    recorded = {str(e["id"]) for e in entries()}
    assert cited <= recorded, cited - recorded


def test_the_check_catches_an_unrecorded_citation() -> None:
    text = TEMPLATE.read_text() + "\nSee [arXiv:2501.00001](https://arxiv.org/abs/2501.00001).\n"
    recorded = {str(e["id"]) for e in entries()}
    assert "arXiv:2501.00001" in cited_ids(text) - recorded
