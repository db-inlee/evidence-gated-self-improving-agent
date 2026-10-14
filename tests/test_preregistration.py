"""사전 등록 해시(Ticket 004 §8): 설계 문서 7.8절이 등록 시점 이후 바뀌지 않았는지.

원문 사본(preregistration/ticket-004-section-7.8.txt)은 git에 있으므로 항상 검사한다. 설계
문서(docs/design.md)는 git에 올리지 않는 로컬 문서라, 없으면 설계 문서와의 비교만 건너뛴다.
"""

import hashlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_FILE = REPO_ROOT / "preregistration" / "ticket-004.sha256"
DESIGN = REPO_ROOT / "docs" / "design.md"
COPY = REPO_ROOT / "preregistration" / "ticket-004-section-7.8.txt"


def fields() -> dict[str, str]:
    out = {}
    for line in HASH_FILE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and ": " in line:
            key, value = line.split(": ", 1)
            out[key] = value
    return out


def test_hash_file_format() -> None:
    f = fields()
    assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"])
    assert f["file"] == "docs/design.md"
    assert f["start_marker"].startswith("### 7.8 ")


def test_committed_copy_matches_the_registered_hash() -> None:
    """설계 문서가 없어도 돈다: git에 있는 원문 사본이 기록된 해시와 같다."""
    f = fields()
    assert f["copy"] == "preregistration/ticket-004-section-7.8.txt"
    assert hashlib.sha256(COPY.read_bytes()).hexdigest() == f["sha256"]
    assert COPY.read_text(encoding="utf-8").startswith(f["start_marker"])
    assert COPY.read_text(encoding="utf-8").count("\n") == int(f["lines"])


@pytest.mark.skipif(not DESIGN.exists(), reason="docs/design.md is a local, untracked document")
def test_section_7_8_matches_the_registered_hash() -> None:
    f = fields()
    text = DESIGN.read_text(encoding="utf-8")
    end = f["end_marker"].replace("\\n", "\n")
    start = text.index(f["start_marker"])
    section = text[start : text.index(end, start)].strip() + "\n"
    assert hashlib.sha256(section.encode("utf-8")).hexdigest() == f["sha256"]
    assert section == COPY.read_text(encoding="utf-8")
