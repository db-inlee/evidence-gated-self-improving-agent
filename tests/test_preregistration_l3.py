"""Ticket 006(L3) 사전 등록 해시: 티켓의 사전 등록 범위가 등록 시점 이후 바뀌지 않았는지.

원문 사본(preregistration/ticket-006-l3-sections.txt)은 git에 있으므로 항상 검사한다. 티켓 문서는
git에 올리지 않는 로컬 문서라, 없으면 티켓과의 비교만 건너뛴다.

범위(티켓 §9): §목표·연구 질문·고정 결정·§1~5("## 목표"부터 "## 6." 직전까지)와
§7("## 7."부터 "## 9." 직전까지). 구간마다 앞뒤 공백을 지우고 빈 줄 하나로 잇고, 끝에 줄바꿈
하나를 붙인 UTF-8이다.
"""

import hashlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_FILE = REPO_ROOT / "preregistration" / "ticket-006-l3.sha256"
COPY = REPO_ROOT / "preregistration" / "ticket-006-l3-sections.txt"
TICKET = REPO_ROOT / "tickets" / "ticket-006-l3-compositional-shift.md"


def fields() -> dict[str, str]:
    out = {}
    for line in HASH_FILE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and ": " in line:
            key, value = line.split(": ", 1)
            out[key] = value
    return out


def ranges() -> list[tuple[str, str]]:
    f = fields()
    return [(f["range1_start"], f["range1_end"]), (f["range2_start"], f["range2_end"])]


def extract(text: str, bounds: list[tuple[str, str]]) -> str:
    parts = []
    for start, end in bounds:
        begin = text.index(f"\n{start}\n") + 1
        stop = text.index(f"\n{end}\n", begin)
        parts.append(text[begin:stop].strip())
    return "\n\n".join(parts) + "\n"


def test_hash_file_format() -> None:
    f = fields()
    assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"])
    assert f["file"] == "tickets/ticket-006-l3-compositional-shift.md"
    assert f["copy"] == "preregistration/ticket-006-l3-sections.txt"
    assert ranges() == [
        ("## 목표", "## 6. 작업 순서 (커밋 지점 단위)"),
        ("## 7. 결과를 보기 전에 정해 둔 것", "## 9. 사전 등록 고정"),
    ]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00", f["recorded_at"])


def test_committed_copy_matches_the_registered_hash() -> None:
    """티켓이 없어도 돈다: git에 있는 원문 사본이 기록된 해시와 같다."""
    f = fields()
    data = COPY.read_bytes()
    assert hashlib.sha256(data).hexdigest() == f["sha256"]
    text = data.decode("utf-8")
    assert text.startswith("## 목표\n")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert len(text.splitlines()) == int(f["lines"])


def test_copy_holds_every_registered_section() -> None:
    text = COPY.read_text(encoding="utf-8")
    for heading in (
        "## 목표",
        "## 연구 질문 (사전 등록 대상)",
        "## 고정된 설계 결정 (변경 금지)",
        "## 1. L3 정의",
        "## 2. 생성기 불변식 (테스트로 고정)",
        "## 3. 분할과 실행 규모",
        "## 4. 분석",
        "## 5. 범위 밖",
        "## 7. 결과를 보기 전에 정해 둔 것",
    ):
        assert f"{heading}\n" in text, heading
    for excluded in ("## 6. 작업 순서", "## 9. 사전 등록 고정", "## 완료 기준", "> 개정:"):
        assert excluded not in text, excluded


@pytest.mark.skipif(not TICKET.exists(), reason="local ticket (not in git) is absent")
def test_ticket_ranges_still_match_the_copy() -> None:
    """티켓이 있으면 범위를 다시 잘라 사본과 같은지 본다(등록 뒤 무심한 수정을 막는다)."""
    assert extract(TICKET.read_text(encoding="utf-8"), ranges()) == COPY.read_text(encoding="utf-8")


def test_extract_joins_ranges_with_one_blank_line() -> None:
    text = "intro\n## A\nx\n\n## B\nskip\n## C\ny \n## D\nend\n"
    assert extract(text, [("## A", "## B"), ("## C", "## D")]) == "## A\nx\n\n## C\ny\n"
