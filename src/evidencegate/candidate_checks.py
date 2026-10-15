"""proposer 후보에만 적용하는 §5 유효성 검사 세 가지 (Ticket 004 §5, 2026-10-16 사용자 결정).

첫 실측 후보를 보고 정한 규칙이다(보고서 2026-10-15-07). 통제 후보에는 적용하지 않는다.

- 줄 수: 정규화한 줄 단위 diff에서 S0 대비 추가·수정 줄 ≤ 1, 삭제 줄 ≤ 2.
- 구조: 비어 있지 않은 줄이 제목과 1부터 연속인 번호 단계뿐.
- 용어: 채점 용어, 실패 라벨 이름, 술어 이름이 없음.

"한 단계에 한 아이디어"는 기계로 다 검사하지 못한다(한 줄에 조건 여러 개를 넣을 수 있음).
"""

import difflib
import re

from opsworld.grader import FailureLabel
from opsworld.predicates import Predicate

SKILL_TITLE = "# Incident Response Skill"
MAX_ADDED_LINES = 1  # 추가·수정 줄 상한
MAX_REMOVED_LINES = 2  # 삭제 줄 상한
GRADING_TERMS = (
    "evidence card",
    "hypothesis",
    "expected effect",
    "failure type",
    "evidence_safe",
    "evidence-safe",
    "correct disposition",
    "predicate",
    *(label.value.lower() for label in FailureLabel),
    *(predicate.value for predicate in Predicate),
)
_STEP = re.compile(r"^(\d+)\. \S")


def normalize_lines(text: str) -> list[str]:
    """소문자, 단계 번호 제거, 구두점·공백 정리, 빈 줄 제거 (티켓 §5 중복 제거와 같은 정규화)."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\d+[.)]\s*", "", raw.strip().lower())
        line = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", line)).strip()
        if line:
            lines.append(line)
    return lines


def changed_lines(text: str, s0_text: str) -> tuple[int, int]:
    """S0 대비 (추가·수정 줄 수, 삭제 줄 수). 수정된 줄은 양쪽에 하나씩 센다."""
    base, candidate = normalize_lines(s0_text), normalize_lines(text)
    added = removed = 0
    matcher = difflib.SequenceMatcher(None, base, candidate, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            added += j2 - j1
        if tag in ("replace", "delete"):
            removed += i2 - i1
    return added, removed


def check_proposer_candidate(text: str, s0_text: str) -> tuple[str, ...]:
    """걸린 검사 이름(`line_count`, `structure`, `terms`)을 이 순서로 돌려준다. 통과하면 빈 튜플."""
    reasons = []
    added, removed = changed_lines(text, s0_text)
    if added > MAX_ADDED_LINES or removed > MAX_REMOVED_LINES:
        reasons.append("line_count")
    if not _structure_ok(text):
        reasons.append("structure")
    lowered = text.lower()
    if any(term in lowered for term in GRADING_TERMS):
        reasons.append("terms")
    return tuple(reasons)


def _structure_ok(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != SKILL_TITLE:
        return False
    numbers = []
    for line in lines[1:]:
        match = _STEP.match(line)
        if match is None:
            return False
        numbers.append(int(match.group(1)))
    return numbers == list(range(1, len(numbers) + 1))
