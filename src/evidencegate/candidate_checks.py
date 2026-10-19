"""proposer 후보 유효성 검사 (Ticket 004 §5). 통제 후보에는 적용하지 않는다.

`validate_proposer_candidate`가 §5 검사 전체를 돌린다. 하나라도 걸리면 무효다.

- 변경 없음: S0와 정규화 일치.
- 영어(ASCII), S0와 같은 제목, 길이 상한(S0의 2배, 문자 수).
- Ticket 003 정규식: 단계 번호 밖의 숫자·`%`, family·하위 유형 이름(밑줄·공백·하이픈 변형).
- 서비스·의존성 ID, 배포 ID(`deploy-…`), 버전 문자열(`v1.2.3`).
- v0 밖 단계 금지어(단어 단위, 사전 고정 목록). 완전하지 않다(v0 밖 지시는 실행될 수 없어 행동
  변화로 평가에 잡힌다).
- 2026-10-16 추가, 첫 실측 후보를 보고 정함(보고서 2026-10-15-07):
  - 줄 수: 정규화한 줄 단위 diff에서 S0 대비 추가·수정 줄 ≤ 1, 삭제 줄 ≤ 2.
  - 구조: 비어 있지 않은 줄이 제목과 1부터 연속인 번호 단계뿐.
  - 용어: 채점 용어, 실패 라벨 이름, 술어 이름이 없음.
  "한 단계에 한 아이디어"는 기계로 다 검사하지 못한다(한 줄에 조건 여러 개를 넣을 수 있음).
- 증거 카드 스키마.
"""

import difflib
import re
from collections.abc import Collection, Mapping
from typing import Any

from opsworld import constants
from opsworld.generator import L2_SUBTYPES
from opsworld.grader import FailureLabel
from opsworld.predicates import Predicate
from opsworld.state import Family

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

MAX_LENGTH_RATIO = 2  # 길이 상한: S0 문자 수의 2배
# v0 밖 단계 금지어(티켓 §5, 사전 고정). 단어 단위, 대소문자 무관.
FORBIDDEN_STEP_WORDS = (
    "log",
    "logs",
    "restart",
    "failover",
    "escalate",
    "page",
    "notify",
    "ticket",
    "wait",
    "database",
    "queue",
    "cache",
    "feature flag",
    "roll forward",
    "redeploy",
    "hotfix",
    "reboot",
    "drain",
)
# 증거 카드(티켓 §4): 실패 유형은 내보낸 실패 라벨 중 proposer가 고를 수 있는 것
CARD_FAILURE_TYPES = ("PREMATURE_ACTION", "WRONG_INTERVENTION")
CARD_EFFECTS = ("up", "down", "none")
CARD_SOURCE_COUNT = 6  # 호출마다 보여 주는 사례 수
_CARD_FIELDS = {"candidate_id", "failure_type", "source_cases", "hypothesis", "expected_effect"}
_CANDIDATE_ID = re.compile(r"^cand-\d{3}$")

_STEP_PREFIX = re.compile(r"^\d+\. ", re.MULTILINE)
_HIDDEN_NAMES = sorted(
    {f.value for f in Family} | {s for subtypes in L2_SUBTYPES.values() for s in subtypes}
)
_HIDDEN_NAME_PATTERN = re.compile(
    "|".join(name.replace("_", r"[ _-]") for name in _HIDDEN_NAMES), re.IGNORECASE
)
_ID_PATTERN = re.compile(
    "|".join(
        [re.escape(i) for i in (*constants.SERVICE_IDS, *constants.DEPENDENCY_IDS)]
        + [r"\bdeploy-[0-9a-f]+", r"\bv\d+(?:\.\d+)+"]
    ),
    re.IGNORECASE,
)
_FORBIDDEN_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in FORBIDDEN_STEP_WORDS) + r")\b", re.IGNORECASE
)


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


def validate_proposer_candidate(
    text: str,
    evidence: Mapping[str, Any],
    s0_text: str,
    export_ids: Collection[str],
) -> tuple[str, ...]:
    """§5 검사 전체. 걸린 검사 이름을 고정된 순서로 돌려준다. 통과하면 빈 튜플."""
    if normalize_lines(text) == normalize_lines(s0_text):
        return ("no_change",)
    reasons = []
    if not text.isascii():
        reasons.append("non_ascii")
    if not text.startswith(SKILL_TITLE + "\n"):
        reasons.append("title")
    if len(text) > MAX_LENGTH_RATIO * len(s0_text):
        reasons.append("length")
    body = _STEP_PREFIX.sub("", text)
    if re.search(r"\d", body) or "%" in body:
        reasons.append("numbers")
    if _HIDDEN_NAME_PATTERN.search(text):
        reasons.append("family_names")
    if _ID_PATTERN.search(text):
        reasons.append("ids")
    if _FORBIDDEN_WORD_PATTERN.search(text):
        reasons.append("forbidden_words")
    reasons += check_proposer_candidate(text, s0_text)
    if not _evidence_card_ok(evidence, export_ids):
        reasons.append("evidence_card")
    return tuple(reasons)


def _evidence_card_ok(evidence: Mapping[str, Any], export_ids: Collection[str]) -> bool:
    if set(evidence) != _CARD_FIELDS:
        return False
    sources = evidence["source_cases"]
    effect = evidence["expected_effect"]
    hypothesis = evidence["hypothesis"]
    return (
        isinstance(evidence["candidate_id"], str)
        and _CANDIDATE_ID.match(evidence["candidate_id"]) is not None
        and evidence["failure_type"] in CARD_FAILURE_TYPES
        and isinstance(sources, list)
        and len(sources) == CARD_SOURCE_COUNT
        and len(set(sources)) == len(sources)
        and all(s in export_ids for s in sources)
        and isinstance(hypothesis, str)
        and bool(hypothesis.strip())
        and isinstance(effect, dict)
        and set(effect) == {"cd", "es"}
        and all(effect[k] in CARD_EFFECTS for k in ("cd", "es"))
    )
