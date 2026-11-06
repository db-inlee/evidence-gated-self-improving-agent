"""후보 중복 제거 (Ticket 004 §5).

변경분: S0 대비 정규화한 줄 단위 diff에서 추가·변경된 줄의 토큰 집합. 추가·변경 줄이 없는
후보(삭제만 한 후보)는 지운 줄의 토큰에 `-`를 붙인 집합을 쓴다(2026-10-16 정의 보완). 원래
정의로는 삭제만 한 후보와 S0가 둘 다 빈 집합이라 자카드 1이 되어 "S0와 통제 후보 7개가 서로
모두 비중복"을 만족할 수 없었다. `-` 표시는 삭제 토큰이 다른 후보의 추가 토큰과 겹치지 않게 한다.

중복: 정규화한 전문이 같거나, 변경분의 자카드 ≥ 0.6. 0.6은 S0·통제 후보 쌍의 최댓값
0.571보다 위로 사전 고정한 값이다. 표현만 다른 같은 수정은 놓친다(한계).

Ticket 009b 정의 D1(`word_change_set`, `is_duplicate_words`, v1 함수는 그대로): 줄 맞춤은 v1과 같고,
같지 않은 구간마다 양쪽 줄을 단어 열로 이어 단어 단위 diff를 한다. 변경분은 추가·교체된 단어(`+`)와
삭제·교체된 단어(`-`)를 항상 함께 담는다. v1은 바뀐 줄의 단어 전체를 담아서, 같은 줄을 서로 다르게
고친 수정이 원래 줄 단어를 공유해 중복으로 잡혔다(Ticket 009 진단, 보고서 2026-11-05-06). 임계값은
호출하는 쪽이 넘긴다(009b 사전 등록에서 정함).
"""

import difflib

from evidencegate.candidate_checks import normalize_lines

DUPLICATE_THRESHOLD = 0.6
REMOVED_MARK = "-"
ADDED_MARK = "+"


def change_set(text: str, s0_text: str) -> frozenset[str]:
    base, candidate = normalize_lines(s0_text), normalize_lines(text)
    added: set[str] = set()
    removed: set[str] = set()
    matcher = difflib.SequenceMatcher(None, base, candidate, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            added.update(token for line in candidate[j1:j2] for token in line.split())
        if tag in ("replace", "delete"):
            removed.update(REMOVED_MARK + token for line in base[i1:i2] for token in line.split())
    return frozenset(added) if added else frozenset(removed)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_duplicate(a: str, b: str, s0_text: str) -> bool:
    if normalize_lines(a) == normalize_lines(b):
        return True
    return jaccard(change_set(a, s0_text), change_set(b, s0_text)) >= DUPLICATE_THRESHOLD


def word_change_set(text: str, s0_text: str) -> frozenset[str]:
    base, candidate = normalize_lines(s0_text), normalize_lines(text)
    changed: set[str] = set()
    lines = difflib.SequenceMatcher(None, base, candidate, autojunk=False)
    for tag, i1, i2, j1, j2 in lines.get_opcodes():
        if tag == "equal":
            continue
        old = " ".join(base[i1:i2]).split()
        new = " ".join(candidate[j1:j2]).split()
        words = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for wtag, a1, a2, b1, b2 in words.get_opcodes():
            if wtag in ("replace", "insert"):
                changed.update(ADDED_MARK + w for w in new[b1:b2])
            if wtag in ("replace", "delete"):
                changed.update(REMOVED_MARK + w for w in old[a1:a2])
    return frozenset(changed)


def is_duplicate_words(a: str, b: str, s0_text: str, threshold: float) -> bool:
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if normalize_lines(a) == normalize_lines(b):
        return True
    return jaccard(word_change_set(a, s0_text), word_change_set(b, s0_text)) >= threshold
