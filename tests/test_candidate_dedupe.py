"""§5 중복 제거: S0 대비 변경분의 자카드 (Ticket 004 §5, 2026-10-16 정의 보완과 임계값)."""

import itertools
from pathlib import Path

import pytest

from evidencegate.candidate_dedupe import (
    DUPLICATE_THRESHOLD,
    change_set,
    is_duplicate,
    jaccard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
S0 = (REPO_ROOT / "skills" / "S0" / "SKILL.md").read_text(encoding="utf-8")
CONTROLS = {
    p.parent.name: p.read_text(encoding="utf-8")
    for p in sorted((REPO_ROOT / "skills" / "controls").glob("*/SKILL.md"))
}
S0_STEPS = [line for line in S0.splitlines() if line[:1].isdigit()]

# 2차 실측 gpt-5.4 후보(풀에 넣지 않음). 한계 기록용.
PROBE2_GPT_5_4 = S0.replace(
    "4. If a recent deployment exists and all dependencies are healthy, roll the deployment back.",
    "4. If a recent deployment exists and all dependencies are healthy, check traffic metrics "
    "to rule out increased traffic before rolling the deployment back.",
)


def skill(*steps: str) -> str:
    body = "\n".join(f"{i}. {s.split('. ', 1)[1]}" for i, s in enumerate(steps, start=1))
    return f"# Incident Response Skill\n\n{body}\n"


def add(step: str) -> str:
    return skill(*S0_STEPS[:3], f"0. {step}", *S0_STEPS[3:])


def test_threshold_is_the_preregistered_value() -> None:
    assert DUPLICATE_THRESHOLD == 0.6


def test_s0_and_controls_are_mutually_non_duplicate() -> None:
    """고정 테스트(티켓 §5): S0와 통제 후보 7개는 서로 모두 비중복이다."""
    texts = {"S0": S0, **CONTROLS}
    for a, b in itertools.combinations(texts, 2):
        assert not is_duplicate(texts[a], texts[b], S0), (a, b)


def test_largest_s0_control_pair_is_0_571() -> None:
    texts = {"S0": S0, **CONTROLS}
    sets = {name: change_set(text, S0) for name, text in texts.items()}
    best = max(
        (jaccard(sets[a], sets[b]), a, b) for a, b in itertools.combinations(sorted(sets), 2)
    )
    assert best[1:] == ("neg-timing-shortcut", "pos-timing-only")
    assert best[0] == pytest.approx(0.571, abs=5e-4)
    assert best[0] < DUPLICATE_THRESHOLD


def test_s0_has_an_empty_change_set() -> None:
    assert change_set(S0, S0) == frozenset()


def test_added_lines_use_plain_tokens_and_ignore_deletions() -> None:
    text = skill(*S0_STEPS[:2], "0. Check traffic first.", *S0_STEPS[3:])  # 3단계를 바꿈
    assert change_set(text, S0) == {"check", "traffic", "first"}


def test_deletion_only_uses_marked_removed_tokens() -> None:
    text = skill(*S0_STEPS[:2], *S0_STEPS[3:])  # "Check dependency health." 삭제
    assert change_set(text, S0) == {"-check", "-dependency", "-health"}


def test_deletion_only_is_not_s0() -> None:
    """빈 집합 모순을 고친 이유: 삭제만 한 neg-skip-metrics와 S0가 중복이면 안 된다."""
    assert not is_duplicate(CONTROLS["neg-skip-metrics"], S0, S0)


def test_opposite_edits_are_not_duplicates() -> None:
    """표시가 없으면 3/5 = 0.6으로 중복이 되는 예. 표시를 붙이면 0이다."""
    deleted = skill(*S0_STEPS[:2], *S0_STEPS[3:])
    added = add("Check dependency health before scaling.")
    assert jaccard(change_set(deleted, S0), change_set(added, S0)) == 0.0
    assert not is_duplicate(deleted, added, S0)


def test_deleting_different_steps_is_not_a_duplicate() -> None:
    a = skill(*S0_STEPS[:2], *S0_STEPS[3:])
    b = skill(*S0_STEPS[1:])
    assert not is_duplicate(a, b, S0)


def test_same_deletion_is_a_duplicate() -> None:
    a = skill(*S0_STEPS[:2], *S0_STEPS[3:])
    assert is_duplicate(a, a, S0)


def test_jaccard_empty_sets() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0


def test_boundary_jaccard_equal_to_threshold_is_a_duplicate() -> None:
    a = add("alpha beta gamma.")
    b = add("alpha beta gamma delta epsilon.")
    c = add("alpha beta gamma delta epsilon zeta.")
    assert jaccard(change_set(a, S0), change_set(b, S0)) == pytest.approx(0.6)
    assert is_duplicate(a, b, S0)
    assert jaccard(change_set(a, S0), change_set(c, S0)) == pytest.approx(0.5)
    assert not is_duplicate(a, c, S0)


def test_normalized_equal_texts_are_duplicates() -> None:
    a = add("Check traffic first.")
    b = a.replace("Check traffic first.", "check  TRAFFIC first")
    assert is_duplicate(a, b, S0)


def test_rewording_the_same_fix_is_missed() -> None:
    """한계: 표현만 다른 같은 수정을 놓친다(2차 실측 gpt-5.4 후보 vs pos-traffic-check)."""
    value = jaccard(change_set(PROBE2_GPT_5_4, S0), change_set(CONTROLS["pos-traffic-check"], S0))
    assert value == pytest.approx(0.457, abs=5e-4)
    assert not is_duplicate(PROBE2_GPT_5_4, CONTROLS["pos-traffic-check"], S0)
