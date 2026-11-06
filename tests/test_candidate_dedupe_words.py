"""Ticket 009b §2 중복 판정 D1: 줄 맞춤 뒤 줄 안 단어 diff, 추가·삭제를 부호로 구분한 변경분."""

import itertools
from fractions import Fraction
from pathlib import Path

import pytest

from evidencegate.candidate_dedupe import (
    is_duplicate_words,
    jaccard,
    word_change_set,
)

ROOT = Path(__file__).resolve().parents[1]
S0 = (ROOT / "skills" / "S0" / "SKILL.md").read_text()
STEP4 = (
    "4. If a recent deployment exists and all dependencies are healthy, roll the deployment back."
)


def edit(old: str, new: str) -> str:
    assert old in S0
    return S0.replace(old, new)


def test_a_clause_added_to_a_step_gives_only_the_new_words() -> None:
    text = edit(STEP4, STEP4.replace("exists and", "exists, traffic is unchanged, and"))
    assert word_change_set(text, S0) == frozenset({"+traffic", "+is", "+unchanged"})


def test_replaced_words_appear_with_both_signs() -> None:
    text = edit("2. Check for recent deployments.", "2. Check traffic metrics.")
    assert word_change_set(text, S0) == frozenset(
        {"+traffic", "+metrics", "-for", "-recent", "-deployments"}
    )


def test_an_inserted_step_gives_its_words_and_a_deleted_step_gives_minus_words() -> None:
    inserted = edit(
        "3. Check dependency health.", "3. Check dependency health.\n4. Check traffic metrics."
    )
    assert word_change_set(inserted, S0) == frozenset({"+check", "+traffic", "+metrics"})
    deleted = edit("3. Check dependency health.\n", "")
    assert word_change_set(deleted, S0) == frozenset({"-check", "-dependency", "-health"})


def test_insert_and_replace_of_the_same_words_are_not_duplicates() -> None:
    """009 cand-007(단계 추가)과 cand-032(2단계 바꿈): v1 줄 토큰은 J 1.0, D1은 2/6."""
    inserted = edit(
        "3. Check dependency health.", "3. Check dependency health.\n4. Check traffic metrics."
    )
    replaced = edit("2. Check for recent deployments.", "2. Check traffic metrics.")
    j = jaccard(word_change_set(inserted, S0), word_change_set(replaced, S0))
    assert j == pytest.approx(2 / 6)
    assert not is_duplicate_words(inserted, replaced, S0, 0.55)


def test_different_clauses_on_the_same_line_are_kept_apart() -> None:
    """009의 B: v1은 같은 줄의 다른 조건을 원래 줄 단어 때문에 중복으로 본다."""
    traffic = edit(STEP4, STEP4.replace("healthy,", "healthy, and traffic is unchanged,"))
    cpu = edit(STEP4, STEP4.replace("healthy,", "healthy, and CPU usage is not elevated,"))
    assert not is_duplicate_words(traffic, cpu, S0, 0.55)


def test_normalized_equal_texts_are_duplicates_at_any_threshold() -> None:
    a = edit(STEP4, STEP4.replace("exists and", "exists, traffic is unchanged, and"))
    b = a.replace("Traffic", "traffic").replace(",", " ,")
    assert is_duplicate_words(a, b, S0, 1.0)


def test_threshold_is_inclusive_and_must_be_in_range() -> None:
    a = edit(STEP4, STEP4.replace("exists and", "exists, traffic is unchanged, and"))
    b = edit(STEP4, STEP4.replace("exists and", "exists, traffic is flat, and"))
    j = jaccard(word_change_set(a, S0), word_change_set(b, S0))
    assert j == pytest.approx(2 / 4)
    assert is_duplicate_words(a, b, S0, 0.5)
    assert not is_duplicate_words(a, b, S0, 0.51)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            is_duplicate_words(a, b, S0, bad)


def test_s0_and_the_controls_are_pairwise_distinct_above_the_d1_bound() -> None:
    """D1에서 통제 쌍의 최댓값은 11/21(약 0.524). 009b의 t는 이보다 커야 한다(티켓 §2)."""
    texts = {"S0": S0}
    texts |= {
        p.parent.name: p.read_text() for p in sorted((ROOT / "skills/controls").glob("*/SKILL.md"))
    }
    pairs = {
        (a, b): jaccard(word_change_set(texts[a], S0), word_change_set(texts[b], S0))
        for a, b in itertools.combinations(texts, 2)
    }
    worst = max(pairs, key=lambda k: pairs[k])
    assert set(worst) == {"mixed-scale-on-traffic", "pos-traffic-check"}
    assert Fraction(pairs[worst]).limit_denominator(100) == Fraction(11, 21)
    for a, b in pairs:
        assert not is_duplicate_words(texts[a], texts[b], S0, 0.55)
