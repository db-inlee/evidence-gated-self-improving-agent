"""Ticket 009b §2: 새 중복 판정의 성질(묶음 안 흡수율, 묶음 분리율, 순차 풀)과 t 선택 규칙."""

import json
from pathlib import Path

import pytest

from evidencegate.t009b import dedupe_properties as dp

ROOT = Path(__file__).resolve().parents[2]


def test_grid_and_bound_follow_the_ticket() -> None:
    assert dp.T_GRID == (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    assert dp.MIN_SEPARATION == 0.95


def test_pair_rates_count_within_and_between_group_pairs() -> None:
    sets = [frozenset("ab"), frozenset("ab"), frozenset("ac"), frozenset("xy")]
    groups = ["G", "G", "G", "H"]
    # 같은 묶음 쌍: (0,1) J 1, (0,2) 1/3, (1,2) 1/3. 다른 묶음 쌍: 셋 다 J 0
    rates = dp.pair_rates(sets, groups, 0.5)
    assert rates == {"absorption": 1 / 3, "separation": 1.0, "within_pairs": 3, "between_pairs": 3}
    assert dp.pair_rates(sets, groups, 0.3)["absorption"] == 1.0


def test_sequential_pool_keeps_the_first_of_each_duplicate_run() -> None:
    items = ["a", "a2", "b", "a3", "c"]
    same = {("a2", "a"), ("a3", "a")}
    pool = dp.sequential_pool(items, lambda x, y: (x, y) in same)
    assert pool == [0, 2, 4]


def test_choose_t_takes_the_largest_absorption_among_separating_values() -> None:
    rows = [
        {"t": 0.55, "separation": 0.90, "absorption": 0.90},
        {"t": 0.60, "separation": 0.96, "absorption": 0.70},
        {"t": 0.65, "separation": 0.97, "absorption": 0.70},
        {"t": 0.70, "separation": 0.99, "absorption": 0.50},
    ]
    assert dp.choose_t(rows) == (0.60, "rule")  # 같은 흡수율이면 작은 t


def test_choose_t_falls_back_to_the_largest_separation() -> None:
    rows = [
        {"t": 0.55, "separation": 0.80, "absorption": 0.9},
        {"t": 0.60, "separation": 0.90, "absorption": 0.8},
        {"t": 0.65, "separation": 0.90, "absorption": 0.7},
    ]
    assert dp.choose_t(rows) == (0.65, "fallback")  # 같은 분리율이면 큰 t


def test_function_word_variant_drops_the_listed_words() -> None:
    s = frozenset({"+and", "+cpu", "-the", "+not"})
    assert dp.without_function_words(s) == frozenset({"+cpu", "+not"})
    assert "not" not in dp.FUNCTION_WORDS


def test_group_file_is_read_in_call_order() -> None:
    labels = dp.call_groups(ROOT / "plans" / "ticket-009b-dedupe-groups.json")
    assert len(labels) == 80
    assert labels[0] == "G4" and labels[5] == "G1" and labels[32] == "O32"


def test_texts_must_match_the_recorded_call_hashes(tmp_path: Path) -> None:
    groups = json.loads((ROOT / "plans" / "ticket-009b-dedupe-groups.json").read_text())
    pool = tmp_path / "pool"
    pool.mkdir()
    for i in range(80):
        (pool / f"call-{i:03d}.json").write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        dp.load_texts(pool, groups["call_sha256"])


def test_properties_table_has_every_definition_and_t(tmp_path: Path) -> None:
    s0 = (ROOT / "skills/S0/SKILL.md").read_text()
    step = "4. If a recent deployment exists and all dependencies are healthy, "
    step += "roll the deployment back."
    texts = [
        s0.replace(step, step.replace("healthy,", "healthy, and traffic is unchanged,")),
        s0.replace(step, step.replace("healthy,", "healthy, and traffic is flat,")),
        s0.replace(step, step.replace("healthy,", "healthy, and CPU usage is not elevated,")),
    ]
    table = dp.properties(texts, ["G", "G", "H"], s0)
    assert [r["definition"] for r in table].count("D1") == len(dp.T_GRID)
    assert {r["definition"] for r in table} == {"D1", "D2", "v1"}
    d1 = {r["t"]: r for r in table if r["definition"] == "D1"}
    # D1: unchanged 대 flat은 J 3/5 = 0.6, CPU 조건과는 2/8
    assert d1[0.55]["pool"] == 2 and d1[0.55]["groups_in_pool"] == 2
    assert d1[0.55]["pool_by_group"] == {"G": 1, "H": 1}
    assert d1[0.65]["pool"] == 3 and d1[0.65]["pool_by_group"] == {"G": 2, "H": 1}
    v1 = {r["t"]: r for r in table if r["definition"] == "v1"}
    assert v1[0.55]["pool"] == 1  # v1 줄 토큰: 같은 줄이라 모두 중복


def test_boundaries_are_inclusive_where_the_ticket_says_so() -> None:
    sets = [frozenset("ab"), frozenset("ac")]  # J = 1/3
    assert dp.pair_rates(sets, ["G", "G"], 1 / 3)["absorption"] == 1.0  # J ≥ t는 흡수
    assert dp.pair_rates(sets, ["G", "H"], 1 / 3)["separation"] == 0.0  # J < t만 분리
    rows = [
        {"t": 0.55, "separation": 0.95, "absorption": 0.4},
        {"t": 0.60, "separation": 0.99, "absorption": 0.3},
    ]
    assert dp.choose_t(rows) == (0.55, "rule")  # 분리율 0.95와 같으면 통과
