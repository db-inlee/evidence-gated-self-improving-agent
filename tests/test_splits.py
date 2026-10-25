"""데이터 분할(Ticket 004 §2): 크기, 칸 구성, 시드 범위 비중복, 결정성."""

from collections import Counter
from collections.abc import Callable

import pytest

from evidencegate.runner import Case
from evidencegate.splits import SPLITS, dev_a, dev_b, oracle
from evidencegate.suites import CALIBRATION_SEED_OFFSET
from opsworld.state import Family, Level

EXPECTED = {
    "dev_a": (dev_a, 60, 10, range(100_000, 100_010)),
    "dev_b": (dev_b, 210, 35, range(200_000, 200_035)),
    "oracle": (oracle, 300, 50, range(300_000, 300_050)),
}
# Ticket 004 split은 L1·L2만이다. Level.L3(Ticket 006)는 사후 채점 전용이라 여기에 없다.
SPLIT_LEVELS = (Level.L1, Level.L2)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_size_and_cells(name: str) -> None:
    make, size, per_cell, _ = EXPECTED[name]
    cases = make()
    assert len(cases) == size
    cells = Counter((c.family, c.level) for c in cases)
    assert set(cells) == {(f, lv) for f in Family for lv in SPLIT_LEVELS}
    assert set(cells.values()) == {per_cell}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_seed_range_is_shared_by_every_cell(name: str) -> None:
    make, _, _, seeds = EXPECTED[name]
    cases = make()
    for family in Family:
        for level in SPLIT_LEVELS:
            cell = sorted(c.incident_seed for c in cases if (c.family, c.level) == (family, level))
            assert cell == list(seeds)


def seed_range(make: Callable[[], list[Case]]) -> range:
    seeds = [c.incident_seed for c in make()]
    return range(min(seeds), max(seeds) + 1)


def test_seed_ranges_do_not_overlap_each_other_or_calibration() -> None:
    ranges = [seed_range(make) for make, *_ in EXPECTED.values()]
    calibration_floor = CALIBRATION_SEED_OFFSET
    for i, a in enumerate(ranges):
        assert a.stop <= calibration_floor  # 보정 목록(900,000번대) 아래
        for b in ranges[i + 1 :]:
            assert a.stop <= b.start or b.stop <= a.start


def test_splits_are_deterministic_and_unique() -> None:
    for make, *_ in EXPECTED.values():
        first, second = make(), make()
        assert first == second
        assert len(set(first)) == len(first)


def test_splits_registry_names_every_split() -> None:
    assert set(SPLITS) == {"dev_a", "dev_b", "oracle"}
    for name, (make, *_rest) in EXPECTED.items():
        assert SPLITS[name]() == make()


def test_split_cases_are_cases_of_both_levels() -> None:
    assert {c.level for c in dev_a()} == {Level.L1, Level.L2}
    assert all(isinstance(c, Case) for c in oracle())
