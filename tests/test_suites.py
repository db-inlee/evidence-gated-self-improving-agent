"""보정용 장애 목록."""

import pytest

from evidencegate.runner import Case
from evidencegate.suites import CALIBRATION_SEED_OFFSET, calibration
from opsworld.state import Family


def test_default_seed_offset_is_900000() -> None:
    assert CALIBRATION_SEED_OFFSET == 900_000


def test_families_are_balanced() -> None:
    cases = calibration(10)
    assert len(cases) == 30
    for family in Family:
        assert sum(c.family is family for c in cases) == 10


def test_seeds_start_at_offset_and_are_distinct_within_family() -> None:
    cases = calibration(20)
    assert min(c.incident_seed for c in cases) == 900_000
    for family in Family:
        seeds = [c.incident_seed for c in cases if c.family is family]
        assert seeds == list(range(900_000, 900_020))


def test_custom_offset() -> None:
    assert {c.incident_seed for c in calibration(2, seed_offset=5)} == {5, 6}


def test_families_are_interleaved() -> None:
    """앞부분만 돌다 멈춰도 family 비율이 고르게 남는다."""
    cases = calibration(3)
    assert cases[:3] == [Case(f, 900_000) for f in Family]


def test_deterministic() -> None:
    assert calibration(7) == calibration(7)


def test_zero_gives_empty_and_negative_raises() -> None:
    assert calibration(0) == []
    with pytest.raises(ValueError):
        calibration(-1)
