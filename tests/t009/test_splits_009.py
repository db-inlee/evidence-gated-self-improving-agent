"""Ticket 009 분할(사전 등록 1절): 시드 범위와 v1 분할·L3·보정과의 겹침."""

from collections.abc import Iterable

import pytest

from evidencegate import splits as v1_splits
from evidencegate.eval_constants import L3_SPLIT_SEEDS
from evidencegate.eval_data import CELLS
from evidencegate.suites import CALIBRATION_SEED_OFFSET
from evidencegate.t009.splits import (
    CALIBRATION_V1,
    SPLITS,
    calibration_all,
    split_cases,
)

V1_CALIBRATION = range(CALIBRATION_SEED_OFFSET, CALIBRATION_SEED_OFFSET + 20)


def seeds(cases: Iterable[tuple[str, str, int]]) -> set[int]:
    return {c[2] for c in cases}


@pytest.mark.parametrize(
    ("name", "lo", "hi", "per_cell"),
    [
        ("t009_dev_a", 500_000, 500_009, 10),
        ("t009_dev_b", 600_000, 600_069, 70),
        ("t009_oracle", 700_000, 700_049, 50),
        ("t009_calibration", 900_020, 900_069, 50),
    ],
)
def test_split_seed_ranges_and_cell_sizes(name: str, lo: int, hi: int, per_cell: int) -> None:
    cases = split_cases(name)
    assert seeds(cases) == set(range(lo, hi + 1))
    assert len(cases) == 6 * per_cell == len(set(cases))
    for cell in CELLS:
        assert sum(1 for c in cases if c[:2] == cell) == per_cell
    assert cases == sorted(cases)  # 칸, incident_seed 순서


def test_split_names_never_reuse_v1_store_folders() -> None:
    # 실행기는 split 이름으로 기록 폴더(runs/<split>/...)를 정한다
    assert all(name.startswith("t009_") for name in SPLITS)
    assert not set(SPLITS) & {"dev_a", "dev_b", "oracle", "calibration", "l3_oracle", "l3_check"}


def test_no_overlap_with_v1_splits_l3_or_v1_calibration() -> None:
    v1 = set()
    for make in v1_splits.SPLITS.values():
        v1 |= {c.incident_seed for c in make()}
    for offset, n in L3_SPLIT_SEEDS.values():
        v1 |= set(range(offset, offset + n))
    v1 |= set(V1_CALIBRATION)
    ours = [seeds(split_cases(name)) for name in SPLITS]
    for i, a in enumerate(ours):
        assert not a & v1
        for b in ours[i + 1 :]:
            assert not a & b


def test_calibration_all_is_v1_plus_the_addition() -> None:
    allc = calibration_all()
    assert seeds(allc) == set(range(900_000, 900_070))
    assert len(allc) == 6 * 70
    assert seeds(CALIBRATION_V1) == set(V1_CALIBRATION)
    assert set(allc) == set(CALIBRATION_V1) | set(split_cases("t009_calibration"))


def test_unknown_split_is_refused() -> None:
    with pytest.raises(ValueError):
        split_cases("dev_b")
