"""Ticket 014 분할(사전 등록 2절). 009 분할 모듈은 고치지 않고 같은 케이스 규칙을 쓴다.

| split | 칸당 | 크기 | incident seed |
| t014_dev_a | 10 | 60 | 1,100,000 ~ 1,100,009 |
| t014_calibration | 150 | 900 | 1,200,000 ~ 1,200,149 |
| t014_dev_b | 150 | 900 | 1,300,000 ~ 1,300,149 |
| t014_oracle | 100 | 600 | 1,400,000 ~ 1,400,099 |

- 케이스는 `(family, level, incident_seed)`이고 정렬 순서(칸, incident_seed)다. 칸마다 같은 seed
  범위다(009와 같음).
"""

from evidencegate.eval_data import CaseId, Cell, cell_of
from evidencegate.suites import calibration
from evidencegate.t009.splits import LEVELS
from evidencegate.t014.settings import CALIBRATION, DEV_A, DEV_B, ORACLE

SPLITS: dict[str, tuple[int, int]] = {
    name: (offset, per_cell) for name, offset, per_cell in (DEV_A, CALIBRATION, DEV_B, ORACLE)
}


def split_cases(name: str) -> list[CaseId]:
    if name not in SPLITS:
        raise ValueError(f"unknown Ticket 014 split {name!r}; expected one of {sorted(SPLITS)}")
    offset, per_cell = SPLITS[name]
    return sorted(
        (c.family.value, c.level.value, c.incident_seed)
        for c in calibration(per_cell, seed_offset=offset, levels=LEVELS)
    )


def cell_cases(name: str, cell: Cell) -> list[CaseId]:
    return [c for c in split_cases(name) if cell_of(c) == cell]
