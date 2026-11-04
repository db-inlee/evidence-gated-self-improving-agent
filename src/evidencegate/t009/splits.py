"""Ticket 009 분할(사전 등록 1절). v1 분할 모듈(`evidencegate.splits`)은 고치지 않는다.

| split | 크기 | 칸당 | 시드(칸마다 같은 범위) |
| t009_dev_a | 60 | 10 | 500,000 ~ 500,009 |
| t009_dev_b | 420 | 70 | 600,000 ~ 600,069 |
| t009_oracle | 300 | 50 | 700,000 ~ 700,049 |
| t009_calibration | 300 | 50 | 900,020 ~ 900,069 (v1 보정 900,000~900,019에 더함) |

- split 이름은 모두 `t009_`로 시작한다. 실행기가 split 이름으로 기록 폴더(`runs/<split>/…`)를
  정하므로, v1 기록(특히 v1 보정 기록 `runs/calibration/…`, v1 δ(N) 재현의 입력)에 이어 쓰지 않기
  위해서다.
- 케이스는 `(family, level, incident_seed)`이고 정렬 순서(칸, incident_seed)다.
- h(n_c)와 δ(N)의 보정은 v1 보정 20개와 추가 50개를 합친 칸당 70개다(`calibration_all`).
"""

from evidencegate.eval_data import CaseId
from evidencegate.suites import CALIBRATION_SEED_OFFSET, calibration
from opsworld.state import Level

LEVELS = (Level.L1, Level.L2)
DEV_A = ("t009_dev_a", 500_000, 10)
DEV_B = ("t009_dev_b", 600_000, 70)
ORACLE = ("t009_oracle", 700_000, 50)
CALIBRATION_ADD = ("t009_calibration", CALIBRATION_SEED_OFFSET + 20, 50)
CALIBRATION_V1_PER_CELL = 20

SPLITS: dict[str, tuple[int, int]] = {
    name: (offset, per_cell) for name, offset, per_cell in (DEV_A, DEV_B, ORACLE, CALIBRATION_ADD)
}


def _cases(offset: int, per_cell: int) -> list[CaseId]:
    return sorted(
        (c.family.value, c.level.value, c.incident_seed)
        for c in calibration(per_cell, seed_offset=offset, levels=LEVELS)
    )


def split_cases(name: str) -> list[CaseId]:
    if name not in SPLITS:
        raise ValueError(f"unknown Ticket 009 split {name!r}; expected one of {sorted(SPLITS)}")
    return _cases(*SPLITS[name])


CALIBRATION_V1: list[CaseId] = _cases(CALIBRATION_SEED_OFFSET, CALIBRATION_V1_PER_CELL)


def calibration_all() -> list[CaseId]:
    """h(n_c)·δ(N)용 보정 케이스 칸당 70개(v1 20 + 추가 50)."""
    return sorted(CALIBRATION_V1 + split_cases("t009_calibration"))
