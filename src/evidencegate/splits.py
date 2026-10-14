"""데이터 분할 (Ticket 004 §2).

| split  | 크기 | 칸당 | 시드(칸마다 같은 범위) | 용도 |
| Dev-A  |  60 | 10 | 100,000 ~ 100,009 | proposer 입력(S0 실패 궤적), C1 판정 |
| Dev-B  | 210 | 35 | 200,000 ~ 200,034 | C2·C3·C4 판정, Budget Curve 재표본 |
| Oracle | 300 | 50 | 300,000 ~ 300,049 | 후보 효과 추정(사후 채점) |

칸 = family x level 6개, 구성은 모두 같다(family 1:1:1, L1:L2 1:1). 같은 시드를 칸끼리 공유한다
(생성기 난수 흐름이 family·level로 갈림). split끼리, 그리고 보정 목록(900,000번대)과 시드 범위가
겹치지 않는다. 칸 크기를 정수로 맞추려고 설계 문서의 Dev-A 50, Dev-B 200을 60, 210으로 바꿨다.
"""

from collections.abc import Callable

from evidencegate.runner import Case
from evidencegate.suites import calibration
from opsworld.state import Level

LEVELS = (Level.L1, Level.L2)

DEV_A_SEED_OFFSET = 100_000
DEV_A_PER_CELL = 10
DEV_B_SEED_OFFSET = 200_000
DEV_B_PER_CELL = 35
ORACLE_SEED_OFFSET = 300_000
ORACLE_PER_CELL = 50


def dev_a() -> list[Case]:
    return calibration(DEV_A_PER_CELL, seed_offset=DEV_A_SEED_OFFSET, levels=LEVELS)


def dev_b() -> list[Case]:
    return calibration(DEV_B_PER_CELL, seed_offset=DEV_B_SEED_OFFSET, levels=LEVELS)


def oracle() -> list[Case]:
    return calibration(ORACLE_PER_CELL, seed_offset=ORACLE_SEED_OFFSET, levels=LEVELS)


SPLITS: dict[str, Callable[[], list[Case]]] = {"dev_a": dev_a, "dev_b": dev_b, "oracle": oracle}
