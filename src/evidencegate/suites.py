"""실행할 장애 목록.

보정용 시드는 900_000부터 쓴다. Ticket 003의 Dev-A/Dev-B/Oracle 시드 범위와 겹치지 않게 하기
위해서다.
"""

from collections.abc import Sequence

from evidencegate.runner import Case
from opsworld.state import Family, Level

CALIBRATION_SEED_OFFSET = 900_000


def calibration(
    n_per_family: int,
    seed_offset: int = CALIBRATION_SEED_OFFSET,
    levels: Sequence[Level] = (Level.L1,),
) -> list[Case]:
    """family 3종을 같은 수로 섞은 목록. family를 번갈아 놓아 중간에 멈춰도 비율이 고르다.

    level마다 같은 시드 범위를 쓰고(키가 level로 갈림), level 순서대로 블록을 이어 붙인다.
    """
    if n_per_family < 0:
        raise ValueError(f"n_per_family must be >= 0, got {n_per_family}")
    return [
        Case(family, seed_offset + i, level)
        for level in levels
        for i in range(n_per_family)
        for family in Family
    ]
