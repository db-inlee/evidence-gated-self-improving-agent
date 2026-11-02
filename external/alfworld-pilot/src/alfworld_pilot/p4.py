"""P4 S0 기준률과 잡음 (Ticket 008 §4 P4).

P5 표적 칸 규칙(2026-11-03 결정 3, constants.P5_TARGET_RULE): P4 반복 3회 평균 성공률이 가장
높은 칸, 동률이면 SIX_TYPES 순서에서 앞. 기준률이 바닥인 칸은 하락을 보일 수 없어서(G6)
미리 정하지 않고 규칙으로 고정한다.
"""

from collections.abc import Mapping

from alfworld_pilot.p1 import SIX_TYPES


def choose_target_cell(mean_success: Mapping[str, float]) -> str:
    if set(mean_success) != set(SIX_TYPES):
        raise ValueError("need the mean success rate of all six task types")
    best = max(mean_success[t] for t in SIX_TYPES)
    return next(t for t in SIX_TYPES if mean_success[t] == best)
