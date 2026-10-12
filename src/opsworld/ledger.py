"""증거 원장. 도구가 반환한 원시 관측을 순서대로 쌓기만 하고 해석하지 않는다."""

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Observation:
    step: int  # 몇 번째 도구 호출인가 (0부터)
    tool: str  # 어느 도구인가
    payload: dict[str, Any]  # 도구가 반환한 원시 값 그대로


class Ledger:
    def __init__(self) -> None:
        self._observations: list[Observation] = []

    def append(self, tool: str, payload: dict[str, Any]) -> Observation:
        # 호출자가 반환값을 바꿔도 원장 기록은 바뀌지 않도록 깊은 복사로 저장한다.
        observation = Observation(len(self._observations), tool, copy.deepcopy(payload))
        self._observations.append(observation)
        return observation

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def __len__(self) -> int:
        return len(self._observations)


def observations_before(ledger: Ledger, action_step: int) -> tuple[Observation, ...]:
    """조치 이전(`step < action_step`) 관측만 돌려준다. 근거 판정은 이 함수만 거친다."""
    return tuple(o for o in ledger.observations if o.step < action_step)
