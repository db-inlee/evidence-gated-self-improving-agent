"""보조 지표 E-a, E-b, E-c (Ticket 008 §2). 원시 관측에서 코드로 계산한다. LLM 판정 없음.

E-a (결정 2): 성공한 `take X from R`마다, 그 걸음 **이전** 관측(첫 관측 포함) 중에 R 안·위에 X가
보인다는 문장이 있었는가. 에피소드 판정은 `won` 그리고 성공한 take가 하나 이상 그리고 모두 충족.

R의 내용을 보이는 문장은 alfworld 0.4.2 관측에서 두 가지다(P1·P2에서 확인).
- `On the R, you see a X, a Y, and a Z.` (도착, examine)
- `You open the R. The R is open. In it, you see a X.` / `The R is open. In it, you see ...`
  ("In it"은 같은 관측에서 바로 앞 문장의 R에만 붙인다)
"you see nothing."은 아무것도 보이지 않은 것이다. 방 첫 관측의 "Looking quickly around you, you see
..."는 수용체 목록이라 내용으로 치지 않는다.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from alfworld_pilot.constants import E_C_K

_NAME = r"[a-z]+ \d+"
_ON = re.compile(rf"On the (?P<r>{_NAME}), you see (?P<items>[^.]*)\.")
_IN = re.compile(
    rf"(?:You open the |The )(?P<r>{_NAME})(?: is open)?\.\s+"
    rf"(?:The (?P=r) is open\.\s+)?In it, you see (?P<items>[^.]*)\."
)
_TAKE = re.compile(rf"^take (?P<o>{_NAME}) from (?P<r>{_NAME})$")
_ITEM = re.compile(rf"^(?:and )?(?:a |an )?(?P<o>{_NAME})$")


@dataclass(frozen=True)
class StepRecord:
    command: str
    observation: str
    admissible: bool  # 행동 직전 가능한 명령 목록에 있었는가(E-b 전용, 프롬프트에는 없음)


def _items(text: str) -> list[str]:
    if text.strip() == "nothing":
        return []
    out = []
    for part in text.split(", "):
        m = _ITEM.match(part.strip())
        if m:
            out.append(m["o"])
    return out


def contents_seen(observation: str) -> set[tuple[str, str]]:
    """관측에서 (물건, 수용체) 쌍. 수용체 안·위에 그 물건이 보인다는 문장만 센다."""
    seen = set()
    for pattern in (_ON, _IN):
        for m in pattern.finditer(observation):
            seen |= {(o, m["r"]) for o in _items(m["items"])}
    return seen


def successful_take(command: str, observation: str) -> tuple[str, str] | None:
    m = _TAKE.match(command)
    if m and f"You pick up the {m['o']} from the {m['r']}." in observation:
        return (m["o"], m["r"])
    return None


def e_a_holds(first_observation: str, steps: Sequence[StepRecord], *, won: bool) -> bool:
    if not won:
        return False
    seen = contents_seen(first_observation)
    took = False
    for s in steps:
        take = successful_take(s.command, s.observation)
        if take is not None:
            if take not in seen:
                return False
            took = True
        seen |= contents_seen(s.observation)
    return took


def e_b_rate(steps: Sequence[StepRecord]) -> tuple[int, int]:
    return (sum(1 for s in steps if s.admissible), len(steps))


def e_c_holds(*, won: bool, steps: int, expert_steps: int | None) -> bool | None:
    """성공 걸음 수 ≤ k x 전문가 계획 길이. 전문가가 못 푼 게임은 정의하지 않는다(None)."""
    if expert_steps is None:
        return None
    return won and steps <= E_C_K * expert_steps
