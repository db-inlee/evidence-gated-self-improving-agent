"""평가 데이터 읽기·검증과 짝 표 (Ticket 005 §1 A, 설계 문서 7.8 A·B·D).

- 케이스 ID = (family, level, incident_seed). 정렬 순서도 이것이다(§8-1).
- 칸 = (family, level) 6개. split은 시드 범위로 정한다(`splits.py`).
- 결과 표는 split의 케이스를 빠짐없이, 한 번씩만 담아야 한다. 누락·중복이면 멈춘다.
- 게이트 짝 표: 후보와 S0 모두 반복 0(7.8 B). Oracle 짝 표: 후보 반복 0(§8-5) 대 S0 반복
  0~4 다수결(7.8 A·D).
- 처분 없이 끝난 에피소드도 기록된 값(CD = ES = 거짓) 그대로 쓴다(§8-6).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidencegate.eval_constants import GATE_REPEAT, S0_ORACLE_REPEATS
from evidencegate.metrics import METRICS
from evidencegate.runner import EpisodeRecord
from evidencegate.splits import SPLITS
from opsworld.state import Family, Level

CaseId = tuple[str, str, int]  # (family, level, incident_seed)
Cell = tuple[str, str]  # (family, level)

CELLS: tuple[Cell, ...] = tuple(sorted((f.value, lv.value) for f in Family for lv in Level))
GATE_SPLITS = ("dev_a", "dev_b")
ORACLE_SPLIT = "oracle"

_SPLIT_CASES: dict[str, tuple[CaseId, ...]] = {
    name: tuple(sorted((c.family.value, c.level.value, c.incident_seed) for c in make()))
    for name, make in SPLITS.items()
}
_CASE_SPLIT: dict[CaseId, str] = {c: name for name, cases in _SPLIT_CASES.items() for c in cases}


def split_cases(split: str) -> list[CaseId]:
    if split not in _SPLIT_CASES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(_SPLIT_CASES)}")
    return list(_SPLIT_CASES[split])


def split_of(case: CaseId) -> str:
    if case not in _CASE_SPLIT:
        raise ValueError(f"case {case} is not in any split")
    return _CASE_SPLIT[case]


def cell_of(case: CaseId) -> Cell:
    return case[0], case[1]


def case_of(record: EpisodeRecord) -> CaseId:
    return record.key.family, record.key.level, record.key.incident_seed


def check_run_conditions(
    records: Sequence[EpisodeRecord],
    *,
    policy_id: str | None = None,
    config_id: str | None = None,
    tool_spec_id: str | None = None,
) -> None:
    """모든 레코드가 한 실행 조건(정책, config, 도구 명세)이어야 한다. 스킬만 다를 수 있다."""
    if not records:
        raise ValueError("no records")
    conditions = {(r.key.policy_id, r.key.config_id, r.key.tool_spec_id) for r in records}
    if len(conditions) > 1:
        raise ValueError(f"records from mixed run conditions: {sorted(conditions)}")
    ((policy, config, tools),) = conditions
    for name, expected, actual in (
        ("policy_id", policy_id, policy),
        ("config_id", config_id, config),
        ("tool_spec_id", tool_spec_id, tools),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} is {actual!r}, expected {expected!r}")


def outcomes(
    records: Sequence[EpisodeRecord], *, skill_id: str, split: str, metric: str, repeat: int
) -> dict[CaseId, int]:
    """(스킬, split, 반복)의 케이스 → 0/1. split의 케이스 순서로, 빠짐없이 한 번씩."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    cases = split_cases(split)
    wanted = set(cases)
    found: dict[CaseId, int] = {}
    for r in records:
        case = case_of(r)
        if r.key.skill_id != skill_id or r.key.repeat != repeat or case not in wanted:
            continue
        if case in found:
            raise ValueError(f"duplicate record for {skill_id} {split} repeat {repeat}: {case}")
        found[case] = int(getattr(r, metric))
    missing = [c for c in cases if c not in found]
    if missing:
        raise ValueError(
            f"{len(missing)} missing cases for {skill_id} {split} repeat {repeat}, "
            f"e.g. {missing[0]}"
        )
    return {c: found[c] for c in cases}


def repeat_outcomes(
    records: Sequence[EpisodeRecord],
    *,
    skill_id: str,
    split: str,
    metric: str,
    repeats: Sequence[int],
) -> dict[int, dict[CaseId, int]]:
    return {
        r: outcomes(records, skill_id=skill_id, split=split, metric=metric, repeat=r)
        for r in repeats
    }


def majority(table: Mapping[int, Mapping[CaseId, int]]) -> dict[CaseId, int]:
    """반복 → 케이스 → 0/1에서 케이스별 다수결. 반복 수는 홀수, 모든 반복이 같은 케이스."""
    if len(table) % 2 == 0:
        raise ValueError(f"majority needs an odd number of repeats, got {len(table)}")
    first = list(next(iter(table.values())))
    if any(set(cases) != set(first) for cases in table.values()):
        raise ValueError("majority needs the same cases in every repeat")
    return {c: int(2 * sum(rep[c] for rep in table.values()) > len(table)) for c in first}


@dataclass(frozen=True)
class PairedTable:
    split: str
    metric: str
    cases: tuple[CaseId, ...]
    cells: tuple[Cell, ...]
    candidate: tuple[int, ...]
    baseline: tuple[int, ...]

    @property
    def deltas(self) -> tuple[int, ...]:
        return tuple(a - b for a, b in zip(self.candidate, self.baseline, strict=True))

    def cell_indices(self) -> list[list[int]]:
        """CELLS 순서로 칸마다 케이스 위치."""
        return [[i for i, c in enumerate(self.cells) if c == cell] for cell in CELLS]


def paired_gate(
    records: Sequence[EpisodeRecord],
    *,
    candidate_skill_id: str,
    s0_skill_id: str,
    split: str,
    metric: str,
) -> PairedTable:
    if split not in GATE_SPLITS:
        raise ValueError(f"gates use {GATE_SPLITS}; {split!r} (oracle) is scored separately")
    cand = outcomes(
        records, skill_id=candidate_skill_id, split=split, metric=metric, repeat=GATE_REPEAT
    )
    base = outcomes(records, skill_id=s0_skill_id, split=split, metric=metric, repeat=GATE_REPEAT)
    return _table(split, metric, cand, base)


def paired_oracle(
    records: Sequence[EpisodeRecord],
    *,
    candidate_skill_id: str,
    s0_skill_id: str,
    metric: str,
) -> PairedTable:
    cand = outcomes(
        records, skill_id=candidate_skill_id, split=ORACLE_SPLIT, metric=metric, repeat=GATE_REPEAT
    )
    reps = repeat_outcomes(
        records, skill_id=s0_skill_id, split=ORACLE_SPLIT, metric=metric, repeats=S0_ORACLE_REPEATS
    )
    return _table(ORACLE_SPLIT, metric, cand, majority(reps))


def _table(
    split: str, metric: str, cand: Mapping[CaseId, int], base: Mapping[CaseId, int]
) -> PairedTable:
    cases = tuple(cand)
    return PairedTable(
        split=split,
        metric=metric,
        cases=cases,
        cells=tuple(cell_of(c) for c in cases),
        candidate=tuple(cand[c] for c in cases),
        baseline=tuple(base[c] for c in cases),
    )
