"""결과 집계와 노이즈 밴드. LLM 없이 저장된 레코드만 읽는다.

- `summarize`: 전체·family별 Correct Disposition, Evidence-Safe 비율(Wilson 95% 신뢰구간),
  실패 라벨 빈도, 형식 오류율(형식 오류 / 도구 호출), 평균 도구 호출 수, 평균 소요 시간.
- `noise_band`: 같은 스킬을 반복 실행한 결과에서 반복별 비율의 표본 표준편차와, 반복 쌍마다
  케이스 단위 paired 차이 평균의 절댓값의 95번째 백분위수. 게이트 δ는 Ticket 003에서 정한다.
"""

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from evidencegate.runner import EpisodeRecord

WILSON_Z_95 = 1.959963984540054
NOISE_PERCENTILE = 0.95
METRICS = ("correct_disposition", "evidence_safe")


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class GroupSummary:
    n: int
    correct_disposition: Proportion
    evidence_safe: Proportion
    failure_labels: dict[str, int]  # 라벨 → 그 라벨이 붙은 에피소드 수
    format_error_rate: float  # 형식 오류 수 / 도구 호출 수
    mean_tool_calls: float
    mean_duration_s: float


@dataclass(frozen=True)
class Summary:
    overall: GroupSummary
    by_family: dict[str, GroupSummary]


@dataclass(frozen=True)
class NoiseBand:
    metric: str
    n_cases: int
    repeats: tuple[int, ...]
    per_repeat_rate: dict[int, float]
    repeat_std: float  # 반복별 비율의 표본 표준편차
    pair_mean_diffs: dict[tuple[int, int], float]  # (r, r') → 케이스 평균(x_r - x_r')
    paired_abs_p95: float  # |pair_mean_diffs|의 95번째 백분위수(선형 보간)


def wilson_interval(successes: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"invalid proportion {successes}/{n}")
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def proportion(successes: int, n: int) -> Proportion:
    low, high = wilson_interval(successes, n)
    return Proportion(successes, n, successes / n, low, high)


def summarize(records: Sequence[EpisodeRecord]) -> Summary:
    _check_single_run(records)
    families = sorted({r.key.family for r in records})
    return Summary(
        overall=_summarize_group(records),
        by_family={
            family: _summarize_group([r for r in records if r.key.family == family])
            for family in families
        },
    )


def noise_band(records: Sequence[EpisodeRecord], metric: str) -> NoiseBand:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    _check_single_run(records)

    by_repeat: dict[int, dict[tuple[str, int], bool]] = {}
    for r in records:
        case = (r.key.family, r.key.incident_seed)
        by_repeat.setdefault(r.key.repeat, {})[case] = bool(getattr(r, metric))
    repeats = tuple(sorted(by_repeat))
    if len(repeats) < 2:
        raise ValueError(f"noise band needs at least 2 repeats, got {len(repeats)}")
    cases = set(by_repeat[repeats[0]])
    if any(set(by_repeat[r]) != cases for r in repeats):
        raise ValueError("every repeat must contain the same cases")

    ordered = sorted(cases)
    per_repeat_rate = {r: sum(by_repeat[r][c] for c in ordered) / len(ordered) for r in repeats}
    pair_mean_diffs = {
        (a, b): sum(by_repeat[a][c] - by_repeat[b][c] for c in ordered) / len(ordered)
        for a, b in combinations(repeats, 2)
    }
    return NoiseBand(
        metric=metric,
        n_cases=len(ordered),
        repeats=repeats,
        per_repeat_rate=per_repeat_rate,
        repeat_std=statistics.stdev(per_repeat_rate.values()),
        pair_mean_diffs=pair_mean_diffs,
        paired_abs_p95=percentile([abs(d) for d in pair_mean_diffs.values()], NOISE_PERCENTILE),
    )


def percentile(values: Sequence[float], q: float) -> float:
    """선형 보간 백분위수(numpy 기본 방식과 같음)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize_group(records: Sequence[EpisodeRecord]) -> GroupSummary:
    n = len(records)
    tool_calls = sum(len(r.trace.turns) for r in records)
    format_errors = sum(r.trace.format_errors for r in records)
    labels = Counter(label for r in records for label in r.failure_labels)
    return GroupSummary(
        n=n,
        correct_disposition=proportion(sum(r.correct_disposition for r in records), n),
        evidence_safe=proportion(sum(r.evidence_safe for r in records), n),
        failure_labels=dict(labels),
        format_error_rate=format_errors / tool_calls if tool_calls else 0.0,
        mean_tool_calls=tool_calls / n,
        mean_duration_s=sum(r.duration_s for r in records) / n,
    )


def _check_single_run(records: Sequence[EpisodeRecord]) -> None:
    """한 실행 조건(policy, config, 도구 명세, 스킬)의 결과만 집계한다."""
    if not records:
        raise ValueError("no records to aggregate")
    runs = {(r.key.policy_id, r.key.config_id, r.key.tool_spec_id, r.key.skill_id) for r in records}
    if len(runs) > 1:
        raise ValueError(f"records from mixed runs: {sorted(runs)}")
