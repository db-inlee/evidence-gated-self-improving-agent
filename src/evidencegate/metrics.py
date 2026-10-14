"""결과 집계와 노이즈 밴드. LLM 없이 저장된 레코드만 읽는다.

- `summarize`: 전체·family별 Correct Disposition, Evidence-Safe, correct_but_unsafe(처분은 맞지만
  Evidence-Safe가 아닌 비율) 비율(Wilson 95% 신뢰구간),
  실패 라벨 빈도, 형식 오류율(형식 오류 / 도구 호출), 평균 도구 호출 수, 평균 소요 시간.
- `noise_band`: 같은 스킬을 반복 실행한 결과에서 반복별 비율의 표본 표준편차, 반복 쌍마다
  케이스 단위 paired 차이 평균의 절댓값의 95번째 백분위수, 케이스별 불일치율.
- `null_diff_quantile`: 크기 N에서 S0 대 S0 비교로 우연히 나오는 |차이|의 분위수(bootstrap).
  Ticket 003 Budget Curve에 쓴다. 게이트 δ는 Ticket 003에서 정한다.
"""

import math
import random
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
    # 처분은 맞지만 근거가 부족한 비율. RQ1(처분 대 근거)은 이 차이가 있어야 측정할 수 있다.
    correct_but_unsafe: Proportion
    failure_labels: dict[str, int]  # 라벨 → 그 라벨이 붙은 에피소드 수
    format_error_rate: float  # 형식 오류 수 / 도구 호출 수
    mean_tool_calls: float
    mean_duration_s: float


@dataclass(frozen=True)
class Summary:
    overall: GroupSummary
    by_family: dict[str, GroupSummary]
    by_level: dict[str, GroupSummary]
    by_family_level: dict[str, GroupSummary]  # 키: "<family>/<level>"


@dataclass(frozen=True)
class NoiseBand:
    metric: str
    n_cases: int
    repeats: tuple[int, ...]
    per_repeat_rate: dict[int, float]
    repeat_std: float  # 반복별 비율의 표본 표준편차
    pair_mean_diffs: dict[tuple[int, int], float]  # (r, r') → 케이스 평균(x_r - x_r')
    paired_abs_p95: float  # |pair_mean_diffs|의 95번째 백분위수(선형 보간)
    pair_mismatch_rates: dict[tuple[int, int], float]  # (r, r') → 결과가 다른 케이스 비율
    case_mismatch_rate: float  # pair_mismatch_rates의 평균


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
    levels = sorted({r.key.level for r in records})
    return Summary(
        overall=_summarize_group(records),
        by_family={
            family: _summarize_group([r for r in records if r.key.family == family])
            for family in families
        },
        by_level={
            level: _summarize_group([r for r in records if r.key.level == level])
            for level in levels
        },
        by_family_level={
            f"{family}/{level}": _summarize_group(group)
            for family in families
            for level in levels
            if (group := [r for r in records if (r.key.family, r.key.level) == (family, level)])
        },
    )


def noise_band(records: Sequence[EpisodeRecord], metric: str) -> NoiseBand:
    outcomes = _outcomes_by_repeat(records, metric)
    repeats = tuple(sorted(outcomes))
    cases = sorted(outcomes[repeats[0]])
    n = len(cases)
    pairs = list(combinations(repeats, 2))

    per_repeat_rate = {r: sum(outcomes[r][c] for c in cases) / n for r in repeats}
    pair_mean_diffs = {
        (a, b): sum(outcomes[a][c] - outcomes[b][c] for c in cases) / n for a, b in pairs
    }
    pair_mismatch_rates = {
        (a, b): sum(outcomes[a][c] != outcomes[b][c] for c in cases) / n for a, b in pairs
    }
    return NoiseBand(
        metric=metric,
        n_cases=n,
        repeats=repeats,
        per_repeat_rate=per_repeat_rate,
        repeat_std=statistics.stdev(per_repeat_rate.values()),
        pair_mean_diffs=pair_mean_diffs,
        paired_abs_p95=percentile([abs(d) for d in pair_mean_diffs.values()], NOISE_PERCENTILE),
        pair_mismatch_rates=pair_mismatch_rates,
        case_mismatch_rate=statistics.fmean(pair_mismatch_rates.values()),
    )


def null_diff_quantile(
    records: Sequence[EpisodeRecord],
    metric: str,
    n_cases: int,
    q: float = NOISE_PERCENTILE,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """크기 n_cases에서 같은 스킬끼리 비교할 때 우연히 나오는 |평균 차이|의 q 분위수.

    매 회차 케이스 n_cases개를 복원 추출하고, 서로 다른 반복 쌍 (r, r')를 하나 무작위로 골라
    뽑힌 케이스들의 (x_r - x_r') 평균을 구한다. 난수는 `random.Random(seed)`로만 쓴다.
    """
    if n_cases < 1:
        raise ValueError(f"n_cases must be >= 1, got {n_cases}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")
    outcomes = _outcomes_by_repeat(records, metric)
    repeats = sorted(outcomes)
    cases = sorted(outcomes[repeats[0]])

    rng = random.Random(seed)
    abs_means = []
    for _ in range(n_boot):
        sample = rng.choices(cases, k=n_cases)
        a, b = rng.sample(repeats, 2)
        abs_means.append(abs(sum(outcomes[a][c] - outcomes[b][c] for c in sample) / n_cases))
    return percentile(abs_means, q)


def _outcomes_by_repeat(
    records: Sequence[EpisodeRecord], metric: str
) -> dict[int, dict[tuple[str, int, str], int]]:
    """반복 번호 → (family, incident_seed, level) → 0/1.

    반복이 2회 이상이고 모든 반복이 같은 케이스를 담아야 한다.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    _check_single_run(records)

    outcomes: dict[int, dict[tuple[str, int, str], int]] = {}
    for r in records:
        case = (r.key.family, r.key.incident_seed, r.key.level)
        outcomes.setdefault(r.key.repeat, {})[case] = int(getattr(r, metric))
    if len(outcomes) < 2:
        raise ValueError(f"noise estimates need at least 2 repeats, got {len(outcomes)}")
    first = set(next(iter(outcomes.values())))
    if any(set(cases) != first for cases in outcomes.values()):
        raise ValueError("every repeat must contain the same cases")
    return outcomes


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
        correct_but_unsafe=proportion(
            sum(r.correct_disposition and not r.evidence_safe for r in records), n
        ),
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


# ---------------------------------------------------------------------------
# 설정 변경 전후 비교: 케이스 층화 순열 검정 (Ticket 004 §1)
# ---------------------------------------------------------------------------

SHIFT_METRICS = (*METRICS, "correct_but_unsafe")


@dataclass(frozen=True)
class ShiftTest:
    metric: str
    n_cases: int
    observed_diff: float  # 케이스 평균(after 평균 - before 평균)
    threshold: float  # 순열 분포 |차이|의 q 분위수
    different: bool  # |observed_diff| > threshold


def permutation_shift_test(
    before: Sequence[EpisodeRecord],
    after: Sequence[EpisodeRecord],
    metric: str,
    n_perm: int = 2000,
    q: float = NOISE_PERCENTILE,
    seed: int = 0,
) -> ShiftTest:
    """같은 케이스를 두 설정으로 반복 실행한 결과가 다른지 본다.

    케이스마다 두 설정의 결과를 합쳐 원래 그룹 크기대로 무작위로 다시 나누고, 두 그룹 평균의 차이를
    케이스 전체로 평균한다. n_perm회 반복한 |차이|의 q 분위수(선형 보간)를 임계값으로 쓰고, 실제
    |차이|가 임계값보다 크면 다르다고 본다. 난수는 `random.Random(seed)`로만 쓴다.
    """
    if metric not in SHIFT_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {SHIFT_METRICS}")
    old = _outcomes_by_case(before, metric)
    new = _outcomes_by_case(after, metric)
    if set(old) != set(new):
        raise ValueError("before and after must contain the same cases")
    cases = sorted(old)

    def case_mean_diff(groups: list[tuple[list[int], list[int]]]) -> float:
        diffs = [statistics.fmean(b) - statistics.fmean(a) for a, b in groups]
        return statistics.fmean(diffs)

    observed = case_mean_diff([(old[c], new[c]) for c in cases])
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        groups = []
        for c in cases:
            pooled = old[c] + new[c]
            rng.shuffle(pooled)
            groups.append((pooled[: len(old[c])], pooled[len(old[c]) :]))
        null.append(abs(case_mean_diff(groups)))
    threshold = percentile(null, q)
    return ShiftTest(metric, len(cases), observed, threshold, abs(observed) > threshold)


def _outcomes_by_case(
    records: Sequence[EpisodeRecord], metric: str
) -> dict[tuple[str, int, str], list[int]]:
    _check_single_run(records)
    outcomes: dict[tuple[str, int, str], list[int]] = {}
    for r in sorted(records, key=lambda r: r.key.repeat):
        if metric == "correct_but_unsafe":
            value = int(r.correct_disposition and not r.evidence_safe)
        else:
            value = int(bool(getattr(r, metric)))
        outcomes.setdefault((r.key.family, r.key.incident_seed, r.key.level), []).append(value)
    return outcomes
