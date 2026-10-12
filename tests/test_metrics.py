"""집계(summarize)와 노이즈 밴드(noise_band). 손으로 만든 결과로 기대값을 확인한다."""

import math
from collections.abc import Sequence

import pytest

from evidencegate.metrics import noise_band, summarize, wilson_interval
from evidencegate.policy import PolicyTrace, StopReason, TurnRecord
from evidencegate.runner import EpisodeKey, EpisodeRecord

FAMILIES = ("bad_deployment", "dependency_failure", "traffic_spike")


def turn(format_error: bool = False) -> TurnRecord:
    return TurnRecord("get_service_metrics", {"service_id": "s"}, "{}", format_error)


def make_record(
    *,
    family: str = "bad_deployment",
    seed: int = 0,
    repeat: int = 0,
    correct: bool = True,
    safe: bool = True,
    labels: Sequence[str] = (),
    turns: int = 3,
    format_errors: int = 0,
    duration_s: float = 1.0,
    policy_id: str = "scripted:t",
) -> EpisodeRecord:
    trace_turns = tuple(turn(i < format_errors) for i in range(turns))
    return EpisodeRecord(
        key=EpisodeKey(policy_id, "cfg", "tools", "skill", family, seed, repeat),
        policy_config={},
        llm_seed=0,
        correct_disposition=correct,
        evidence_safe=safe,
        action="observe_only",
        missing_predicates=(),
        failure_labels=tuple(labels),
        trace=PolicyTrace(trace_turns, format_errors, StopReason.ACTION),
        started_at="2026-10-12T00:00:00+00:00",
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Wilson 95% 신뢰구간
# ---------------------------------------------------------------------------


def test_wilson_all_success() -> None:
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-4)
    assert high == pytest.approx(1.0)


def test_wilson_all_failure() -> None:
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.2775, abs=1e-4)


def test_wilson_half() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


def test_wilson_rejects_empty_or_invalid() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_all_success() -> None:
    records = [make_record(family=f, seed=i) for f in FAMILIES for i in range(4)]
    summary = summarize(records)
    assert summary.overall.n == 12
    assert summary.overall.correct_disposition.successes == 12
    assert summary.overall.correct_disposition.rate == 1.0
    assert summary.overall.evidence_safe.rate == 1.0
    assert summary.overall.failure_labels == {}
    assert set(summary.by_family) == set(FAMILIES)


def test_summarize_all_failure() -> None:
    records = [
        make_record(seed=i, correct=False, safe=False, labels=("NO_DISPOSITION",)) for i in range(5)
    ]
    summary = summarize(records)
    assert summary.overall.correct_disposition.rate == 0.0
    assert summary.overall.evidence_safe.rate == 0.0
    assert summary.overall.correct_disposition.ci_low == 0.0
    assert summary.overall.failure_labels == {"NO_DISPOSITION": 5}


def test_summarize_mixed_by_family_with_labels_and_means() -> None:
    records = [
        make_record(family="bad_deployment", seed=0, turns=4, duration_s=2.0),
        make_record(
            family="bad_deployment",
            seed=1,
            safe=False,
            labels=("PREMATURE_ACTION",),
            turns=2,
            format_errors=1,
            duration_s=4.0,
        ),
        make_record(
            family="dependency_failure",
            seed=0,
            correct=False,
            safe=False,
            labels=("PREMATURE_ACTION", "WRONG_INTERVENTION", "UNNECESSARY_INTERVENTION"),
            turns=3,
            duration_s=6.0,
        ),
    ]
    summary = summarize(records)

    overall = summary.overall
    assert overall.n == 3
    assert overall.correct_disposition.successes == 2
    assert overall.evidence_safe.successes == 1
    assert overall.failure_labels == {
        "PREMATURE_ACTION": 2,
        "WRONG_INTERVENTION": 1,
        "UNNECESSARY_INTERVENTION": 1,
    }
    assert overall.format_error_rate == pytest.approx(1 / 9)  # 형식 오류 / 도구 호출
    assert overall.mean_tool_calls == pytest.approx(3.0)
    assert overall.mean_duration_s == pytest.approx(4.0)
    low, high = wilson_interval(2, 3)
    assert overall.correct_disposition.ci_low == pytest.approx(low)
    assert overall.correct_disposition.ci_high == pytest.approx(high)

    bad = summary.by_family["bad_deployment"]
    assert (bad.n, bad.correct_disposition.rate, bad.evidence_safe.rate) == (2, 1.0, 0.5)
    assert bad.mean_duration_s == pytest.approx(3.0)
    dep = summary.by_family["dependency_failure"]
    assert (dep.n, dep.correct_disposition.rate) == (1, 0.0)
    assert "traffic_spike" not in summary.by_family


def test_summarize_rejects_empty_and_mixed_runs() -> None:
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError, match="mixed"):
        summarize([make_record(), make_record(seed=1, policy_id="scripted:other")])


# ---------------------------------------------------------------------------
# noise_band
# ---------------------------------------------------------------------------

# 케이스 4개 x 반복 3회. 반복별 correct_disposition:
#   r0: 1 1 1 1 (1.00)   r1: 1 1 0 0 (0.50)   r2: 1 0 0 0 (0.25)
OUTCOMES = {0: [1, 1, 1, 1], 1: [1, 1, 0, 0], 2: [1, 0, 0, 0]}


def band_records(metric_values: dict[int, list[int]], other: int = 1) -> list[EpisodeRecord]:
    records = []
    for repeat, values in metric_values.items():
        for case, value in enumerate(values):
            family = FAMILIES[case % 3]
            records.append(
                make_record(
                    family=family, seed=case, repeat=repeat, correct=bool(value), safe=bool(other)
                )
            )
    return records


def test_noise_band_with_known_repeat_differences() -> None:
    band = noise_band(band_records(OUTCOMES), "correct_disposition")

    assert band.n_cases == 4
    assert band.repeats == (0, 1, 2)
    assert band.per_repeat_rate == {0: 1.0, 1: 0.5, 2: 0.25}
    rates = [1.0, 0.5, 0.25]
    mean = sum(rates) / 3
    expected_std = math.sqrt(sum((r - mean) ** 2 for r in rates) / 2)  # 표본 표준편차
    assert band.repeat_std == pytest.approx(expected_std)
    assert band.pair_mean_diffs == {
        (0, 1): pytest.approx(0.5),
        (0, 2): pytest.approx(0.75),
        (1, 2): pytest.approx(0.25),
    }
    # |차이| = [0.25, 0.5, 0.75], 선형 보간 95번째 백분위수 = 0.5 + 0.9 * 0.25
    assert band.paired_abs_p95 == pytest.approx(0.725)


def test_noise_band_identical_repeats_is_zero() -> None:
    same = {r: [1, 0, 1, 1] for r in range(5)}
    band = noise_band(band_records(same), "correct_disposition")
    assert band.repeat_std == 0.0
    assert band.paired_abs_p95 == 0.0
    assert len(band.pair_mean_diffs) == 10


def test_noise_band_uses_the_requested_metric() -> None:
    records = band_records(OUTCOMES, other=0)  # evidence_safe는 전부 거짓
    band = noise_band(records, "evidence_safe")
    assert band.metric == "evidence_safe"
    assert band.per_repeat_rate == {0: 0.0, 1: 0.0, 2: 0.0}
    assert band.paired_abs_p95 == 0.0


def test_noise_band_requires_two_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        noise_band(band_records({0: [1, 0]}), "correct_disposition")


def test_noise_band_requires_the_same_cases_in_every_repeat() -> None:
    records = band_records({0: [1, 0, 1], 1: [1, 0]})
    with pytest.raises(ValueError, match="cases"):
        noise_band(records, "correct_disposition")


def test_noise_band_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        noise_band(band_records(OUTCOMES), "accuracy")
