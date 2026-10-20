"""정확한 계산 도구 (Ticket 005 §1, §8-1·3·4·21·29). 손 계산 가능한 예로 확인한다."""

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from evidencegate.exact import (
    clopper_pearson,
    mean_fraction,
    null_diff_quantile_exact,
    percentile_exact,
    stratified_bootstrap_indices,
    stratified_subsample_indices,
)
from evidencegate.metrics import null_diff_quantile, percentile
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord

# ---------------------------------------------------------------------------
# 유리수 평균과 백분위수(§8-3·4)
# ---------------------------------------------------------------------------


def test_mean_fraction_is_exact() -> None:
    assert mean_fraction([1, 0, -1, 1]) == Fraction(1, 4)
    assert mean_fraction([1] * 10 + [0] * 200) == Fraction(10, 210)
    assert isinstance(mean_fraction([1, 0]), Fraction)


def test_mean_fraction_of_empty_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean_fraction([])


def test_percentile_exact_interpolates_linearly() -> None:
    assert percentile_exact([Fraction(0), Fraction(1)], Fraction(1, 2)) == Fraction(1, 2)
    assert percentile_exact([Fraction(3)], Fraction(19, 20)) == Fraction(3)
    values = [Fraction(i) for i in range(2000)]
    # 위치 = 19/20 * 1999 = 1899 + 1/20
    assert percentile_exact(values, Fraction(19, 20)) == Fraction(1899) + Fraction(1, 20)
    assert percentile_exact(values, Fraction(1, 20)) == Fraction(99) + Fraction(19, 20)


def test_percentile_exact_sorts_its_input() -> None:
    assert percentile_exact([Fraction(2), Fraction(0), Fraction(1)], Fraction(1, 2)) == 1


def test_percentile_exact_matches_the_float_percentile() -> None:
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(1, 300)
        values = [Fraction(rng.randint(-50, 50), rng.randint(1, 40)) for _ in range(n)]
        for q in (Fraction(1, 20), Fraction(1, 2), Fraction(19, 20), Fraction(39, 40)):
            exact = percentile_exact(values, q)
            assert math.isclose(
                float(exact), percentile([float(v) for v in values], float(q)), abs_tol=1e-12
            )


@pytest.mark.parametrize("q", [Fraction(-1, 10), Fraction(11, 10)])
def test_percentile_exact_rejects_q_outside_0_1(q: Fraction) -> None:
    with pytest.raises(ValueError, match="q"):
        percentile_exact([Fraction(1)], q)


def test_percentile_exact_of_empty_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile_exact([], Fraction(1, 2))


# ---------------------------------------------------------------------------
# 유리수 δ(N) (§8-29): null_diff_quantile과 같은 알고리즘·같은 추출 순서
# ---------------------------------------------------------------------------


def noise_record(
    family: str, seed: int, level: str, repeat: int, cd: bool, es: bool
) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey("p", "cfg", "tools", "s0", family, seed, repeat, level),
        policy_config={},
        llm_seed=0,
        correct_disposition=cd,
        evidence_safe=es,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-10-21T00:00:00+00:00",
        duration_s=1.0,
    )


def noise_records(n_cases: int, repeats: int, flip: float, seed: int) -> list[EpisodeRecord]:
    """케이스마다 기본값을 정하고, 반복마다 flip 확률로 CD를 뒤집는다. ES는 반복끼리 같다."""
    rng = random.Random(seed)
    records = []
    families = ("bad_deployment", "dependency_failure", "traffic_spike")
    for i in range(n_cases):
        family, level = families[i % 3], ("L1", "L2")[(i // 3) % 2]
        base_cd, base_es = rng.random() < 0.6, rng.random() < 0.4
        for r in range(repeats):
            cd = base_cd != (rng.random() < flip)
            records.append(noise_record(family, 900_000 + i, level, r, cd, base_es))
    return records


@pytest.mark.parametrize("n_cases", [24, 48, 96, 204, 210, 7])
@pytest.mark.parametrize("metric", ["correct_disposition", "evidence_safe"])
def test_exact_delta_matches_null_diff_quantile(n_cases: int, metric: str) -> None:
    records = noise_records(120, 5, flip=0.08, seed=3)
    exact = null_diff_quantile_exact(records, metric, n_cases)
    assert isinstance(exact, Fraction)
    assert math.isclose(float(exact), null_diff_quantile(records, metric, n_cases), abs_tol=1e-12)


def test_exact_delta_matches_with_other_seed_and_n_boot() -> None:
    records = noise_records(60, 3, flip=0.15, seed=11)
    exact = null_diff_quantile_exact(records, "correct_disposition", 30, n_boot=500, seed=4)
    expected = null_diff_quantile(records, "correct_disposition", 30, n_boot=500, seed=4)
    assert math.isclose(float(exact), expected, abs_tol=1e-12)


def test_exact_delta_has_the_expected_denominator() -> None:
    """|평균|은 1/N의 배수이고, q = 19/20 보간은 그 사이를 1/20 단위로 나눈다."""
    records = noise_records(120, 5, flip=0.08, seed=3)
    for n in (24, 210):
        value = null_diff_quantile_exact(records, "correct_disposition", n)
        assert (value * n * 20).denominator == 1


def test_exact_delta_of_identical_repeats_is_zero() -> None:
    records = noise_records(120, 5, flip=0.0, seed=3)
    assert null_diff_quantile_exact(records, "correct_disposition", 210) == 0
    assert null_diff_quantile_exact(records, "evidence_safe", 210) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_cases": 0}, "n_cases"),
        ({"n_cases": 5, "n_boot": 0}, "n_boot"),
        ({"n_cases": 5, "q": Fraction(3, 2)}, "q"),
    ],
)
def test_exact_delta_rejects_bad_arguments(kwargs: dict[str, object], match: str) -> None:
    records = noise_records(12, 2, flip=0.1, seed=1)
    with pytest.raises(ValueError, match=match):
        null_diff_quantile_exact(records, "correct_disposition", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Clopper-Pearson (§8-21)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.308497),
        (10, 10, 0.691503, 1.0),
        (5, 10, 0.187086, 0.812914),
        (1, 12, 0.002107, 0.384796),
        (0, 1, 0.0, 0.975),
    ],
)
def test_clopper_pearson_known_values(k: int, n: int, low: float, high: float) -> None:
    interval = clopper_pearson(k, n)
    assert interval is not None
    assert interval[0] == pytest.approx(low, abs=1e-6)
    assert interval[1] == pytest.approx(high, abs=1e-6)


def test_clopper_pearson_tails_are_two_and_a_half_percent() -> None:
    k, n = 3, 12
    interval = clopper_pearson(k, n)
    assert interval is not None
    low, high = interval

    def upper_tail(p: float) -> float:  # P(X >= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def lower_tail(p: float) -> float:  # P(X <= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    assert upper_tail(low) == pytest.approx(0.025, abs=1e-9)
    assert lower_tail(high) == pytest.approx(0.025, abs=1e-9)


def test_clopper_pearson_with_zero_denominator_is_undecidable() -> None:
    assert clopper_pearson(0, 0) is None


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (0, -1)])
def test_clopper_pearson_rejects_bad_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError):
        clopper_pearson(k, n)


# ---------------------------------------------------------------------------
# 층화 재표본 색인(§8-1·7·9·12): numpy PCG64, 호출마다 seed로 새로 시작
# ---------------------------------------------------------------------------


def test_bootstrap_indices_shape_and_range() -> None:
    blocks = stratified_bootstrap_indices([35, 35, 4], n_boot=2000, seed=0)
    assert [b.shape for b in blocks] == [(2000, 35), (2000, 35), (2000, 4)]
    for block, size in zip(blocks, [35, 35, 4], strict=True):
        assert block.min() >= 0
        assert block.max() < size
        assert block.dtype.kind == "i"


def test_bootstrap_indices_restart_from_the_seed_every_call() -> None:
    """공통 난수: 같은 seed로 부르면 모든 후보가 같은 색인을 받는다."""
    a = stratified_bootstrap_indices([35] * 6, n_boot=50, seed=0)
    b = stratified_bootstrap_indices([35] * 6, n_boot=50, seed=0)
    c = stratified_bootstrap_indices([35] * 6, n_boot=50, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_bootstrap_indices_follow_the_documented_draw_order() -> None:
    """칸 순서대로 PCG64(seed).integers(0, 칸 크기, (n_boot, 칸 크기))를 뽑는다."""
    rng = np.random.Generator(np.random.PCG64(0))
    expected = [rng.integers(0, 5, size=(3, 5)), rng.integers(0, 2, size=(3, 2))]
    got = stratified_bootstrap_indices([5, 2], n_boot=3, seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(got, expected, strict=True))


def test_subsample_without_replacement_keeps_cell_counts_and_uniqueness() -> None:
    resamples = stratified_subsample_indices(
        [35] * 6, take=[4] * 6, n_resample=200, seed=0, replace=False
    )
    assert len(resamples) == 200
    for cells in resamples:
        assert [len(c) for c in cells] == [4] * 6
        for c in cells:
            assert len(set(c.tolist())) == 4
            assert c.min() >= 0 and c.max() < 35


def test_subsample_with_replacement_allows_repeats_and_other_sizes() -> None:
    """RQ2 민감도: 칸 크기 10에서 18개를 복원 추출할 수 있다."""
    resamples = stratified_subsample_indices(
        [10] * 6, take=[18, 18, 3, 3, 9, 9], n_resample=100, seed=0, replace=True
    )
    assert all([len(c) for c in cells] == [18, 18, 3, 3, 9, 9] for cells in resamples)
    assert any(len(set(cells[0].tolist())) < 18 for cells in resamples)


def flatten(resamples: list[list[np.ndarray]]) -> list[list[int]]:
    return [x.tolist() for cells in resamples for x in cells]


def test_subsample_is_deterministic_and_seeded() -> None:
    a = stratified_subsample_indices([35] * 6, take=[8] * 6, n_resample=20, seed=0, replace=False)
    b = stratified_subsample_indices([35] * 6, take=[8] * 6, n_resample=20, seed=0, replace=False)
    c = stratified_subsample_indices([35] * 6, take=[8] * 6, n_resample=20, seed=5, replace=False)
    assert flatten(a) == flatten(b)
    assert flatten(a) != flatten(c)


def test_subsample_without_replacement_rejects_take_larger_than_cell() -> None:
    with pytest.raises(ValueError, match="take"):
        stratified_subsample_indices([4], take=[5], n_resample=1, seed=0, replace=False)


def test_subsample_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="take"):
        stratified_subsample_indices([4, 4], take=[1], n_resample=1, seed=0, replace=False)


@pytest.mark.parametrize("seed", range(20))
def test_exact_delta_matches_draw_by_draw(seed: int) -> None:
    """bootstrap 1회면 백분위수가 그 한 값이다. 추출 하나하나가 같아야 맞는다(잡음이 큰 데이터)."""
    records = noise_records(30, 3, flip=0.5, seed=seed)
    for n_cases in (1, 5, 31):
        exact = null_diff_quantile_exact(
            records, "correct_disposition", n_cases, n_boot=1, seed=seed
        )
        expected = null_diff_quantile(records, "correct_disposition", n_cases, n_boot=1, seed=seed)
        assert math.isclose(float(exact), expected, abs_tol=1e-12)


def test_subsample_resamples_differ_from_each_other() -> None:
    """재표본은 하나의 seed 흐름에서 이어서 뽑는다(§8-9·12). 매번 같으면 안 된다."""
    resamples = stratified_subsample_indices(
        [35] * 6, take=[4] * 6, n_resample=50, seed=0, replace=False
    )
    assert len({tuple(tuple(c.tolist()) for c in cells) for cells in resamples}) == 50


def test_subsample_follows_the_documented_draw_order() -> None:
    """재표본마다 칸 순서대로: 비복원 choice(크기, k, replace=False), 복원 integers(0, 크기, k)."""
    rng = np.random.Generator(np.random.PCG64(0))
    expected = [
        [rng.choice(5, size=2, replace=False), rng.choice(3, size=1, replace=False)]
        for _ in range(2)
    ]
    got = stratified_subsample_indices([5, 3], take=[2, 1], n_resample=2, seed=0, replace=False)
    assert [[c.tolist() for c in cells] for cells in got] == [
        [c.tolist() for c in cells] for cells in expected
    ]
    rng = np.random.Generator(np.random.PCG64(0))
    expected = [[rng.integers(0, 5, size=4), rng.integers(0, 3, size=2)]]
    got = stratified_subsample_indices([5, 3], take=[4, 2], n_resample=1, seed=0, replace=True)
    assert [[c.tolist() for c in cells] for cells in got] == [
        [c.tolist() for c in cells] for cells in expected
    ]
