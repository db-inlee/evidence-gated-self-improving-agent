"""정확한 계산 도구 (Ticket 005 §1, §8 해석 1·3·4·21·29).

- 판정 비교는 유리수(`Fraction`)로 한다(§8-4). 평균 Δ는 1/N의 배수라 경계에서 등호가 실제로 생긴다.
- `null_diff_quantile_exact`는 `metrics.null_diff_quantile`과 같은 알고리즘·같은
  `random.Random(seed)` 추출 순서로 유리수를 돌려준다(§8-29). 기존 함수는 그대로 둔다.
- 층화 재표본 색인은 `numpy.random.Generator(PCG64(seed))`로 만들고, 호출마다 seed로 새로
  시작한다(§8-1). 그래서 같은 seed로 부른 모든 후보가 같은 색인을 쓴다(공통 난수).
"""

import math
import random
from collections.abc import Callable, Sequence
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from evidencegate.eval_constants import CI_ALPHA, N_BOOT, NOISE_Q, SEED
from evidencegate.metrics import _outcomes_by_repeat
from evidencegate.runner import EpisodeRecord

IndexArray = npt.NDArray[np.int64]
_BISECTION_STEPS = 200


def mean_fraction(values: Sequence[int | Fraction]) -> Fraction:
    if not values:
        raise ValueError("mean of empty sequence")
    return Fraction(sum(values, Fraction(0)), len(values))


def percentile_exact(values: Sequence[Fraction], q: Fraction) -> Fraction:
    """선형 보간 백분위수(`metrics.percentile`과 같은 규칙)를 유리수로 계산한다."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def null_diff_quantile_exact(
    records: Sequence[EpisodeRecord],
    metric: str,
    n_cases: int,
    q: Fraction = NOISE_Q,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Fraction:
    """δ(N)의 유리수판. 추출 순서는 `null_diff_quantile`과 같다(복원 추출, 반복 쌍 무작위)."""
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
        total = sum(outcomes[a][c] - outcomes[b][c] for c in sample)
        abs_means.append(abs(Fraction(total, n_cases)))
    return percentile_exact(abs_means, q)


def clopper_pearson(k: int, n: int, alpha: Fraction = CI_ALPHA) -> tuple[float, float] | None:
    """정확한 양측 구간(양쪽 각 alpha/2). 분모 0이면 None(판정 불가, §8-21)."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return None
    tail = float(alpha) / 2
    low = 0.0 if k == 0 else _bisect(lambda p: _upper_tail(k, n, p) - tail)
    high = 1.0 if k == n else _bisect(lambda p: tail - _lower_tail(k, n, p))
    return low, high


def stratified_bootstrap_indices(
    cell_sizes: Sequence[int], n_boot: int = N_BOOT, seed: int = SEED
) -> list[IndexArray]:
    """칸마다 칸 크기만큼 복원 추출한 색인. 칸 순서대로 `integers(0, 크기, (n_boot, 크기))`."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.integers(0, size, size=(n_boot, size)) for size in cell_sizes]


def stratified_subsample_indices(
    cell_sizes: Sequence[int],
    take: Sequence[int],
    n_resample: int,
    seed: int = SEED,
    *,
    replace: bool,
) -> list[list[IndexArray]]:
    """재표본 n_resample개. 재표본마다 칸 순서대로 칸에서 take개를 뽑는다(하나의 seed 흐름)."""
    if len(take) != len(cell_sizes):
        raise ValueError(f"take has {len(take)} cells, cell_sizes has {len(cell_sizes)}")
    if not replace and any(k > size for k, size in zip(take, cell_sizes, strict=True)):
        raise ValueError(f"take {list(take)} exceeds cell sizes {list(cell_sizes)}")
    rng = np.random.Generator(np.random.PCG64(seed))
    resamples = []
    for _ in range(n_resample):
        cells = []
        for size, k in zip(cell_sizes, take, strict=True):
            if replace:
                cells.append(rng.integers(0, size, size=k))
            else:
                cells.append(rng.choice(size, size=k, replace=False).astype(np.int64))
        resamples.append(cells)
    return resamples


def _upper_tail(k: int, n: int, p: float) -> float:  # P(X >= k)
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _lower_tail(k: int, n: int, p: float) -> float:  # P(X <= k)
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _bisect(increasing: Callable[[float], float]) -> float:
    """[0, 1]에서 증가 함수의 근."""
    lo, hi = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2
        if increasing(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
