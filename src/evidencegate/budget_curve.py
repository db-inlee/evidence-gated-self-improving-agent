"""Budget Curve와 RQ2 민감도 분석 (7.8 C, Ticket 005 §8-1·7·9~12).

Budget Curve:
- N마다 Dev-B에서 칸 층화 비복원 재표본 1,000개(칸당 N/6개)를 하나의 seed 0 흐름으로 만들고, 모든
  후보·metric이 같은 재표본을 쓴다(공통 난수, §8-9).
- 재표본마다 C2·C3·C4를 다시 판정한다. δ는 그 N의 δ(N)이다. C4 bootstrap은 seed 0으로 새로
  시작하므로(§8-9) 칸 크기가 같은 N 안에서는 색인이 같다. 그래서 N마다 한 번 만들어 넘긴다.
- Oracle 값은 고정이다(§8-11). 지표는 재표본마다 계산해 평균을 낸다. 분모가 0인 재표본은 평균에서
  빼고 그 수를 센다. 재표본 분포의 2.5·97.5 백분위수도 낸다(§8-10).

RQ2 민감도:
- Dev-A에서 칸마다 복원 추출해 bad 18·18, dep 3·3, traffic 9·9(합 60)를 1,000개 만든다(§8-12).
  모든 후보가 공유하고, 재표본마다 C1만 다시 판정한다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from evidencegate.eval_constants import (
    BUDGET_NS,
    BUDGET_SPREAD_QS,
    N_BOOT,
    N_RESAMPLE,
    SEED,
    SENSITIVITY_TAKE,
)
from evidencegate.eval_data import CELLS, PairedTable, cell_of, split_cases
from evidencegate.eval_metrics import CandidateScore, gate_metrics
from evidencegate.exact import (
    percentile_exact,
    stratified_bootstrap_indices,
    stratified_subsample_indices,
)
from evidencegate.gates import gate_c1, gate_c2, gate_c3, gate_c4
from evidencegate.metrics import METRICS
from evidencegate.oracle import OracleEstimate

BUDGET_GATES = ("C2", "C3", "C4")
METRIC_NAMES = ("harmful", "unsupported", "useful", "gain", "false_rejection", "local_harm")
SENSITIVITY_RATES = ("harmful", "local_harm")

Decisions = dict[str, dict[tuple[str, str], bool]]  # 후보 → (게이트, metric) → 채택


@dataclass(frozen=True)
class BudgetCandidate:
    name: str
    dev_a: Mapping[str, PairedTable]  # metric → Dev-A 짝 표
    dev_b: Mapping[str, PairedTable]  # metric → Dev-B 짝 표
    oracle: Mapping[str, OracleEstimate]  # metric → Oracle 추정(고정)
    expected_effect: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Summary:
    mean: Fraction | None  # 정의된 재표본의 평균
    low: Fraction | None  # 2.5 백분위수
    high: Fraction | None  # 97.5 백분위수
    undefined: int  # 분모 0이라 뺀 재표본 수
    n_resample: int


@dataclass(frozen=True)
class BudgetPoint:
    gate: str
    metric: str
    acceptance: Mapping[str, Fraction]  # 후보 → 채택된 재표본 비율
    acceptance_rate: Summary  # 재표본마다 채택 후보 비율
    metrics: Mapping[str, Summary]  # 지표 이름 → 요약


def resample_positions(n: int, n_resample: int = N_RESAMPLE, seed: int = SEED) -> list[list[int]]:
    """Dev-B 표(케이스 정렬 순서)에서 재표본마다 고른 위치. 칸 순서대로, 칸 안에서 비복원."""
    if n % len(CELLS):
        raise ValueError(f"N must split evenly into the {len(CELLS)} cells, got {n}")
    groups = _cell_groups("dev_b")
    take = [n // len(CELLS)] * len(CELLS)
    picks = stratified_subsample_indices(
        [len(g) for g in groups], take, n_resample, seed, replace=False
    )
    return [[groups[c][int(i)] for c, cell in enumerate(cells) for i in cell] for cells in picks]


def sensitivity_positions(n_resample: int = N_RESAMPLE, seed: int = SEED) -> list[list[int]]:
    """Dev-A 표에서 칸마다 복원 추출한 위치(칸 구성 SENSITIVITY_TAKE)."""
    groups = _cell_groups("dev_a")
    picks = stratified_subsample_indices(
        [len(g) for g in groups], SENSITIVITY_TAKE, n_resample, seed, replace=True
    )
    return [[groups[c][int(i)] for c, cell in enumerate(cells) for i in cell] for cells in picks]


def subtable(table: PairedTable, positions: Sequence[int]) -> PairedTable:
    if table.cases != tuple(split_cases(table.split)):
        raise ValueError("subtable needs a table in the split's case order")
    return PairedTable(
        split=table.split,
        metric=table.metric,
        cases=tuple(table.cases[p] for p in positions),
        cells=tuple(table.cells[p] for p in positions),
        candidate=tuple(table.candidate[p] for p in positions),
        baseline=tuple(table.baseline[p] for p in positions),
    )


def budget_decisions(
    candidates: Sequence[BudgetCandidate],
    deltas: Mapping[tuple[str, int], Fraction],
    ns: Sequence[int] = BUDGET_NS,
    n_resample: int = N_RESAMPLE,
    seed: int = SEED,
) -> dict[int, list[Decisions]]:
    """N → 재표본마다 후보별 C2·C3·C4 판정(CD·ES). deltas는 (metric, N) → δ(N)."""
    out: dict[int, list[Decisions]] = {}
    for n in ns:
        blocks = stratified_bootstrap_indices([n // len(CELLS)] * len(CELLS), N_BOOT, SEED)
        per_n = []
        for positions in resample_positions(n, n_resample, seed):
            decisions: Decisions = {}
            for c in candidates:
                d: dict[tuple[str, str], bool] = {}
                for m in METRICS:
                    t = subtable(c.dev_b[m], positions)
                    delta = deltas[(m, n)]
                    d[("C2", m)] = gate_c2(t).accepted
                    d[("C3", m)] = gate_c3(t, delta).accepted
                    d[("C4", m)] = gate_c4(t, delta, blocks=blocks).accepted
                decisions[c.name] = d
            per_n.append(decisions)
        out[n] = per_n
    return out


def budget_points(
    decisions: Sequence[Decisions], candidates: Sequence[BudgetCandidate]
) -> dict[tuple[str, str], BudgetPoint]:
    """한 N의 재표본 판정과 한 후보 집합에서 (게이트, metric)마다 채택률과 지표 요약."""
    points = {}
    for gate in BUDGET_GATES:
        for m in METRICS:
            values: dict[str, list[Fraction | None]] = {name: [] for name in METRIC_NAMES}
            rates: list[Fraction | None] = []
            for per_resample in decisions:
                scores = [_score(c, per_resample) for c in candidates]
                gm = gate_metrics(scores, gate, m)
                values["harmful"].append(gm.harmful.value)
                values["unsupported"].append(gm.unsupported.value)
                values["useful"].append(gm.useful.value)
                values["gain"].append(gm.gain.mean)
                values["false_rejection"].append(gm.false_rejection.value)
                values["local_harm"].append(gm.local_harm.value)
                rates.append(Fraction(len(gm.accepted), len(candidates)))
            acceptance = {
                c.name: Fraction(sum(d[c.name][(gate, m)] for d in decisions), len(decisions))
                for c in candidates
            }
            points[(gate, m)] = BudgetPoint(
                gate=gate,
                metric=m,
                acceptance=acceptance,
                acceptance_rate=summarize_values(rates),
                metrics={name: summarize_values(v) for name, v in values.items()},
            )
    return points


def sensitivity_decisions(
    candidates: Sequence[BudgetCandidate], n_resample: int = N_RESAMPLE, seed: int = SEED
) -> list[Decisions]:
    out = []
    for positions in sensitivity_positions(n_resample, seed):
        out.append(
            {
                c.name: {
                    ("C1", m): gate_c1(subtable(c.dev_a[m], positions)).accepted for m in METRICS
                }
                for c in candidates
            }
        )
    return out


def sensitivity_rates(
    decisions: Sequence[Decisions], candidates: Sequence[BudgetCandidate], metric: str
) -> dict[str, Summary]:
    """민감도 재표본마다 C1의 Harmful·Local Harm 비율, 그 요약(RQ2 둘째 문장의 입력)."""
    values: dict[str, list[Fraction | None]] = {name: [] for name in SENSITIVITY_RATES}
    for per_resample in decisions:
        gm = gate_metrics([_score(c, per_resample) for c in candidates], "C1", metric)
        values["harmful"].append(gm.harmful.value)
        values["local_harm"].append(gm.local_harm.value)
    return {name: summarize_values(v) for name, v in values.items()}


def summarize_values(values: Sequence[Fraction | None]) -> Summary:
    defined = [v for v in values if v is not None]
    if not defined:
        return Summary(None, None, None, len(values), len(values))
    low, high = (percentile_exact(defined, q) for q in BUDGET_SPREAD_QS)
    return Summary(
        mean=sum(defined, Fraction(0)) / len(defined),
        low=low,
        high=high,
        undefined=len(values) - len(defined),
        n_resample=len(values),
    )


def _score(c: BudgetCandidate, decisions: Decisions) -> CandidateScore:
    return CandidateScore(c.name, decisions[c.name], c.oracle, c.expected_effect)


def _cell_groups(split: str) -> list[list[int]]:
    cases = split_cases(split)
    return [[i for i, case in enumerate(cases) if cell_of(case) == cell] for cell in CELLS]
