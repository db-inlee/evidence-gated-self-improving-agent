"""Ticket 009 정의(§2.2 h, §2.3 Severe Local Harm, §2.4 RQ9-1·2·3)의 정확한 구현."""

import random
from fractions import Fraction
from math import comb

import pytest

from evidencegate.eval_data import CELLS, PairedTable
from evidencegate.exact import percentile_exact, stratified_bootstrap_indices
from evidencegate.t009.rq9 import (
    cell_condition,
    cell_threshold_h,
    fisher_one_sided,
    rq9_1,
    rq9_2,
    rq9_3,
    severe_local_harm,
)


def calibration(n_per_cell: int, seed: int = 1) -> dict[int, dict[tuple[str, str, int], int]]:
    rng = random.Random(seed)
    cases = [(f, lv, 900_000 + i) for (f, lv) in CELLS for i in range(n_per_cell)]
    return {r: {c: int(rng.random() < 0.7) for c in cases} for r in range(5)}


def reference_h(outcomes, n_c: int, n_boot: int) -> tuple[Fraction, Fraction]:  # type: ignore[no-untyped-def]
    """§2.2 문장을 그대로 따른 느린 구현."""
    repeats = sorted(outcomes)
    by_cell = {cell: sorted(c for c in outcomes[0] if c[:2] == cell) for cell in CELLS}
    rng = random.Random(0)
    ms = []
    for _ in range(n_boot):
        r, s = rng.sample(repeats, 2)
        cell_deltas = []
        for cell in CELLS:
            sample = rng.choices(by_cell[cell], k=n_c)
            cell_deltas.append(Fraction(sum(outcomes[r][c] - outcomes[s][c] for c in sample), n_c))
        ms.append(min(cell_deltas))
    q = percentile_exact(ms, Fraction(1, 20))
    return -q, Fraction(sum(1 for m in ms if m < q), n_boot)


@pytest.mark.parametrize("n_c", [35, 4, 70])
def test_h_follows_the_preregistered_procedure(n_c: int) -> None:
    out = calibration(35)
    h, achieved = cell_threshold_h(out, n_c, n_boot=300)
    assert (h, achieved) == reference_h(out, n_c, 300)
    assert isinstance(h, Fraction) and h > 0


def test_h_is_reproducible_and_seeded_fresh_per_call() -> None:
    out = calibration(35)
    assert cell_threshold_h(out, 35, n_boot=200) == cell_threshold_h(out, 35, n_boot=200)


def test_h_needs_five_repeats_and_every_cell() -> None:
    out = calibration(35)
    with pytest.raises(ValueError):
        cell_threshold_h({r: out[r] for r in range(4)}, 35, n_boot=10)
    missing = {r: {c: v for c, v in out[r].items() if c[:2] != CELLS[0]} for r in range(5)}
    with pytest.raises(ValueError):
        cell_threshold_h(missing, 35, n_boot=10)


def test_cell_condition_is_exact_and_non_strict() -> None:
    h = Fraction(1, 5)
    assert cell_condition([Fraction(-1, 5)] * 6, h)  # 경계는 통과
    assert not cell_condition([Fraction(0)] * 5 + [Fraction(-7, 35)], Fraction(6, 35))


def table(deltas_by_cell: list[list[int]]) -> PairedTable:
    cases, cells, cand, base = [], [], [], []
    for (f, lv), ds in zip(CELLS, deltas_by_cell, strict=True):
        for i, d in enumerate(ds):
            cases.append((f, lv, 700_000 + i))
            cells.append((f, lv))
            cand.append(1 if d > 0 else 0)
            base.append(1 if d < 0 else 0)
    return PairedTable(
        "oracle", "correct_disposition", tuple(cases), tuple(cells), tuple(cand), tuple(base)
    )


TAU = dict.fromkeys(CELLS, Fraction(1, 20))


def test_severe_needs_the_cell_upper_bound_below_minus_tau() -> None:
    clean = [[0] * 50 for _ in CELLS]
    assert severe_local_harm(table(clean), TAU) is False
    broken = [list(c) for c in clean]
    broken[2] = [-1] * 30 + [0] * 20  # 칸 하락 -0.6: 상한도 -τ 아래
    assert severe_local_harm(table(broken), TAU) is True
    small = [list(c) for c in clean]
    small[2] = [-1] * 4 + [0] * 46  # 점추정 -0.08 < -τ지만 구간 상한은 -τ 위
    assert severe_local_harm(table(small), TAU) is False  # 불확정 범주 없이 거짓


def test_severe_uses_the_cell_block_of_the_stratified_indices() -> None:
    deltas = [[(-1 if (i * 7 + k) % 5 == 0 else 0) for i in range(50)] for k in range(6)]
    t = table(deltas)
    blocks = stratified_bootstrap_indices([50] * 6, 2000, 0)
    expected = False
    for k, cell in enumerate(CELLS):
        import numpy as np

        totals = sorted(int(x) for x in np.asarray(deltas[k])[blocks[k]].sum(axis=1))
        upper = percentile_exact([Fraction(x) for x in totals], Fraction(39, 40)) / 50
        expected = expected or upper < -TAU[cell]
    assert severe_local_harm(t, TAU) is expected


def test_fisher_one_sided_is_the_hypergeometric_upper_tail() -> None:
    # n 15, Severe 8, 걸러짐 4가 모두 Severe: p = C(8,4)/C(15,4)
    assert fisher_one_sided(a=4, f=4, k=8, n=15) == Fraction(comb(8, 4), comb(15, 4))
    total = sum(
        fisher_one_sided(a=x, f=5, k=6, n=12) - fisher_one_sided(a=x + 1, f=5, k=6, n=12)
        for x in range(0, 5)
    ) + fisher_one_sided(a=5, f=5, k=6, n=12)
    assert total == 1


@pytest.mark.parametrize(
    ("rows", "verdict"),
    [
        ([(False, True)] * 5 + [(True, False)] * 10, "supported"),
        ([(False, True)] * 4 + [(True, True)] * 4 + [(True, False)] * 7, "direction_only"),
        ([(False, False)] * 3 + [(True, True)] * 3, "not_supported"),
        ([(True, True)], "undecidable"),  # n < 2
        ([(True, True), (True, False)], "undecidable"),  # f = 0
        ([(False, True), (False, False)], "undecidable"),  # f = n
        ([(False, False), (True, False)], "undecidable"),  # k = 0
        ([(False, True), (True, True)], "undecidable"),  # k = n
    ],
)
def test_rq9_1_verdicts(rows: list[tuple[bool, bool]], verdict: str) -> None:
    # 행 하나 = C4 채택 후보 (C5 채택, Severe)
    assert rq9_1(rows)["verdict"] == verdict


def test_rq9_1_reports_the_table_and_p() -> None:
    r = rq9_1([(False, True)] * 5 + [(True, False)] * 10)
    assert (r["a"], r["f"], r["k"], r["n"]) == (5, 5, 5, 15)
    assert r["p"] == Fraction(1, comb(15, 5))


@pytest.mark.parametrize(
    ("useful_c4", "useful_rejected", "any_filtered", "verdict"),
    [
        (10, 0, True, "supported"),
        (10, 2, True, "supported"),  # 1/5 경계는 지지
        (10, 3, True, "not_supported"),
        (0, 0, True, "undecidable"),
        (10, 0, False, "undecidable"),  # C5 = C4
    ],
)
def test_rq9_2(useful_c4: int, useful_rejected: int, any_filtered: bool, verdict: str) -> None:
    assert rq9_2(useful_c4, useful_rejected, any_filtered=any_filtered)["verdict"] == verdict


def test_rq9_3_uses_the_clopper_pearson_lower_bound() -> None:
    assert rq9_3(severe_accepted=6, accepted=15)["verdict"] == "supported"
    assert rq9_3(severe_accepted=1, accepted=15)["verdict"] == "not_supported"
    assert rq9_3(severe_accepted=0, accepted=0)["verdict"] == "undecidable"


def test_rq9_1_p_exactly_one_twentieth_is_not_supported_but_direction_only() -> None:
    rows = [(False, True)] * 3 + [(True, False)] * 3  # n 6, k 3, f 3, a 3: p = 1/C(6,3)
    r = rq9_1(rows)
    assert r["p"] == Fraction(1, 20) and r["verdict"] == "direction_only"
