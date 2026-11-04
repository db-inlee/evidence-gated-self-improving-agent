"""Ticket 009 §2.5 검정력 점검(계획용, LLM 없음).

모의실험은 후보 여러 개를 행렬로 한 번에 판정한다. 그 결과가 실제 함수(gates.gate_c4, rq9,
oracle.useful·local_harm, exact.null_diff_quantile_exact)와 정확히 같은지 본다.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from evidencegate.eval_data import CELLS, PairedTable
from evidencegate.exact import null_diff_quantile_exact, stratified_bootstrap_indices
from evidencegate.gates import gate_c4
from evidencegate.oracle import estimate, local_harm, useful
from evidencegate.t009.power_sim import (
    Scenario,
    c4_many,
    cell_sums,
    delta_from_outcomes,
    local_harm_many,
    severe_many,
    simulate,
    summarize,
    useful_many,
)
from evidencegate.t009.rq9 import severe_local_harm

RNG = np.random.default_rng(7)


def random_deltas(n_per_cell: int, n_cands: int, bias: float = 0.0) -> np.ndarray:
    """케이스(칸 순서) x 후보, 값은 -1/0/1."""
    p = RNG.uniform(0.3, 0.9, size=(6 * n_per_cell, 1))
    s0 = RNG.random((6 * n_per_cell, n_cands)) < p
    cand = RNG.random((6 * n_per_cell, n_cands)) < np.clip(p + bias, 0, 1)
    return cand.astype(np.int64) - s0.astype(np.int64)


def as_table(d: np.ndarray, split: str, n_per_cell: int) -> PairedTable:
    cases, cells, cand, base = [], [], [], []
    for k, (f, lv) in enumerate(CELLS):
        for i in range(n_per_cell):
            v = int(d[k * n_per_cell + i])
            cases.append((f, lv, 600_000 + i))
            cells.append((f, lv))
            cand.append(1 if v > 0 else 0)
            base.append(1 if v < 0 else 0)
    return PairedTable(
        split, "correct_disposition", tuple(cases), tuple(cells), tuple(cand), tuple(base)
    )


@pytest.mark.parametrize("n_per_cell", [35, 70])
def test_c4_many_matches_gate_c4(n_per_cell: int) -> None:
    d = random_deltas(n_per_cell, 25, bias=0.04)
    delta = Fraction(1, 21)
    blocks = stratified_bootstrap_indices([n_per_cell] * 6, 2000, 0)
    got = c4_many(d, blocks, delta)
    want = [gate_c4(as_table(d[:, j], "dev_b", n_per_cell), delta).accepted for j in range(25)]
    assert list(got) == want


def test_cell_sums_are_per_cell_totals() -> None:
    d = random_deltas(35, 3)
    s = cell_sums(d, 35)
    assert s.shape == (6, 3)
    assert s[2, 1] == d[70:105, 1].sum()


def oracle_tau() -> dict[tuple[str, str], Fraction]:
    return {cell: Fraction(1, 20) if i % 2 else Fraction(73, 1000) for i, cell in enumerate(CELLS)}


def test_severe_many_matches_rq9() -> None:
    d = random_deltas(50, 25, bias=-0.06)
    tau = oracle_tau()
    blocks = stratified_bootstrap_indices([50] * 6, 2000, 0)
    got = severe_many(d, blocks, tau)
    want = [severe_local_harm(as_table(d[:, j], "oracle", 50), tau) for j in range(25)]
    assert list(got) == want and any(want) and not all(want)


def test_useful_and_local_harm_many_match_oracle() -> None:
    d = random_deltas(50, 30, bias=0.05)
    tau = oracle_tau()
    blocks = stratified_bootstrap_indices([50] * 6, 2000, 0)
    got_u = useful_many(d, blocks)
    got_lh = local_harm_many(d, 50, tau)
    for j in range(30):
        e = estimate(as_table(d[:, j], "oracle", 50), tau)
        assert got_u[j] == useful(e)
        assert got_lh[j] == local_harm(e)


CALIBRATION = Path(
    "runs/calibration/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/bf65e569ae37.jsonl"
)


@pytest.mark.skipif(not CALIBRATION.exists(), reason="needs v1 calibration records")
def test_delta_from_outcomes_matches_the_v1_function() -> None:
    from evidencegate.store import EpisodeStore

    records = EpisodeStore(CALIBRATION, repair=False).records()
    outcomes: dict[int, dict[tuple[str, int, str], int]] = {}
    for r in records:
        outcomes.setdefault(r.key.repeat, {})[(r.key.family, r.key.incident_seed, r.key.level)] = (
            int(r.correct_disposition)
        )
    for n in (24, 210):
        assert delta_from_outcomes(outcomes, n) == null_diff_quantile_exact(
            records, "correct_disposition", n
        )


EFFECTS = {
    "a": (0.0, -0.1, 0.0, 0.0, 0.3, 0.2),
    "b": (-0.4, -0.5, 0.0, 0.0, 0.8, 0.8),
    "c": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}
P_HAT = {cell: [0.2, 0.6, 0.8, 1.0, 1.0] * 10 for cell in CELLS}


def test_simulate_is_deterministic_and_covers_every_scenario() -> None:
    kw = dict(effects=EFFECTS, p_hat=P_HAT, n_pools=2, n_cands=10, n_null=5, n_boot=200, seed=3)
    a = simulate(**kw)  # type: ignore[arg-type]
    b = simulate(**kw)  # type: ignore[arg-type]
    assert a == b
    assert {(s.dev_b_per_cell, s.shrink) for s in a["scenarios"]} == {
        (35, 1.0),
        (35, 0.5),
        (70, 1.0),
        (70, 0.5),
    }
    pool = a["pools"][0]
    assert set(pool) >= {"h", "delta", "candidates", "null"}
    assert pool["h"][35] > 0 and pool["h"][70] > 0


def test_summarize_counts_verdicts_per_n() -> None:
    kw = dict(effects=EFFECTS, p_hat=P_HAT, n_pools=3, n_cands=10, n_null=5, n_boot=200, seed=3)
    sim = simulate(**kw)  # type: ignore[arg-type]
    table = summarize(sim, ns=(4, 8, 10))
    row = table[Scenario(35, 1.0)][8]
    for rq in ("rq9_1", "rq9_2", "rq9_3"):
        assert sum(row[rq].values()) == 3
    assert set(table[Scenario(70, 0.5)]) == {4, 8, 10}


def test_c4_many_matches_gate_c4_at_the_boundaries() -> None:
    delta = Fraction(1, 21)
    blocks = stratified_bootstrap_indices([35] * 6, 2000, 0)
    on_delta = np.zeros(210, dtype=np.int64)
    on_delta[::21] = 1  # 합 10, 평균이 δ와 정확히 같다
    on_zero = np.zeros(210, dtype=np.int64)  # +1 33개, -1 22개: 평균 11/210 > δ, 하한이 정확히 0
    idx = np.linspace(0, 209, 55).astype(int)
    on_zero[idx[:33]] = 1
    on_zero[idx[33:]] = -1
    d = np.stack([on_delta, on_zero], axis=1)
    got = c4_many(d, blocks, delta)
    results = [gate_c4(as_table(d[:, j], "dev_b", 35), delta) for j in range(2)]
    assert list(got) == [r.accepted for r in results] == [False, False]
    assert results[0].mean_delta == delta and results[0].lower is not None
    assert results[0].lower > 0
    assert results[1].lower == 0 and results[1].mean_delta > delta


def test_c5_many_matches_the_cell_condition_on_the_boundary() -> None:
    from evidencegate.t009.power_sim import c5_many
    from evidencegate.t009.rq9 import cell_condition

    h = Fraction(6, 35)
    d = np.zeros((210, 3), dtype=np.int64)
    d[0:6, 0] = -1  # 첫 칸 합 -6: Δ = -6/35 = -h, 통과(비엄격)
    d[0:7, 1] = -1  # -7/35 < -h, 기각
    c4 = np.array([True, True, False])
    got = c5_many(c4, d, 35, h)
    want = [
        bool(c4[j]) and cell_condition([Fraction(int(s), 35) for s in cell_sums(d, 35)[:, j]], h)
        for j in range(3)
    ]
    assert list(got) == want == [True, False, False]


def test_calibration_size_is_a_setting() -> None:
    kw = dict(effects=EFFECTS, p_hat=P_HAT, n_pools=1, n_cands=4, n_null=2, n_boot=100, seed=3)
    a = simulate(**kw)  # type: ignore[arg-type]
    b = simulate(**kw, cal_per_cell=70)  # type: ignore[arg-type]
    assert a == simulate(**kw, cal_per_cell=35)  # type: ignore[arg-type]  # 기본값 35
    assert a["pools"][0]["calibration_per_cell"] == 35
    assert b["pools"][0]["calibration_per_cell"] == 70


def fake_sim(rows: list[tuple[bool, bool, bool, bool, bool | None]]) -> dict[str, object]:
    sc = Scenario(70, 1.0)
    return {"scenarios": [sc], "pools": [{"candidates": {sc: rows}}]}


def test_clean_useful_basis_for_rq9_2() -> None:
    # (C4, C5, Severe, Local Harm, Useful)
    rows = [
        (True, False, True, True, True),  # 섞인 Useful: C5가 거름
        (True, False, True, True, True),
        (True, True, False, False, True),  # 깨끗한 Useful: 남음
        (True, False, False, False, None),  # Useful 불확정(C5가 거름): 세지 않음
        (True, True, False, False, False),
    ]
    sim = fake_sim(rows)
    default = summarize(sim, ns=(5,))[Scenario(70, 1.0)][5]
    assert default["rq9_2"]["not_supported"] == 1  # 모든 Useful: 손실 2/3(불확정은 세지 않음)
    assert "rq9_2_all_useful" not in default
    clean = summarize(sim, ns=(5,), clean_useful=True)[Scenario(70, 1.0)][5]
    assert clean["rq9_2"]["supported"] == 1  # 깨끗한 Useful: 손실 0/1
    assert clean["rq9_2_all_useful"]["not_supported"] == 1  # 기술용으로 함께
