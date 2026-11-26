"""Ticket 013 4단계: 009b 탐색 재생(`evidencegate.t013.replay`). 모든 값은 Exploratory다."""

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import pytest

from evidencegate.eval_data import CELLS
from evidencegate.t013 import replay as rp
from evidencegate.t013 import sim
from evidencegate.t013 import sim_core as sc

REPO_ROOT = Path(__file__).resolve().parents[2]


def outcomes(values: dict[tuple[str, str], list[int]]) -> dict[tuple[str, str, int], int]:
    return {(f, lv, i): v for (f, lv), vs in values.items() for i, v in enumerate(vs)}


def test_cell_counts_give_wins_losses_and_s0_successes_per_cell() -> None:
    cell = CELLS[0]
    cand = outcomes({cell: [1, 0, 1, 0]})
    s0 = outcomes({cell: [0, 1, 1, 0]})
    got = rp.cell_counts(cand, s0)
    assert got[0] == {"w": 1, "l": 1, "n_s": 2, "n": 4}
    with pytest.raises(ValueError, match="cases"):
        rp.cell_counts(cand, outcomes({cell: [0, 1, 1]}))


def test_replay_gates_use_the_frozen_parameters() -> None:
    counts = [{"w": 8, "l": 1, "n_s": 60, "n": 70}] + [{"w": 0, "l": 0, "n_s": 70, "n": 70}] * 5
    got = rp.replay_gates(counts)
    big_w, big_l, big_s = 8, 1, 60 + 5 * 70
    assert got["SAGE_l1"] == sim.sage_accept(
        big_w, big_l, big_s, lam=1, tau=sc.SAGE_TAU, alpha=sc.ALPHA
    )
    assert got["SAGE_l2"] == sim.sage_accept(
        big_w, big_l, big_s, lam=2, tau=sc.SAGE_TAU, alpha=sc.ALPHA
    )
    cells_ok = sim.sage_cells_ok([1, 0, 0, 0, 0, 0], [60] + [70] * 5, tau=sc.SAGE_TAU)
    assert got["SAGE_cell_l1"] == (got["SAGE_l1"] and cells_ok)
    assert got["credit"] == sim.credit_accept(big_w, big_l, 420, big_s, eps=sc.CREDIT_EPS)
    assert set(got) == set(rp.REPLAY_GATES)


def test_tally_counts_accepted_severe_useful_and_clean_useful() -> None:
    labels: dict[str, dict[str, Any]] = {
        "x": {"severe": True, "useful": True, "clean_useful": False},
        "y": {"severe": False, "useful": None, "clean_useful": False},
        "z": {"severe": False, "useful": True, "clean_useful": True},
    }
    accepted = {"x": True, "y": True, "z": False}
    got = rp.tally(accepted, labels, ["x", "y", "z"])
    assert got == {"candidates": 3, "accepted": 2, "severe": 1, "useful": 1, "clean_useful": 0,
                   "accepted_names": ["x", "y"]}  # fmt: skip
    assert rp.tally(accepted, labels, ["z"])["accepted"] == 0


def test_a_recomputed_c4_or_c5_that_differs_from_the_record_stops_the_replay() -> None:
    rp.check_recorded({"a": {"C4": True, "C5": False}}, {"a": {"C4": True, "C5": False}})
    with pytest.raises(ValueError, match="C5"):
        rp.check_recorded({"a": {"C4": True, "C5": True}}, {"a": {"C4": True, "C5": False}})


def test_parameters_are_the_frozen_ones() -> None:
    plan = sim.load_plan(REPO_ROOT)
    params = rp.parameters()
    assert params["SAGE"] == {"lambda": plan["gates"]["SAGE"]["lambda"], "tau": 0.05, "alpha": 0.05}
    assert params["credit_type"] == {"epsilon": plan["gates"]["credit_type"]["epsilon"]}
    assert (
        params["plan_sha256"] == sim.PLAN_SHA256
        and params["addendum_sha256"] == sim.ADDENDUM_SHA256
    )
    assert F(1, 20) == sc.ALPHA


def test_committed_replay_matches_its_sidecar_and_is_exploratory() -> None:
    path = REPO_ROOT / rp.OUT
    recorded = path.with_name(path.name + ".sha256").read_text().split()
    assert recorded == [hashlib.sha256(path.read_bytes()).hexdigest(), path.name]
    data = json.loads(path.read_text())
    assert data["status"] == "Exploratory"
    assert set(data["pools"]) == {"all", "representatives"}
    assert set(data["pools"]["all"]) == {"C4", "C5", *rp.REPLAY_GATES}
    assert (
        data["pools"]["all"]["C4"]["accepted"] == 22
        and data["pools"]["all"]["C5"]["accepted"] == 13
    )
    assert data["consistency"]["c4_c5_recomputed_equal_record"] is True


def test_the_cell_cap_can_reject_what_sage_accepts() -> None:
    # 전체 rho = 5 / 410 <= 0.05지만 첫 칸 rho_c = 5 / 60 > 0.05
    counts = [{"w": 40, "l": 5, "n_s": 60, "n": 70}] + [{"w": 0, "l": 0, "n_s": 70, "n": 70}] * 5
    got = rp.replay_gates(counts)
    assert got["SAGE_l1"] is True and got["SAGE_cell_l1"] is False


# C6-IUT 탐색 재생([2-prime]) ------------------------------------------------------------


def test_c6_needs_c4_and_every_cell_bound_above_minus_m() -> None:
    from fractions import Fraction

    quiet = [{"w": 0, "l": 0, "n_s": 70, "n": 70}] * 6
    m = Fraction(1, 20)
    assert rp.c6_accept(quiet, c4=True, m=m, bound="tango") is True  # 불일치 0: 하한 -0.037
    assert rp.c6_accept(quiet, c4=False, m=m, bound="tango") is False
    lossy = [{"w": 0, "l": 4, "n_s": 70, "n": 70}, *quiet[1:]]
    assert sim.tango_lower(0, 4, 70, sc.ALPHA) < -float(m)
    assert rp.c6_accept(lossy, c4=True, m=m, bound="tango") is False
    assert rp.c6_accept(lossy, c4=True, m=Fraction(1, 5), bound="tango") is True
    exact = sim.exact_conditional_lower(0, 4, 70, sc.ALPHA)
    assert rp.c6_accept(lossy, c4=True, m=m, bound="exact_conditional") is (exact > -float(m))


def test_committed_c6_replay_is_exploratory_and_says_it_changes_nothing() -> None:
    path = REPO_ROOT / rp.OUT_C6
    recorded = path.with_name(path.name + ".sha256").read_text().split()
    assert recorded == [hashlib.sha256(path.read_bytes()).hexdigest(), path.name]
    data = json.loads(path.read_text())
    assert data["status"] == "Exploratory"
    assert "does not change m_c or n_c" in data["header"]
    assert data["margins"] == ["1/20", "1/10"] and data["primary_bound"] == "tango"
    assert data["pools"]["all"]["C4"]["accepted"] == 22
