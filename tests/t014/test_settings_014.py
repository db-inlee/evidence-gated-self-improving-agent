"""Ticket 014 고정 값이 사전 등록·동결 결과와 같은지(사전 등록 2·4·7.2절). LLM 없음."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from evidencegate.t013.sim import SIM_SHA256
from evidencegate.t014 import settings as s

ROOT = Path(__file__).resolve().parents[2]


def test_the_prediction_is_the_frozen_simulation_power() -> None:
    path = ROOT / "results/ticket-013/sim.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SIM_SHA256
    rows = json.loads(path.read_text())["decision"]["a"]
    (row,) = [r for r in rows if r["gate"] == "C6_IUT" and r["n_c"] == 150 and r["m_c"] == "1/20"]
    power = row["useful_power"]
    assert (power["events"], power["eligible"]) == (s.SIM_POWER_HITS, s.SIM_POWER_POOLS)
    assert Fraction(20_648, 80_000) == s.PREDICTED_LOSS


def test_gate_sizes_and_margins() -> None:
    assert (s.N_C, s.N_DEV_B) == (150, 900)
    assert s.MARGIN == s.C6_ALPHA == s.SAGE_ALPHA == s.SAGE_TAU == s.USEFUL_EPS == Fraction(1, 20)
    assert s.SAGE_LAMBDA == 1 and s.ORACLE_REPEATS == (0, 1, 2)


def test_fixed_run_options_add_only_the_exclude_dir() -> None:
    options = s.fixed_run_options(Path("/r"))
    assert options == {
        "previous": "valid",
        "dedupe": "word",
        "dedupe_threshold": 0.55,
        "stop_after_idle": 25,
        "exclude_dir": "/r/skills/candidates-009b",
    }
