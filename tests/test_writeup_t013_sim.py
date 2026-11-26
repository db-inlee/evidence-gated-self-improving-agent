"""문서 값 확장(Ticket 013 마무리 [3]): 모의·보정·재생 결과(`t013.`, exploratory)와 모의 문서."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evidencegate import writeup as w
from evidencegate import writeup_artifacts as wa
from evidencegate import writeup_render as wr

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"
RESULTS = {
    "t013_sim": "results/ticket-013/sim.json",
    "t013_gain": "results/ticket-013/sim-gain-correction.json",
    "t013_replay": "results/ticket-013/replay-009b.json",
    "t013_replay_c6": "results/ticket-013/replay-009b-c6.json",
}


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(VALUES.read_text(encoding="utf-8"))
    return data


def test_the_013_results_are_fixed_inputs_with_their_sidecars() -> None:
    for name, rel in RESULTS.items():
        path, digest = w.FOLLOW_UP_INPUTS[name]
        assert path == Path(rel)
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest
        sidecar = (REPO_ROOT / (rel + ".sha256")).read_text().split()[0]
        assert sidecar == digest


def test_013_values_are_exploratory_and_read_from_the_results(values: dict[str, Any]) -> None:
    keys = [k for k in values if k.startswith("t013.")]
    assert keys and {values[k]["kind"] for k in keys} == {"exploratory"}
    shown = {
        "t013.dec.a.C6_IUT.n150.m05.false_accept": "0.0442",
        "t013.dec.a.C6_IUT.n150.m05.status": "Go",
        "t013.dec.b.C6_IUT.n70.m05.useful_power": "0.7525",
        "t013.dec.a.C5.n300.m10.status": "Go",
        "t013.go.a": "4",
        "t013.go.b": "2",
        "t013.go.any": "yes",
        "t013.replay.all.C5.severe": "2",
        "t013.replay.all.SAGE_l1.severe": "1",
        "t013.c6.all.tango_m05.accepted": "1",
        "t013.c6.all.exact_m05.severe": "1",
        "t013.gate.reproduced": "yes",
    }
    for key, display in shown.items():
        assert values[key]["display"] == display, key


def test_gain_values_come_only_from_the_correction_file(values: dict[str, Any]) -> None:
    gain_keys = [k for k in values if k.startswith("t013.") and "gain" in k]
    assert gain_keys and all(k.startswith("t013.gain.") for k in gain_keys)
    gain = json.loads((REPO_ROOT / RESULTS["t013_gain"]).read_text())
    want = gain["gain_per_proposal"]["a_150_none"]["1/20|empirical|-|main|C6_IUT"]["estimate"]
    assert values["t013.gain.a.C6_IUT.n150.m05.empirical"]["exact"] == repr(want)


def test_c6_and_lambda_names_are_allowed(values: dict[str, Any]) -> None:
    text = "## Design\n\nC6-IUT, SAGE λ1 and SAGE λ2.\n"
    assert wr.check_source(text, "md", values) == []
    assert not wa.ALLOWED_NAME.sub("", "C6 λ1 λ2").strip()
    found = wr.check_source("## Design\n\nC7 and λ3.\n", "md", values)
    assert {v.rule for v in found} == {"number"}


def test_the_simulation_page_is_rendered() -> None:
    docs = {d.template.name: d for d in wr.DOCUMENTS}
    doc = docs["study-2-gate-simulation.template"]
    assert doc.output == Path("docs/study-2-gate-simulation.md")
    text = (REPO_ROOT / doc.output).read_text(encoding="utf-8")
    assert "sim-gain-correction.json" in text and "invalid" in text
    assert "C5 failed" not in text
