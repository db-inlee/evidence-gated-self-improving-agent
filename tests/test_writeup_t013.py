"""문서 값 확장(Ticket 013 1단계): 감사 요약(`audit.`, kind exploratory)과 감사 문서."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evidencegate import writeup as w
from evidencegate import writeup_render as wr
from evidencegate.t013 import audit as au

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(VALUES.read_text(encoding="utf-8"))
    return data


def test_the_audit_is_a_fixed_input_with_its_sidecar() -> None:
    path, digest = w.FOLLOW_UP_INPUTS["audit_013"]
    assert path == au.DEFAULT_OUT
    assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest
    assert Path("writeup/inputs/audit_013.json.sha256") == w.AUDIT_013_SIDECAR


def test_audit_values_are_exploratory(values: dict[str, Any]) -> None:
    keys = [k for k in values if k.startswith("audit.")]
    assert keys and {values[k]["kind"] for k in keys} == {"exploratory"}
    shown = {
        "audit.oracle_asymmetry.cd.bad_deployment/L2.r0_minus_majority": "-0.060",
        "audit.oracle_asymmetry.cd.traffic_spike/L2.single_minus_majority": "0.068",
        "audit.s0_rates.edges.es.floor": "4",
        "audit.seed_sharing.t009_dev_b.shared_by_l1_l2": "210",
        "audit.labels.local_harm_only.this_cell_only.bad_deployment/L2": "9",
        "audit.t012_pairs.llm_seeds": "30",
    }
    for key, display in shown.items():
        assert values[key]["display"] == display, key
    assert values["hash.audit_013"]["kind"] == "provenance"


def test_the_audit_page_is_rendered_and_checked() -> None:
    docs = {d.template.name: d for d in wr.DOCUMENTS}
    doc = docs["study-2-cell-gate-audit.template"]
    assert doc.output == Path("docs/study-2-cell-gate-audit.md")
    assert (REPO_ROOT / doc.output).read_text(encoding="utf-8").startswith("# ")


def test_audit_values_outside_an_exploratory_heading_are_refused(values: dict[str, Any]) -> None:
    key = "audit.oracle_asymmetry.cd.bad_deployment/L2.r0_minus_majority"
    bad = "## Limitations\n\nOffset {{" + key + "}}.\n"
    assert {v.rule for v in wr.check_source(bad, "md", values)} == {"placement"}


def test_the_fixed_notation_p10_and_p01_is_an_allowed_name(values: dict[str, Any]) -> None:
    from evidencegate import writeup_artifacts as wa

    text = "## Design\n\np10 is an improvement, p01 a regression, and Δ = p10 \u2212 p01.\n"
    assert wr.check_source(text, "md", values) == []
    assert wa.ALLOWED_NAME.sub("", "p10 p01").split() == []
    assert {v.rule for v in wr.check_source("## Design\n\np11 is not a name.\n", "md", values)} == {
        "number"
    }
