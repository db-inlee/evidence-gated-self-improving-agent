"""문서 값 확장(Ticket 012): 0단계·1단계 결과(kind exploratory), 허용 이름, 표 T8."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evidencegate import writeup as w
from evidencegate import writeup_artifacts as wa
from evidencegate import writeup_render as wr

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(VALUES.read_text(encoding="utf-8"))
    return data


def test_the_012_results_are_fixed_inputs() -> None:
    assert w.FOLLOW_UP_INPUTS["t012_stage0"] == (
        Path("results/ticket-012/stage0.json"),
        "9bbe3ba4ccbe572065452865ac952b40fc25b196b3da854cb6ae5036c6f34739",
    )
    assert w.FOLLOW_UP_INPUTS["t012_stage1"] == (
        Path("results/ticket-012/stage1.json"),
        "f36fdbccbd45d2055eecc1f1273f149251362094bbc01f9b97cb07a45fd497a9",
    )
    for path, digest in w.FOLLOW_UP_INPUTS.values():
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest


def test_the_012_values_are_exploratory(values: dict[str, Any]) -> None:
    keys = [k for k in values if k.startswith(("t012.stage0.", "t012.stage1."))]
    assert keys and {values[k]["kind"] for k in keys} == {"exploratory"}
    assert values["t012.stage1.decision.verdict"]["display"] == "reconsider"
    assert values["t012.stage1.rule.k"]["display"] == "21"
    assert values["t012.stage0.verdict"]["display"] == "pass"
    assert values["t012.stage1.baseline.traffic_episodes"]["display"] == "22"
    for name in ("t012_stage0", "t012_stage1"):
        assert values[f"hash.{name}"]["kind"] == "provenance"
        assert values[f"hash.{name}"]["display"] == w.FOLLOW_UP_INPUTS[name][1]


def test_ticket_012_and_t8_are_allowed_names(values: dict[str, Any]) -> None:
    for text in ("Ticket 012 closed path A.", "Table T8 lists the second probe."):
        assert wr.check_source(f"## Design\n\n{text}\n", "md", values) == [], text
        assert not re.search(r"\d", wa.ALLOWED_NAME.sub("", text)), text
    for text in ("Ticket 013 is next.", "Table T9 does not exist."):
        found = wr.check_source(f"## Design\n\n{text}\n", "md", values)
        assert {v.rule for v in found} == {"number"}, text


@pytest.fixture(scope="module")
def t8(values: dict[str, Any]) -> dict[str, Any]:
    return wa.build_tables(values)["t8"]


def keys(table: dict[str, Any]) -> list[str]:
    texts = [table["title"], table["note"], *table["columns"]]
    texts += [c for r in table["rows"] for c in r]
    return [k for t in texts for k in wa.PLACEHOLDER.findall(t)]


def test_t8_names_its_baseline_column_and_has_a_baseline_row(
    t8: dict[str, Any], values: dict[str, Any]
) -> None:
    t7 = wa.build_tables(values)["t7"]
    assert t7["columns"][2] == "Dispositions differing from S0"
    assert t8["columns"][2] == "Dispositions differing from the S0\u2032 baseline run"
    assert t8["columns"][:2] + t8["columns"][3:] == t7["columns"][:2] + t7["columns"][3:]
    names = [wa.fill(r[0], values) for r in t8["rows"]]
    assert names == ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "S0\u2032 (baseline run)"]
    found = set(keys(t8))
    assert "t012.stage1.baseline.traffic_episodes" in found
    assert "t012.stage1.baseline.mean_tool_calls" in found
    assert "t012.stage1.decision.verdict" in found and "t012.stage1.rule.k" in found
    assert t8["rows"][0][3] == "n/a (wiring check)"  # P0는 판정 집계 밖


def test_t8_holds_only_exploratory_values_and_no_bare_numbers(
    t8: dict[str, Any], values: dict[str, Any]
) -> None:
    assert {values[k]["kind"] for k in keys(t8)} == {"exploratory"}
    texts = [t8["title"], t8["note"], *t8["columns"], *(c for r in t8["rows"] for c in r)]
    for text in texts:
        assert not re.search(r"\d", wa.ALLOWED_NAME.sub("", wa.PLACEHOLDER.sub("", text))), text
    bad = "## Results\n\n{{table:t8}}\n"
    assert {v.rule for v in wr.check_source(bad, "md", values)} == {"placement"}


def test_a_changed_012_result_is_refused(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "root"
    for rel in (
        *(p for p, _ in w.FOLLOW_UP_INPUTS.values()),
        w.T009B_SIDECAR,
        w.RUN_SUMMARY_009B_SIDECAR,
        w.AUDIT_013_SIDECAR,
        w.PREREG_009B_PATH,
        w.PREREG_009_PATH,
        w.PREREG_009_SHA_PATH,
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, root / rel)
    path = root / "results/ticket-012/stage1.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sha256"):
        w.load_follow_up(root)


def test_the_012_criteria_give_the_base_rate_and_k_rule(values: dict[str, Any]) -> None:
    assert w.FOLLOW_UP_INPUTS["t012_criteria"] == (
        Path("plans/ticket-012-criteria.json"),
        "63a8f73ee8bac878c51007105036aa1bdeaed5d1f8a1678ce3893611efc5fe94",
    )
    expected = {
        "t012.criteria.base_rate.cd_successes_min": "2",
        "t012.criteria.base_rate.cd_successes_max": "8",
        "t012.criteria.base_rate.per_cell": "10",
        "t012.criteria.base_rate.cells": "6",
        "t012.criteria.base_rate.cells_needed": "4",
        "t012.criteria.k_rule.tail_below": "1/100",
    }
    for key, display in expected.items():
        assert values[key] == {**values[key], "display": display, "kind": "exploratory"}, key
    # 문서에 쓰는 절만 편다(금지 단어 목록 같은 나머지는 값으로 내지 않는다)
    assert not [k for k in values if k.startswith("t012.criteria.order_ban")]
    assert values["hash.t012_criteria"]["display"] == w.FOLLOW_UP_INPUTS["t012_criteria"][1]
