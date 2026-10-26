"""값 층 (Ticket 007 §1·§2): 세 출처 → writeup/generated/values.json, 표시 형식, E1.

결과 파일 두 개와 E1 정의 파일의 sha256이 고정값과 다르면 멈춘다. 결과 파일은 읽기만 한다.
"""

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate import eval_constants as k
from evidencegate.writeup import (
    KINDS,
    build_values,
    check_inputs,
    display,
    e1_rows,
    e1_summary,
    flatten_result,
    main,
    values_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
T005 = REPO_ROOT / "results" / "ticket-005" / "analysis.json"
T006 = REPO_ROOT / "results" / "ticket-006" / "analysis.json"
E1_DEF = REPO_ROOT / "writeup" / "inputs" / "e1_definition.txt"
RUN = REPO_ROOT / "writeup" / "inputs" / "run_summary.json"
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"
CD, ES = "correct_disposition", "evidence_safe"


def frac(x: Fraction) -> dict[str, Any]:
    return {"exact": str(x), "float": float(x)}


# ---------------------------------------------------------------------------
# 입력 확인
# ---------------------------------------------------------------------------


def test_fixed_hashes() -> None:
    assert k.TICKET006_ANALYSIS_SHA256.startswith("1de4e7d3")
    assert k.E1_DEFINITION_SHA256.startswith("ef70f01a")
    assert hashlib.sha256(T006.read_bytes()).hexdigest() == k.TICKET006_ANALYSIS_SHA256
    assert hashlib.sha256(E1_DEF.read_bytes()).hexdigest() == k.E1_DEFINITION_SHA256


@pytest.mark.parametrize("which", ["t005", "t006", "e1"])
def test_a_changed_input_stops(tmp_path: Path, which: str) -> None:
    paths = {"t005": T005, "t006": T006, "e1": E1_DEF}
    changed = tmp_path / "x"
    changed.write_bytes(paths[which].read_bytes() + b" ")
    paths[which] = changed
    with pytest.raises(ValueError, match="sha256"):
        check_inputs(paths["t005"], paths["t006"], paths["e1"])


def test_ticket_006_l12_metrics_must_equal_ticket_005() -> None:
    t005 = json.loads(T005.read_text())
    t006 = json.loads(T006.read_text())
    broken = copy.deepcopy(t006)
    broken["metrics"]["proposer"][CD]["C2"]["l12"]["harmful"]["successes"] += 1
    run = json.loads(RUN.read_text())
    with pytest.raises(ValueError, match="Ticket 006 l12"):
        build_values(t005, broken, run)


# ---------------------------------------------------------------------------
# 표시 형식(한 곳)
# ---------------------------------------------------------------------------


def test_display_rules() -> None:
    assert display(frac(Fraction(-21, 100))) == ("-21/100", "-0.210")
    assert display(frac(Fraction(1, 3))) == ("1/3", "0.333")
    assert display(Fraction(1, 8)) == ("1/8", "0.125")
    # 반올림은 정확한 유리수에서 half-up(float 오차로 0.0925가 0.092가 되지 않게)
    assert display(Fraction(37, 400)) == ("37/400", "0.093")
    assert display(Fraction(-37, 400)) == ("-37/400", "-0.093")
    assert display(frac(Fraction(37, 400)))[1] == "0.093"
    assert display(0.0925) == ("0.0925", "0.093")
    assert display(5) == ("5", "5")
    assert display(True) == ("true", "yes") and display(False) == ("false", "no")
    assert display(None) == ("null", "n/a")
    assert display(0.0123456) == ("0.0123456", "0.012")
    assert display("partially_supported") == ("partially_supported", "partially supported")
    assert display("not_confirmed") == ("not_confirmed", "manipulation not confirmed")
    assert display("preserved") == ("preserved", "preserved")
    assert display("cand-015") == ("cand-015", "cand-015")


def test_flatten_rates_intervals_and_metric_names() -> None:
    result = {
        "metrics": {
            "proposer": {
                CD: {
                    "C4": {
                        "harmful": {
                            "successes": 1,
                            "n": 8,
                            "excluded": 4,
                            "value": frac(Fraction(1, 8)),
                            "interval": [0.00316, 0.52651],
                        }
                    }
                }
            }
        },
        "candidates": {
            "cand-015": {
                "oracle": {CD: {"interval": [frac(Fraction(-6, 25)), frac(Fraction(-1, 5))]}}
            }
        },
        "rq": {"RQ1": {"verdict": "not_supported"}},
    }
    flat = flatten_result("t006", result)
    rate = flat["t006.metrics.proposer.cd.C4.harmful"]
    assert rate == {"exact": "1/8", "display": "1/8", "kind": "result"}
    assert flat["t006.metrics.proposer.cd.C4.harmful.excluded"]["display"] == "4"
    assert flat["t006.metrics.proposer.cd.C4.harmful.interval"]["display"] == "[0.003, 0.527]"
    assert flat["t006.metrics.proposer.cd.C4.harmful.value"]["display"] == "0.125"
    assert flat["t006.candidates.cand-015.oracle.cd.interval"]["display"] == "[-0.240, -0.200]"
    assert flat["t006.candidates.cand-015.oracle.cd.interval.0"]["exact"] == "-6/25"
    assert flat["t006.rq.RQ1.verdict"]["display"] == "not supported"
    assert all(v["kind"] == "result" for v in flat.values())


def test_an_undefined_rate_displays_as_undefined() -> None:
    flat = flatten_result(
        "t005",
        {"r": {"successes": 0, "n": 0, "excluded": 2, "value": None, "interval": None}},
    )
    assert flat["t005.r"]["display"] == "0/0 (undefined)"
    assert flat["t005.r.value"]["display"] == "n/a"


# ---------------------------------------------------------------------------
# E1: 정의 파일 그대로 (L2 세 칸의 평균)
# ---------------------------------------------------------------------------


def cells(values: dict[str, Fraction], level: str) -> dict[str, Any]:
    return {f"{f}/{level}": frac(v) for f, v in values.items()}


def tiny_results() -> tuple[dict[str, Any], dict[str, Any]]:
    fam = ("bad_deployment", "dependency_failure", "traffic_spike")
    l1 = dict(zip(fam, [Fraction(0), Fraction(0), Fraction(1, 10)], strict=True))
    l2 = dict(zip(fam, [Fraction(-9, 10), Fraction(0), Fraction(3, 10)], strict=True))
    l3 = dict(zip(fam, [Fraction(-1), Fraction(0), Fraction(3, 10)], strict=True))
    l12_cells = {**cells(l1, "L1"), **cells(l2, "L2")}
    l12_delta = sum(l1.values(), Fraction(0)) / 6 + sum(l2.values(), Fraction(0)) / 6
    t005 = {
        "candidates": {
            "cand-x": {
                "group": "proposer",
                "oracle": {m: {"delta": frac(l12_delta), "cells": l12_cells} for m in (CD, ES)},
            }
        }
    }
    t006 = {
        "candidates": {
            "cand-x": {
                "group": "proposer",
                "oracle_l3": {
                    m: {"delta": frac(sum(l3.values(), Fraction(0)) / 3), "cells": cells(l3, "L3")}
                    for m in (CD, ES)
                },
            }
        }
    }
    return t005, t006


def test_e1_is_the_mean_of_the_three_l2_cells() -> None:
    t005, t006 = tiny_results()
    (row,) = [r for r in e1_rows(t005, t006) if r["metric"] == CD]
    assert row["e1"] == (Fraction(-9, 10) + 0 + Fraction(3, 10)) / 3 == Fraction(-1, 5)
    assert row["l12"] == Fraction(-6, 10) / 6 + Fraction(1, 10) / 6  # L2 합 -0.6, L1 합 0.1
    assert row["l3"] == Fraction(-7, 30)
    assert row["residual_e1"] == row["l3"] - row["e1"]
    assert row["residual_l12"] == row["l3"] - row["l12"]
    assert row["family"]["bad_deployment"] == (Fraction(-9, 10), Fraction(-1))


def test_the_rejected_first_definition_is_the_l1_l2_delta() -> None:
    """정의 파일의 '버린 첫 정의' 문장: family 1/3 x (L1·L2 평균) = L1/L2 전체 Δ."""
    t005, _ = tiny_results()
    c = t005["candidates"]["cand-x"]["oracle"][CD]
    cell = {key: Fraction(v["exact"]) for key, v in c["cells"].items()}
    families = {key.split("/")[0] for key in cell}
    rejected = sum((cell[f"{f}/L1"] + cell[f"{f}/L2"]) / 2 for f in families) / 3
    assert rejected == Fraction(c["delta"]["exact"])


def test_e1_summary_counts_closer_candidates_and_mean_absolute_residuals() -> None:
    rows = [
        {
            "name": "a",
            "group": "proposer",
            "metric": CD,
            "residual_e1": Fraction(1, 10),
            "residual_l12": Fraction(-3, 10),
        },
        {
            "name": "b",
            "group": "proposer",
            "metric": CD,
            "residual_e1": Fraction(-2, 10),
            "residual_l12": Fraction(1, 10),
        },
        {
            "name": "c",
            "group": "proposer",
            "metric": CD,
            "residual_e1": Fraction(1, 10),
            "residual_l12": Fraction(1, 10),
        },
        {
            "name": "d",
            "group": "control",
            "metric": CD,
            "residual_e1": Fraction(0),
            "residual_l12": Fraction(1),
        },
        {
            "name": "a",
            "group": "proposer",
            "metric": ES,
            "residual_e1": Fraction(0),
            "residual_l12": Fraction(1),
        },
    ]
    s = e1_summary(rows)
    assert s == {
        "closer_count": 1,  # a만(동률 c는 세지 않음)
        "n": 3,
        "mean_abs_residual_e1": Fraction(4, 30),
        "mean_abs_residual_l12": Fraction(5, 30),
    }


# ---------------------------------------------------------------------------
# 전체 values.json
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    return build_values(
        json.loads(T005.read_text()), json.loads(T006.read_text()), json.loads(RUN.read_text())
    )


def test_every_value_has_exact_display_and_a_known_kind(values: dict[str, Any]) -> None:
    assert values
    for key, v in values.items():
        assert set(v) == {"exact", "display", "kind"}, key
        assert v["kind"] in KINDS, key


def test_values_cover_the_results_e1_design_and_run_summary(values: dict[str, Any]) -> None:
    assert values["t005.rq.RQ1.verdict"]["display"] == "not supported"
    assert values["t006.rq.RQ_L3_0.all.verdict"]["display"] == "manipulation not confirmed"
    assert values["t006.candidates.cand-015.oracle_l3.cd.delta"]["display"] == "-0.210"
    assert values["t005.meta.analysis_commit"]["display"] == "bdbecf0"
    assert values["e1.cand-015.cd.e1"]["kind"] == "exploratory"
    assert values["e1.summary.proposer.cd.n"]["display"] == "12"
    assert values["design.epsilon"]["exact"] == str(k.EPSILON)
    assert values["design.n_boot"]["display"] == "2000"
    assert values["design.dev_b_size"]["display"] == "210"
    assert values["run.t006.idle.count"]["display"] == "19"
    assert values["run.t006.idle.count"]["kind"] == "run"
    assert values["hash.t006_analysis"]["display"] == k.TICKET006_ANALYSIS_SHA256


def test_es_floor_rates_are_derived_from_rq_l3_0(values: dict[str, Any]) -> None:
    """ES율 = CD율 - g(S0 다수결). bad_deployment는 L1/L2와 L3 모두 0이다."""
    for side in ("l12", "l3"):
        v = values[f"derived.s0_es_rate.bad_deployment.{side}"]
        assert v["exact"] == "0" and v["kind"] == "derived"


def test_e1_values_use_the_ticket_005_l2_cells(values: dict[str, Any]) -> None:
    t005 = json.loads(T005.read_text())
    cells = t005["candidates"]["cand-003"]["oracle"][CD]["cells"]
    expected = (
        sum(
            (
                Fraction(cells[f"{f}/L2"]["exact"])
                for f in ("bad_deployment", "dependency_failure", "traffic_spike")
            ),
            Fraction(0),
        )
        / 3
    )
    assert values["e1.cand-003.cd.e1"]["exact"] == str(expected)


def test_build_is_deterministic(values: dict[str, Any]) -> None:
    again = build_values(
        json.loads(T005.read_text()), json.loads(T006.read_text()), json.loads(RUN.read_text())
    )
    assert values_bytes(again) == values_bytes(values)


def test_committed_values_equal_a_rebuild(values: dict[str, Any]) -> None:
    assert VALUES.read_bytes() == values_bytes(values)
    sha = VALUES.with_name("values.json.sha256").read_text().split()[0]
    assert sha == hashlib.sha256(VALUES.read_bytes()).hexdigest()


def test_main_checks_inputs_then_writes(tmp_path: Path) -> None:
    out = tmp_path / "values.json"
    assert main(["--out", str(out)]) == 0
    assert out.read_bytes() == VALUES.read_bytes()
    bad = tmp_path / "e1.txt"
    bad.write_text("changed")
    assert main(["--out", str(tmp_path / "v2.json"), "--e1-definition", str(bad)]) == 1
    assert not (tmp_path / "v2.json").exists()


def test_results_files_are_unchanged() -> None:
    assert hashlib.sha256(T005.read_bytes()).hexdigest() == k.TICKET005_ANALYSIS_SHA256
    assert hashlib.sha256(T006.read_bytes()).hexdigest() == k.TICKET006_ANALYSIS_SHA256


EXPECTED_PATHS = {
    "preregistration": "preregistration/ticket-004.sha256",
    "candidate_pool": "skills/candidates/POOL.json",
    "l3_preregistration": "preregistration/ticket-006-l3.sha256",
    "t005_results": "results/ticket-005/analysis.json",
    "t006_results": "results/ticket-006/analysis.json",
    "e1_definition": "writeup/inputs/e1_definition.txt",
}


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="needs the git history")
@pytest.mark.parametrize("name", sorted(EXPECTED_PATHS))
def test_fixed_procedure_commits_exist_and_touch_their_files(name: str) -> None:
    import subprocess

    from evidencegate.writeup import COMMITS

    assert set(COMMITS) == set(EXPECTED_PATHS)
    changed = subprocess.run(
        ["git", "show", "--name-only", "--format=", COMMITS[name]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert EXPECTED_PATHS[name] in changed, (name, changed)
