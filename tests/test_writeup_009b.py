"""값 층 확장 (Ticket 011 §1): 009b 결과, 010 탐침, 009 h·δ, 실행 요약 009b, 설계 상수, 커밋.

v1 값(키·exact·display·kind)은 이전 values.json(커밋 0a178fd)과 바이트까지 같아야 한다.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evidencegate import writeup as w
from evidencegate.t009 import splits as s9
from evidencegate.t009b import settings as s9b

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"
V1_VALUES_COMMIT = "0a178fd"  # Ticket 007 README 렌더 커밋(v1 values.json의 마지막 변경)
HAS_GIT = (REPO_ROOT / ".git").exists()


def git(*argv: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *argv], cwd=REPO_ROOT, capture_output=True, check=False)


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(VALUES.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def built() -> dict[str, dict[str, str]]:
    t005 = json.loads((REPO_ROOT / w.T005_PATH).read_text())
    t006 = json.loads((REPO_ROOT / w.T006_PATH).read_text())
    run = json.loads((REPO_ROOT / w.RUN_SUMMARY_PATH).read_text())
    release = json.loads((REPO_ROOT / w.RELEASE_MANIFEST_PATH).read_text())
    return w.build_values(
        t005, t006, run, REPO_ROOT, release=release, follow_up=w.load_follow_up(REPO_ROOT)
    )


# 입력과 고정 sha256 ---------------------------------------------------------------------------


def test_fixed_follow_up_hashes() -> None:
    assert w.T009B_ANALYSIS_SHA256 == (
        "21c87f2798fa032e3ce627029e8b01c7629cb9b73215aeac1b3742f36d3cdafe"
    )
    assert w.T010_PROBE_SHA256 == (
        "f907489e277e3e3fd3d237d5e61d71554b2f1176bf81d53d62a0bc254835158c"
    )
    assert w.T009_THRESHOLDS_SHA256 == s9b.THRESHOLDS_SHA256
    assert w.RUN_SUMMARY_009B_SHA256 == (
        "ff203e7e926de66210d5f25852fb20909830905537db22f55b4d778aa78a326f"
    )
    for path, digest in w.FOLLOW_UP_INPUTS.values():
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest, path


def copy_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for rel in (
        *(p for p, _ in w.FOLLOW_UP_INPUTS.values()),
        w.T009B_SIDECAR,
        w.RUN_SUMMARY_009B_SIDECAR,
        w.PREREG_009B_PATH,
        w.PREREG_009_PATH,
        w.PREREG_009_SHA_PATH,
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, root / rel)
    return root


def test_copied_inputs_load(tmp_path: Path) -> None:
    follow = w.load_follow_up(copy_inputs(tmp_path))
    assert follow.t009b["rq"]["rq9_1"]["verdict"] == "supported"


@pytest.mark.parametrize("name", sorted(w.FOLLOW_UP_INPUTS))
def test_a_changed_follow_up_input_is_refused(tmp_path: Path, name: str) -> None:
    root = copy_inputs(tmp_path)
    path = root / w.FOLLOW_UP_INPUTS[name][0]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sha256"):
        w.load_follow_up(root)


@pytest.mark.parametrize("sidecar", ["t009b", "run_summary_009b"])
def test_a_sidecar_that_differs_is_refused(tmp_path: Path, sidecar: str) -> None:
    root = copy_inputs(tmp_path)
    rel = w.T009B_SIDECAR if sidecar == "t009b" else w.RUN_SUMMARY_009B_SIDECAR
    (root / rel).write_text("0" * 64 + "  x\n")
    with pytest.raises(ValueError, match="sidecar"):
        w.load_follow_up(root)


def test_009b_inputs_must_name_the_preregistration_and_thresholds(tmp_path: Path) -> None:
    root = copy_inputs(tmp_path)
    prereg = root / w.PREREG_009B_PATH
    prereg.write_bytes(prereg.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="preregistration"):
        w.load_follow_up(root)
    analysis = json.loads((REPO_ROOT / w.T009B_PATH).read_text())
    assert analysis["inputs"]["thresholds"] == w.T009_THRESHOLDS_SHA256
    bad = dict(analysis, inputs=analysis["inputs"] | {"thresholds": "0" * 64})
    with pytest.raises(ValueError, match="thresholds"):
        w.check_t009b_inputs(bad, root=REPO_ROOT)


@pytest.mark.skipif(not HAS_GIT, reason="needs the git history")
def test_probe_json_equals_the_file_at_the_ticket_010_closed_tag() -> None:
    shown = git("show", "ticket-010-closed:results/ticket-010/probe.json")
    if shown.returncode != 0:
        pytest.skip("tag ticket-010-closed is not available")
    assert shown.stdout == (REPO_ROOT / w.T010_PATH).read_bytes()


# v1 값 불변 -----------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GIT, reason="needs the git history")
def test_every_v1_value_is_unchanged(built: dict[str, Any]) -> None:
    shown = git("show", f"{V1_VALUES_COMMIT}:writeup/generated/values.json")
    assert shown.returncode == 0
    old: dict[str, Any] = json.loads(shown.stdout)
    assert len(old) == 11_471
    changed = [k for k, v in old.items() if built.get(k) != v]
    assert changed == []


def test_committed_values_equal_a_rebuild_with_the_follow_up(
    values: dict[str, Any], built: dict[str, Any]
) -> None:
    assert VALUES.read_bytes() == w.values_bytes(built)


# 펴기, kind, 표시 -----------------------------------------------------------------------------


def test_follow_up_kinds(values: dict[str, Any]) -> None:
    expected = {
        "t009b.": "result",
        "t009.thresholds.": "result",
        "t010.": "exploratory",
        "design.t009b.": "design",
        "provenance.": "provenance",
        "derived.t009b.": "derived",
    }
    for prefix, kind in expected.items():
        keys = [k for k in values if k.startswith(prefix)]
        assert keys, prefix
        assert {values[k]["kind"] for k in keys} == {kind}, prefix
    run_keys = [k for k in values if k.startswith(("run.t009.", "run.t009b.", "run.same_seed."))]
    assert run_keys and {values[k]["kind"] for k in run_keys} == {"run"}
    for key in ("hash.t009b_analysis", "hash.t010_probe", "hash.t009_thresholds"):
        assert values[key]["kind"] == "provenance"


def test_p_values_display_as_exact_fractions(values: dict[str, Any]) -> None:
    assert values["t009b.rq.rq9_1.p"]["display"] == "1/9044"
    assert values["t009b.rq.rq9_5.p"]["display"] == "1/252"
    assert values["t009b.cluster_sensitivity.rq9_1.p"]["display"] == "1/429"
    flat = w.flatten_result("x", {"p": {"exact": "1/3", "float": 1 / 3}, "q": {"exact": "1/3",
                                  "float": 1 / 3}})  # fmt: skip
    assert flat["x.p"]["display"] == "1/3" and flat["x.q"]["display"] == "0.333"


def test_follow_up_values_read_from_the_files(values: dict[str, Any]) -> None:
    assert values["t009b.rq.rq9_1.verdict"]["display"] == "supported"
    assert values["t009b.rq.rq9_1_local_harm.verdict"]["display"] == "direction only"
    assert values["t009b.rq.rq9_3.interval"]["display"] == "[0.282, 0.718]"
    assert values["t009b.metrics.C5.cd.severe.successes"]["display"] == "2"
    assert values["t009b.metrics.C5.cd.accepted.count"]["display"] == "13"
    assert values["t010.variants.0.name"]["display"] == "P0"
    assert values["t010.decision.verdict"]["display"] == "reconsider"
    assert values["t010.rule.k"]["display"] == "8"
    assert values["run.t009.pool.size"]["display"] == "14"
    assert values["run.same_seed.same_disposition"]["display"] == "575"
    assert values["provenance.t009b.main.sessions.1.start"]["display"] == (
        "2026-11-07T00:39:48+00:00"
    )
    assert values["derived.t009b.cluster_groups"]["display"] == "29"
    assert not any(
        k.startswith("provenance.inputs.") for k in values
    )  # v1처럼 입력 목록은 펴지 않음


def test_design_values_come_from_the_split_and_settings_modules(values: dict[str, Any]) -> None:
    cells = 6
    expected = {
        "cells": cells,
        "dev_a_per_cell": s9.DEV_A[2],
        "dev_a_size": s9.DEV_A[2] * cells,
        "dev_b_per_cell": s9.DEV_B[2],
        "dev_b_size": s9.DEV_B[2] * cells,
        "oracle_per_cell": s9.ORACLE[2],
        "oracle_size": s9.ORACLE[2] * cells,
        "calibration_per_cell": s9.CALIBRATION_V1_PER_CELL + s9.CALIBRATION_ADD[2],
        "h_n": s9.DEV_B[2],
        "delta_n": s9.DEV_B[2] * cells,
        "pool_target": s9b.POOL_TARGET,
        "n_go": s9b.N_GO,
        "stop_after_idle": s9b.STOP_AFTER_IDLE,
        "max_calls": s9b.MAX_CALLS,
    }
    for name, value in expected.items():
        assert values[f"design.t009b.{name}"]["exact"] == str(value), name
    assert values["design.t009b.h_n"]["display"] == "70"
    assert values["design.t009b.delta_n"]["display"] == "420"
    assert values["design.t009b.dedupe_threshold"]["exact"] == repr(s9b.DEDUPE_THRESHOLD)


def test_follow_up_hash_values(values: dict[str, Any]) -> None:
    analysis = json.loads((REPO_ROOT / w.T009B_PATH).read_text())
    assert values["hash.t009b_analysis"]["display"] == w.T009B_ANALYSIS_SHA256
    assert values["hash.t010_probe"]["display"] == w.T010_PROBE_SHA256
    assert values["hash.t009_thresholds"]["display"] == w.T009_THRESHOLDS_SHA256
    assert values["hash.run_summary_009b"]["display"] == w.RUN_SUMMARY_009B_SHA256
    assert values["hash.preregistration_009b"]["display"] == analysis["inputs"]["preregistration"]
    for name in ("pool", "plan", "groups"):
        assert values[f"hash.t009b_{name}"]["display"] == analysis["inputs"][name]
    prereg_009 = hashlib.sha256((REPO_ROOT / w.PREREG_009_PATH).read_bytes()).hexdigest()
    assert values["hash.preregistration_009"]["display"] == prereg_009


# 연구 ② 고정 절차 커밋 ------------------------------------------------------------------------

EXPECTED_009B_PATHS = {
    "t009_preregistration": "preregistration/ticket-009.txt",
    "t009_thresholds": "results/ticket-009/thresholds.json",
    "t009_closed": "results/ticket-009/proposer-calls-manifest.json",
    "t009b_preregistration": "preregistration/ticket-009b.txt",
    "t009b_pool": "skills/candidates-009b/POOL-009b.json",
    "t009b_groups": "plans/ticket-009b-pool-groups.json",
    "t009b_plan": "plans/ticket-009b-main-plan.json",
    "t009b_analysis_code": "src/evidencegate/t009b/analysis.py",
    "t009b_results": "results/ticket-009b/analysis.json",
    "t010_probe": "results/ticket-010/probe.json",
}


def test_follow_up_commit_names_and_values(values: dict[str, Any]) -> None:
    assert set(w.COMMITS_009B) == set(EXPECTED_009B_PATHS)
    assert not set(w.COMMITS_009B) & set(w.COMMITS)
    assert w.COMMITS_009B["t009b_preregistration"] == "12a36fd"
    assert w.COMMITS_009B["t009b_analysis_code"] == "94fa123"
    assert w.COMMITS_009B["t009b_results"] == "adb9939"
    assert w.COMMITS_009B["t010_probe"] == "cd76a4e"
    for name, commit in w.COMMITS_009B.items():
        assert values[f"commit.{name}"] == {
            "exact": commit,
            "display": commit,
            "kind": "provenance",
        }


@pytest.mark.skipif(not HAS_GIT, reason="needs the git history")
@pytest.mark.parametrize("name", sorted(EXPECTED_009B_PATHS))
def test_follow_up_commits_exist_and_touch_their_files(name: str) -> None:
    shown = git("show", "--name-only", "--format=", w.COMMITS_009B[name])
    assert shown.returncode == 0, name
    assert EXPECTED_009B_PATHS[name] in shown.stdout.decode().split(), name


def test_v1_only_build_has_no_follow_up_keys() -> None:
    t005 = json.loads((REPO_ROOT / w.T005_PATH).read_text())
    t006 = json.loads((REPO_ROOT / w.T006_PATH).read_text())
    run = json.loads((REPO_ROOT / w.RUN_SUMMARY_PATH).read_text())
    release = json.loads((REPO_ROOT / w.RELEASE_MANIFEST_PATH).read_text())
    v1 = w.build_values(t005, t006, run, REPO_ROOT, release=release)
    assert not any(k.startswith(("t009b.", "t010.", "design.t009b.", "provenance.")) for k in v1)
