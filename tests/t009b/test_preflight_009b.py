"""Ticket 009b 실행 전 점검(§5의 4): 사전 등록·태그·h·δ 파일·내보내기·출력 경로·전원."""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidencegate.power import Run
from evidencegate.preflight import Check
from evidencegate.t009b import preflight as pf
from evidencegate.t009b import settings
from evidencegate.t009b.propose import FIXED_RUN_OPTIONS, Paths

ROOT = Path(__file__).resolve().parents[2]

AC = "Now drawing from 'AC Power'"
LID = '"AppleClamshellState" = No'
CAF = "pid 1(caffeinate): [0x1] 00:00:01 PreventUserIdleSystemSleep named: x"


def power(batt: str = AC) -> Run:
    def run(argv: Sequence[str]) -> str:
        return {"pmset -g batt": batt, "pmset -g assertions": CAF}.get(" ".join(argv), LID)

    return run


def git_ok(argv: Sequence[str]) -> int:
    return 0


def names(checks: list[Check]) -> dict[str, bool]:
    return {c.name: c.ok for c in checks}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """저장소의 고정 파일만 복사한 작은 root."""
    r = tmp_path / "root"
    shutil.copytree(ROOT / "preregistration", r / "preregistration")
    (r / "results/ticket-009").mkdir(parents=True)
    shutil.copy(ROOT / "results/ticket-009/thresholds.json", r / "results/ticket-009/")
    return r


def test_fixed_values_follow_the_preregistration() -> None:
    assert settings.THRESHOLDS_SHA256.startswith("d030b173")
    assert settings.THRESHOLDS_SHA256.endswith("3520")
    assert settings.EXPORT_SHA256.startswith("4f9afa27")
    assert settings.PREREG_TAG == "t009b-prereg"
    assert settings.N_GO == 25 and settings.DEDUPE_THRESHOLD == 0.55


def test_preregistration_009b_check(root: Path) -> None:
    assert pf.check_preregistration_009b(root).ok
    text = root / "preregistration/ticket-009b.txt"
    text.write_bytes(text.read_bytes() + b" ")
    check = pf.check_preregistration_009b(root)
    assert not check.ok and check.name == "preregistration_009b"


def test_prereg_tag_must_be_an_ancestor_of_head(root: Path) -> None:
    seen: list[list[str]] = []

    def git(argv: Sequence[str]) -> int:
        seen.append(list(argv))
        return 1

    assert pf.check_prereg_tag(root, git_ok).ok
    check = pf.check_prereg_tag(root, git)
    assert not check.ok and "not an ancestor" in check.detail
    assert seen[0][-3:] == ["--is-ancestor", "t009b-prereg", "HEAD"]
    assert not pf.check_prereg_tag(root, lambda argv: 128).ok


def test_thresholds_check(root: Path) -> None:
    assert pf.check_thresholds(root).ok
    path = root / "results/ticket-009/thresholds.json"
    path.write_text(path.read_text() + " ")
    assert not pf.check_thresholds(root).ok


def test_the_real_repository_passes_the_file_checks() -> None:
    assert pf.check_preregistration_009b(ROOT).ok
    assert pf.check_thresholds(ROOT).ok


def make_paths(tmp_path: Path) -> Paths:
    run = tmp_path / "proposer-009b"
    return Paths(
        inputs=tmp_path / "inputs.json",
        ledger=run / "ledger.jsonl",
        out_dir=run / "pool",
        response_models=run / "response_models.jsonl",
        skill=ROOT / "skills/S0/SKILL.md",
        controls=ROOT / "skills/controls",
    )


def test_outputs_check_allows_a_new_run_or_a_resume_of_the_same_run(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    assert pf.check_outputs(paths).ok  # 폴더 없음
    paths.out_dir.mkdir(parents=True)
    (paths.out_dir / "call-000.json").write_text("{}")
    check = pf.check_outputs(paths)
    assert not check.ok and "run_options" in check.detail  # 설정 기록 없는 결과
    (paths.out_dir / "run_options.json").write_text(json.dumps(FIXED_RUN_OPTIONS))
    check = pf.check_outputs(paths)
    assert check.ok and "resume" in check.detail and "1 call" in check.detail
    other = {**FIXED_RUN_OPTIONS, "dedupe_threshold": 0.6}
    (paths.out_dir / "run_options.json").write_text(json.dumps(other))
    assert not pf.check_outputs(paths).ok


def test_power_check_needs_ac_lid_and_caffeinate() -> None:
    assert pf.check_power_now(power()).ok
    check = pf.check_power_now(power(batt="Now drawing from 'Battery Power'"))
    assert not check.ok and "AC" in check.detail


def test_proposer_preflight_lists_every_check(root: Path, tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    checks = pf.proposer_preflight(root, paths, power_run=power(), git=git_ok)
    assert list(names(checks)) == [
        "preregistration_009b",
        "prereg_tag",
        "thresholds",
        "export",
        "strata",
        "prompt",
        "paths",
        "outputs",
        "power",
    ]
    assert not names(checks)["export"]  # tmp 경로에 내보내기 없음


def test_paths_check_rejects_v1_and_009_paths(root: Path, tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    assert pf.check_paths(root, paths).ok
    for bad in ("runs/proposer/ledger.jsonl", "runs/proposer-009/ledger.jsonl"):
        check = pf.check_paths(root, Paths(**{**paths.__dict__, "ledger": root / bad}))
        assert not check.ok, bad
    for bad in ("runs/proposer/pool", "runs/proposer-009/pool"):
        check = pf.check_paths(root, Paths(**{**paths.__dict__, "out_dir": root / bad}))
        assert not check.ok, bad


def test_main_run_preflight_keeps_the_009_checks_and_adds_three(root: Path) -> None:
    extra = pf.main_run_extra_checks(root, git=git_ok)
    assert [c.name for c in extra] == ["preregistration_009b", "prereg_tag", "thresholds"]
    assert all(c.ok for c in extra)
