"""Ticket 009 사전 등록 고정.

파일 전체 바이트의 sha256이 기록과 같고, 파일이 git 무시 대상이 아니다.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from evidencegate.t009.preflight import (
    PREREG_FILE,
    PREREG_SHA_FILE,
    check_preregistration_009,
    recorded_sha256,
)

ROOT = Path(__file__).resolve().parents[2]


def test_recorded_sha256_matches_the_whole_file() -> None:
    actual = hashlib.sha256((ROOT / PREREG_FILE).read_bytes()).hexdigest()
    assert recorded_sha256(ROOT / PREREG_SHA_FILE) == actual


def test_sha256_file_follows_the_existing_format() -> None:
    lines = (ROOT / PREREG_SHA_FILE).read_text().splitlines()
    fields = dict(line.split(": ", 1) for line in lines if not line.startswith("#"))
    assert set(fields) >= {"sha256", "file", "range", "lines", "recorded_at"}
    assert fields["file"] == PREREG_FILE.as_posix()
    assert fields["range"] == "whole file (UTF-8 bytes)"
    assert int(fields["lines"]) == len((ROOT / PREREG_FILE).read_text().splitlines())


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_preregistration_files_are_not_git_ignored() -> None:
    for path in (PREREG_FILE, PREREG_SHA_FILE):
        r = subprocess.run(
            ["git", "check-ignore", "-q", path.as_posix()], cwd=ROOT, capture_output=True
        )
        assert r.returncode == 1, f"{path} is ignored by git"


def test_preflight_check_passes_for_the_repo_and_fails_on_any_change(tmp_path: Path) -> None:
    assert check_preregistration_009(ROOT).ok
    shutil.copytree(ROOT / "preregistration", tmp_path / "preregistration")
    copy = tmp_path / PREREG_FILE
    copy.write_bytes(copy.read_bytes() + b" ")
    check = check_preregistration_009(tmp_path)
    assert not check.ok and check.name == "preregistration_009"


def test_preflight_check_fails_when_files_are_missing(tmp_path: Path) -> None:
    assert not check_preregistration_009(tmp_path).ok
