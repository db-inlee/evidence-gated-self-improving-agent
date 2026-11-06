"""Ticket 009b 사전 등록(원칙 5, 완료 기준): 009 r1에서 '그대로' 옮긴 절이 바이트까지 같은지."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from evidencegate.t009.preflight import recorded_sha256
from evidencegate.t009b import prereg_sections as ps

ROOT = Path(__file__).resolve().parents[2]
P009 = ROOT / "preregistration" / "ticket-009.txt"
P009B = ROOT / "preregistration" / "ticket-009b.txt"
R1_SHA256 = "dd5f93a345bd2c9ec8bb30ad14ffaa875e80b92bea3645447de499f183dae882"


def test_the_009_source_is_the_r1_text() -> None:
    assert hashlib.sha256(P009.read_bytes()).hexdigest() == R1_SHA256


def test_sections_split_on_level_two_and_three_headings() -> None:
    text = "# T\n\nhead\n\n## 1. A\n\na\n\n### 1.1 B\n\nb\n## 2. C\nc\n"
    sections = ps.split_sections(text)
    assert list(sections) == ["", "## 1. A", "### 1.1 B", "## 2. C"]
    assert sections["### 1.1 B"] == "\nb\n"
    assert sections["## 2. C"] == "c\n"
    assert "".join((h + "\n" if h else "") + body for h, body in sections.items()) == text


def test_every_009_section_has_a_rule() -> None:
    headings = [h for h in ps.split_sections(P009.read_text(encoding="utf-8")) if h]
    assert sorted(headings) == sorted(ps.ruled_headings())


def test_009b_keeps_the_same_sections_byte_for_byte() -> None:
    assert ps.check(P009.read_text(encoding="utf-8"), P009B.read_text(encoding="utf-8")) == []


def test_check_catches_a_changed_same_section() -> None:
    old, new = P009.read_text(encoding="utf-8"), P009B.read_text(encoding="utf-8")
    changed = new.replace("단측 p = Σ_{x=a}", "단측 p = Σ_{x=a+1}", 1)
    assert changed != new
    assert any("6.1" in p for p in ps.check(old, changed))


def test_check_catches_an_unlisted_change_in_a_replaced_section() -> None:
    old, new = P009.read_text(encoding="utf-8"), P009B.read_text(encoding="utf-8")
    changed = new.replace(
        "- 실행 중단 규칙은 v1과 같다(`eval_constants`의 값).", "- 실행 중단 규칙.", 1
    )
    assert changed != new
    assert any("## 8." in p for p in ps.check(old, changed))


def test_check_catches_a_moved_or_prefix_section_change() -> None:
    old, new = P009.read_text(encoding="utf-8"), P009B.read_text(encoding="utf-8")
    moved = new.replace(
        "- 귀무 라벨률(효과 0 후보 200,000개)", "- 귀무 라벨률(효과 0 후보 100,000개)", 1
    )
    assert any("9." in p for p in ps.check(old, moved))
    prefix = new.replace(
        "- **proposer 후보 수:** 40(최소선과 같음)", "- **proposer 후보 수:** 41", 1
    )
    assert any("10." in p for p in ps.check(old, prefix))


@pytest.mark.parametrize(
    "needle",
    [
        "--previous valid",
        "t = 0.55",
        "25회 연속",
        "120회",
        "$5",
        "N_go = 25",
        "e0ebf7893fcaa20810f90aedc872b0c8ffcb0c4b6de997e141d7aec6eed3d494",
        "dfaac95fa7097fcd51fd6c60e60f10fcaebb1ae240ec686b8443a3f505f95795",
        "cf02d6fd943e4d97b744f594436a49ac1c62f05569f25e11326ed892bf4e5491",
        "d030b1734811d8a0acc8fdfd198d3d2f0b8ce87c1e81faf322e1f6b1805c3520",
        "4f9afa27360f01cfbf6a680bbc4ae2fc0a9f52cbe30f5d14f7bba11b54504c6d",
        "plans/ticket-009b-pool-groups.json",
        "response_models.jsonl",
        "gpt-5.4-2026-03-05",
        "no credits remaining",
        "runs/proposer-009b/",
        "skills/candidates-009b/POOL-009b.json",
        "민감도 판정",
        "격자의 아래 끝",
    ],
)
def test_009b_states_the_decided_values(needle: str) -> None:
    assert needle in P009B.read_text(encoding="utf-8")


def test_check_catches_an_extra_or_missing_heading() -> None:
    old, new = P009.read_text(encoding="utf-8"), P009B.read_text(encoding="utf-8")
    extra = new + "\n### 12.1 덧붙인 절\n\n- x\n"
    assert any("headings differ" in p for p in ps.check(old, extra))
    missing = new.replace(ps.H92 + "\n", "", 1)
    assert any("headings differ" in p for p in ps.check(old, missing))


# 고정(해시 파일) ------------------------------------------------------------------

P009B_SHA = ROOT / "preregistration" / "ticket-009b.sha256"


def sha_fields() -> dict[str, str]:
    lines = P009B_SHA.read_text(encoding="utf-8").splitlines()
    return dict(line.split(": ", 1) for line in lines if not line.startswith("#"))


def test_recorded_sha256_matches_the_whole_009b_file() -> None:
    assert recorded_sha256(P009B_SHA) == hashlib.sha256(P009B.read_bytes()).hexdigest()


def test_009b_sha256_file_follows_the_009_format() -> None:
    fields = sha_fields()
    assert set(fields) == set(
        dict(
            line.split(": ", 1)
            for line in (ROOT / "preregistration" / "ticket-009.sha256").read_text().splitlines()
            if not line.startswith("#")
        )
    )
    assert fields["file"] == "preregistration/ticket-009b.txt"
    assert fields["range"] == "whole file (UTF-8 bytes)"
    assert int(fields["lines"]) == len(P009B.read_text(encoding="utf-8").splitlines())
    assert fields["recorded_at"].endswith("+09:00")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_009b_preregistration_files_are_not_git_ignored() -> None:
    for path in (P009B, P009B_SHA):
        r = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True)
        assert r.returncode == 1, f"{path} is ignored by git"
