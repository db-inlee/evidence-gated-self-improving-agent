"""재현용 기록 공개 (Ticket 007 §3 재현용 기록 공개): 비밀 값 확인 → 결정적 압축 → 매니페스트.

비밀 값이 하나라도 걸리면 압축하지 않는다. proposer 장부는 공개 대상이 아니다.
"""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from evidencegate.release import (
    EXCLUDED,
    INCLUDED,
    build_release,
    main,
    release_files,
    scan_secrets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "writeup" / "inputs" / "release_manifest.json"


def fake_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for rel, text in {
        "dev_a/p/t/s0.jsonl": '{"key": 1}\n',
        "oracle/p/t/s0.jsonl": '{"key": 2}\n',
        "l3_oracle/p/t/s0.jsonl": '{"key": 3}\n',
        "ticket006/sessions.jsonl": '{"event": "probe"}\n',
        "proposer/ledger.jsonl": '{"content": "anything"}\n',
        "smoke/x.jsonl": "{}\n",
    }.items():
        path = runs / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return runs


def test_included_and_excluded_parts() -> None:
    assert INCLUDED == (
        "dev_a",
        "dev_b",
        "oracle",
        "l3_oracle",
        "calibration",
        "l3_check",
        "ticket005/sessions.jsonl",
        "ticket006/sessions.jsonl",
    )
    assert "proposer/ledger.jsonl" in EXCLUDED


def test_release_files_skip_the_ledger_and_unlisted_folders(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    files = [p.relative_to(runs).as_posix() for p in release_files(runs)]
    assert files == [
        "dev_a/p/t/s0.jsonl",
        "l3_oracle/p/t/s0.jsonl",
        "oracle/p/t/s0.jsonl",
        "ticket006/sessions.jsonl",
    ]


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "OPENAI_API_KEY=x",
        "Authorization: Bearer abcdefghijklmnop",
        "AKIAABCDEFGHIJKLMNOP",
        "-----BEGIN RSA PRIVATE KEY-----",
        "ghp_" + "a" * 36,
        "mail me at someone@example.com",
        "/Users/someone/project",
        '"api_key": "x"',
    ],
)
def test_a_secret_stops_the_release(tmp_path: Path, secret: str) -> None:
    runs = fake_runs(tmp_path)
    (runs / "oracle" / "p" / "t" / "s0.jsonl").write_text(json.dumps({"x": secret}) + "\n")
    hits = scan_secrets(release_files(runs), runs)
    assert hits and hits[0]["path"] == "oracle/p/t/s0.jsonl"
    out = tmp_path / "dist"
    assert (
        main(
            ["--runs-dir", str(runs), "--out-dir", str(out), "--manifest", str(tmp_path / "m.json")]
        )
        == 1
    )
    assert not out.exists() or not any(out.iterdir())


def test_a_secret_in_the_excluded_ledger_does_not_matter(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    (runs / "proposer" / "ledger.jsonl").write_text("sk-proj-abcdefghijklmnopqrstuvwxyz012345\n")
    assert scan_secrets(release_files(runs), runs) == []


def test_build_is_deterministic_and_the_manifest_lists_every_file(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    a = build_release(runs, tmp_path / "a")
    b = build_release(runs, tmp_path / "b")
    assert a == b
    archive = tmp_path / "a" / a["archive"]
    assert a["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (tmp_path / "a" / a["archive"]).read_bytes() == (
        tmp_path / "b" / b["archive"]
    ).read_bytes()
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert names == [f"runs/{f['path']}" for f in a["files"]]
        for member in tar.getmembers():
            assert member.mtime == 0 and member.uid == 0 and member.uname == ""
    for f in a["files"]:
        assert f["sha256"] == hashlib.sha256((runs / f["path"]).read_bytes()).hexdigest()
    assert a["excluded"] == list(EXCLUDED)
    assert a["secret_scan"] == {
        "patterns": len(a["secret_scan"]["checked"]),
        "hits": 0,
        "checked": a["secret_scan"]["checked"],
    }


def test_main_writes_the_archive_hash_and_manifest(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    out, manifest = tmp_path / "dist", tmp_path / "release_manifest.json"
    assert main(["--runs-dir", str(runs), "--out-dir", str(out), "--manifest", str(manifest)]) == 0
    data = json.loads(manifest.read_text())
    archive = out / data["archive"]
    assert (out / (data["archive"] + ".sha256")).read_text().split()[0] == data["archive_sha256"]
    assert archive.exists()


@pytest.mark.skipif(not (REPO_ROOT / "runs" / "l3_oracle").exists(), reason="needs runs/")
def test_committed_manifest_matches_the_runs() -> None:
    data = json.loads(MANIFEST.read_text())
    runs = REPO_ROOT / "runs"
    listed = [f["path"] for f in data["files"]]
    assert listed == [p.relative_to(runs).as_posix() for p in release_files(runs)]
    for f in data["files"]:
        assert f["sha256"] == hashlib.sha256((runs / f["path"]).read_bytes()).hexdigest(), f["path"]
    assert data["secret_scan"]["hits"] == 0
    assert not any("proposer" in p for p in listed)


def test_committed_manifest_matches_the_run_summary_inputs() -> None:
    """실행 요약이 읽은 기록 파일은 모두 공개본에 있고, 해시가 같다."""
    data = json.loads(MANIFEST.read_text())
    summary = json.loads((REPO_ROOT / "writeup" / "inputs" / "run_summary.json").read_text())
    released = {f"runs/{f['path']}": f["sha256"] for f in data["files"]}
    for path, sha in summary["inputs"].items():
        if path.startswith("runs/"):
            assert released.get(path) == sha, path
