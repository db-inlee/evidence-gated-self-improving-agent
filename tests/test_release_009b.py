"""009b·010 탐침 기록 압축본(Ticket 011 §4). v1 `release`의 결정적 압축을 그대로 쓴다.

비밀 검사 이름 목록은 환경 변수 `EVIDENCEGATE_SECRET_NAMES`에서만 읽는다. 테스트는 가짜 이름만 쓴다.
"""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from evidencegate import release as v1
from evidencegate import release_009b as r
from evidencegate import writeup as w
from evidencegate.analysis import REPO_ROOT

FAKE = "Acme Widgets|Globex"  # 가짜 이름(검사 동작 확인용)
ENV = "EVIDENCEGATE_SECRET_NAMES"


def test_v1_release_lists_are_unchanged() -> None:
    assert v1.INCLUDED == (
        "dev_a",
        "dev_b",
        "oracle",
        "l3_oracle",
        "calibration",
        "l3_check",
        "ticket005/sessions.jsonl",
        "ticket006/sessions.jsonl",
    )
    assert v1.ARCHIVE_NAME == "evidencegate-records.tar.gz"


def test_archive_lists_are_the_ticket_lists() -> None:
    assert set(r.ARCHIVES) == {"t009b", "t010_probe"}
    a = r.ARCHIVES["t009b"]
    assert a.name == "evidencegate-records-009b.tar.gz"
    assert a.included == (
        "t009_calibration",
        "t009_dev_a",
        "t009_dev_b",
        "t009_oracle",
        "ticket009/calibration/sessions.jsonl",
        "ticket009/dev_a/sessions.jsonl",
        "ticket009b/main/sessions.jsonl",
        "proposer_inputs-009",
        "proposer-009b/pool",
        "proposer-009b/response_models.jsonl",
        "proposer-009/pool",
    )
    assert set(a.excluded) == {
        "proposer-009/ledger.jsonl",
        "proposer-009b/ledger.jsonl",
        "proposer-009b/sessions.jsonl",
        "ticket009/calibration/run.log",
        "ticket009/calibration/run_wrapper.sh",
        "ticket009/calibration/wrapper_status.txt",
        "ticket009/dev_a/run.log",
    }
    p = r.ARCHIVES["t010_probe"]
    assert p.name == "evidencegate-records-010-probe.tar.gz"
    assert p.included == ("t010_probe", "ticket010/probe/sessions.jsonl")
    expected = {
        "t009b": Path("writeup/inputs/release_manifest_009b.json"),
        "t010_probe": Path("writeup/inputs/release_manifest_010_probe.json"),
    }
    assert expected == r.MANIFESTS


def fake_runs(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    runs = tmp_path / "runs"
    files = {
        "t009_calibration/x/y/s0.jsonl": '{"a": 1}\n',
        "t009_dev_a/x/y/s0.jsonl": '{"a": 2}\n',
        "t009_dev_b/x/y/c.jsonl": '{"a": 3}\n',
        "t009_oracle/x/y/c.jsonl": '{"a": 4}\n',
        "ticket009/calibration/sessions.jsonl": '{"event": "session_start"}\n',
        "ticket009/calibration/run.log": "log\n",
        "ticket009/calibration/run_wrapper.sh": "#!/bin/sh\n",
        "ticket009/calibration/wrapper_status.txt": "ok\n",
        "ticket009/dev_a/sessions.jsonl": '{"event": "session_end"}\n',
        "ticket009/dev_a/run.log": "log\n",
        "ticket009b/main/sessions.jsonl": '{"event": "probe"}\n',
        "proposer_inputs-009/dev_a.json": "{}\n",
        "proposer-009b/pool/call-000.json": "{}\n",
        "proposer-009b/response_models.jsonl": "{}\n",
        "proposer-009b/ledger.jsonl": '{"content": "x"}\n',
        "proposer-009b/sessions.jsonl": '{"stop_file": "/Users/someone/x"}\n',
        "proposer-009/pool/call-000.json": "{}\n",
        "proposer-009/ledger.jsonl": "{}\n",
        "dev_a/x.jsonl": "{}\n",  # v1 기록: 009b 압축본에 들어가지 않음
        "t010_probe/x/y/s0.jsonl": '{"a": 5}\n',
        "ticket010/probe/sessions.jsonl": '{"event": "probe_start"}\n',
        **(extra or {}),
    }
    for rel, text in files.items():
        (runs / rel).parent.mkdir(parents=True, exist_ok=True)
        (runs / rel).write_text(text, encoding="utf-8")
    return runs


def rels(files: list[Path], runs: Path) -> list[str]:
    return [p.relative_to(runs).as_posix() for p in files]


def test_files_follow_the_lists_and_never_take_an_excluded_file(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    got = rels(r.archive_files(r.ARCHIVES["t009b"], runs), runs)
    assert got == sorted(got)
    assert got == [
        "proposer-009/pool/call-000.json",
        "proposer-009b/pool/call-000.json",
        "proposer-009b/response_models.jsonl",
        "proposer_inputs-009/dev_a.json",
        "t009_calibration/x/y/s0.jsonl",
        "t009_dev_a/x/y/s0.jsonl",
        "t009_dev_b/x/y/c.jsonl",
        "t009_oracle/x/y/c.jsonl",
        "ticket009/calibration/sessions.jsonl",
        "ticket009/dev_a/sessions.jsonl",
        "ticket009b/main/sessions.jsonl",
    ]
    probe = rels(r.archive_files(r.ARCHIVES["t010_probe"], runs), runs)
    assert probe == ["t010_probe/x/y/s0.jsonl", "ticket010/probe/sessions.jsonl"]


def test_an_excluded_file_inside_an_included_folder_is_left_out(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path, {"proposer-009b/pool/ledger.jsonl": "{}\n"})
    spec = r.Archive("x.tar.gz", ("proposer-009b/pool",), ("proposer-009b/pool/ledger.jsonl",))
    assert rels(r.archive_files(spec, runs), runs) == ["proposer-009b/pool/call-000.json"]


def test_a_missing_included_part_is_refused(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    spec = r.Archive("x.tar.gz", ("t009_dev_a", "no_such_folder"), ())
    with pytest.raises(ValueError, match="no_such_folder"):
        r.archive_files(spec, runs)


# 비밀 검사 -----------------------------------------------------------------------------------


def test_names_come_only_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(ValueError, match=ENV):
        r.secret_names()
    for empty in ("", " ", "| |"):
        monkeypatch.setenv(ENV, empty)
        with pytest.raises(ValueError, match=ENV):
            r.secret_names()
    monkeypatch.setenv(ENV, " Acme Widgets | Globex |")
    assert r.secret_names() == ("Acme Widgets", "Globex")


def test_names_are_found_ignoring_case_in_contents_and_paths(tmp_path: Path) -> None:
    runs = fake_runs(
        tmp_path,
        {
            "t009_dev_b/x/y/d.jsonl": '{"a": 1}\n{"note": "built at ACME widgets"}\n',
            "t009_oracle/globex/e.jsonl": "{}\n",
        },
    )
    files = r.archive_files(r.ARCHIVES["t009b"], runs)
    hits = r.scan(files, runs, ("Acme Widgets", "Globex"))
    assert {"path": "t009_dev_b/x/y/d.jsonl", "line": 2} in hits
    assert {"path": "t009_oracle/globex/e.jsonl", "line": 0} in hits
    assert all(set(h) == {"path", "line"} for h in hits)  # 걸린 이름·패턴을 담지 않는다
    assert "acme" not in json.dumps(hits).lower()  # 내용에서 걸린 이름은 결과에 없다


def test_the_v1_patterns_are_still_checked(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path, {"t009_dev_a/x/y/k.jsonl": '{"k": "OPENAI_API_KEY"}\n'})
    files = r.archive_files(r.ARCHIVES["t009b"], runs)
    assert r.scan(files, runs, ("Globex",)) == [{"path": "t009_dev_a/x/y/k.jsonl", "line": 1}]


def build(tmp_path: Path, runs: Path, key: str = "t009b") -> dict[str, Any]:
    return r.build_archive(key, runs, tmp_path / "dist", names=("Acme Widgets", "Globex"))


def test_a_hit_stops_the_build_and_writes_nothing(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path, {"t009_dev_b/x/y/d.jsonl": "Globex\n"})
    with pytest.raises(ValueError, match="1 hit") as error:
        build(tmp_path, runs)
    assert "globex" not in str(error.value).lower()
    assert not (tmp_path / "dist").exists()


def test_build_is_deterministic_and_the_manifest_has_the_v1_format(tmp_path: Path) -> None:
    runs = fake_runs(tmp_path)
    a = build(tmp_path / "a", runs)
    b = build(tmp_path / "b", runs)
    name = "evidencegate-records-009b.tar.gz"
    data = (tmp_path / "a" / "dist" / name).read_bytes()
    assert data == (tmp_path / "b" / "dist" / name).read_bytes() and a == b
    v1_keys = set(json.loads((REPO_ROOT / w.RELEASE_MANIFEST_PATH).read_text()))
    assert set(a) == v1_keys
    assert a["archive"] == name and a["archive_sha256"] == hashlib.sha256(data).hexdigest()
    assert a["archive_bytes"] == len(data)
    assert a["included"] == list(r.ARCHIVES["t009b"].included)
    assert a["excluded"] == list(r.ARCHIVES["t009b"].excluded)
    assert a["secret_scan"]["hits"] == 0
    assert a["secret_scan"]["checked"] == [*sorted(v1.SECRET_PATTERNS), "secret_names"]
    assert "acme" not in json.dumps(a).lower() and "globex" not in json.dumps(a).lower()
    assert (tmp_path / "a" / "dist" / f"{name}.sha256").read_text() == (
        f"{a['archive_sha256']}  {name}\n"
    )
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data))) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == [f"runs/{f['path']}" for f in a["files"]]
    assert {(m.mtime, m.mode, m.uid, m.gid, m.uname, m.gname) for m in members} == {
        (0, 0o644, 0, 0, "", "")
    }


def test_main_needs_the_names_variable_and_keeps_names_out_of_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = fake_runs(tmp_path, {"t009_dev_b/x/y/d.jsonl": "{}\nacme widgets\n"})
    out = tmp_path / "dist"
    root = tmp_path / "root"
    argv = ["build", "t009b", "--runs-dir", str(runs), "--out-dir", str(out), "--root", str(root)]
    monkeypatch.delenv(ENV, raising=False)
    assert r.main(argv) == 1 and not out.exists()
    assert ENV in capsys.readouterr().err
    monkeypatch.setenv(ENV, FAKE)
    assert r.main(argv) == 1 and not out.exists()
    text = capsys.readouterr()
    assert "t009_dev_b/x/y/d.jsonl:2" in text.err
    assert "acme" not in (text.out + text.err).lower()
    assert "globex" not in (text.out + text.err).lower()


def test_main_writes_the_archive_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = fake_runs(tmp_path)
    out, root = tmp_path / "dist", tmp_path / "root"
    monkeypatch.setenv(ENV, FAKE)
    for key in ("t009b", "t010_probe"):
        argv = ["build", key, "--runs-dir", str(runs), "--out-dir", str(out), "--root", str(root)]
        assert r.main(argv) == 0
        manifest = json.loads((root / r.MANIFESTS[key]).read_text())
        assert (out / manifest["archive"]).exists()
        assert r.main(argv) == 0  # 같은 기록이면 같은 매니페스트
    path = root / r.MANIFESTS["t009b"]
    path.write_text("{}\n")
    argv = ["build", "t009b", "--runs-dir", str(runs), "--out-dir", str(out), "--root", str(root)]
    assert r.main(argv) == 1  # 다른 매니페스트는 덮어쓰지 않는다


def test_dry_run_lists_files_and_size_without_names_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = fake_runs(tmp_path)
    monkeypatch.delenv(ENV, raising=False)
    out = tmp_path / "dist"
    assert r.main(["dry-run", "t010_probe", "--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "t010_probe/x/y/s0.jsonl" in text and "2 files" in text and "secret scan skipped" in text
    assert not out.exists()


# writeup 값 -----------------------------------------------------------------------------------


def test_release_values_come_from_the_follow_up_manifests() -> None:
    manifest = {
        "archive": "evidencegate-records-009b.tar.gz",
        "archive_sha256": "a" * 64,
        "archive_bytes": 10,
        "files": [{"path": "x", "sha256": "b" * 64, "bytes": 1}] * 3,
        "included": [],
        "excluded": [],
        "secret_scan": {"patterns": 11, "hits": 0, "checked": []},
    }
    values = w.follow_up_release_values({"t009b": manifest})
    assert values["hash.records_archive_009b"] == {
        "exact": "a" * 64,
        "display": "a" * 64,
        "kind": "provenance",
    }
    assert values["release.t009b.archive"]["display"] == "evidencegate-records-009b.tar.gz"
    assert values["release.t009b.file_count"] == {"exact": "3", "display": "3", "kind": "run"}
    assert values["release.t009b.secret_hits"]["display"] == "0"
    assert values["release.t009b.archive_bytes"]["kind"] == "run"
    probe = w.follow_up_release_values({"t010_probe": manifest | {"archive": "p.tar.gz"}})
    assert "hash.records_archive_010_probe" in probe and "release.t010_probe.archive" in probe
    assert w.follow_up_release_values({}) == {}


def test_load_follow_up_reads_manifests_only_when_they_exist(tmp_path: Path) -> None:
    follow = w.load_follow_up(REPO_ROOT)
    present = {k for k, p in r.MANIFESTS.items() if (REPO_ROOT / p).exists()}
    assert set(follow.releases) == present
    assert w.RELEASE_MANIFESTS_009B == r.MANIFESTS


def test_load_follow_up_picks_up_a_written_manifest(tmp_path: Path) -> None:
    import shutil

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
    manifest = {"archive": "p.tar.gz", "archive_sha256": "c" * 64, "archive_bytes": 1,
                "files": [], "included": [], "excluded": [],
                "secret_scan": {"patterns": 11, "hits": 0, "checked": []}}  # fmt: skip
    (root / r.MANIFESTS["t010_probe"]).write_text(json.dumps(manifest))
    follow = w.load_follow_up(root)
    assert set(follow.releases) == {"t010_probe"}
    assert follow.releases["t010_probe"]["archive_sha256"] == "c" * 64
