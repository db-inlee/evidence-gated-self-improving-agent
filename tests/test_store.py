"""JSONL 결과 저장소: 왕복, 경로, 깨진 줄 처리."""

from pathlib import Path

import pytest

from evidencegate.policy import ScriptedPolicy
from evidencegate.runner import EpisodeRecord, run_episode
from evidencegate.skills import Skill
from evidencegate.store import EpisodeStore, StoreCorruptedError
from opsworld.state import Family

SKILL = Skill.from_text("# Skill\n")


def make_records(n: int) -> list[EpisodeRecord]:
    calls = ["get_service_metrics", "get_logs", "get_recent_deployments", "rollback_deployment"]
    policy = ScriptedPolicy(calls, name="t")
    families = list(Family)
    return [run_episode(policy, SKILL, families[i % 3], 900_000 + i, repeat=0) for i in range(n)]


def test_records_round_trip_through_file(tmp_path: Path) -> None:
    records = make_records(4)
    store = EpisodeStore(tmp_path / "r.jsonl")
    for record in records:
        store.append(record)

    reopened = EpisodeStore(tmp_path / "r.jsonl")
    assert reopened.records() == records
    assert reopened.keys() == {r.key for r in records}
    assert all(r.key in reopened for r in records)


def test_each_append_is_one_flushed_line(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = EpisodeStore(path)
    for i, record in enumerate(make_records(3), start=1):
        store.append(record)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == i


def test_appending_an_existing_key_raises(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "r.jsonl")
    record = make_records(1)[0]
    store.append(record)
    with pytest.raises(ValueError, match="already"):
        store.append(record)


def test_for_run_layout_uses_safe_policy_id_config_and_tool_spec(tmp_path: Path) -> None:
    store = EpisodeStore.for_run(
        tmp_path,
        suite="calibration",
        policy_id="ollama:gemma4:12b",
        config_id="c0nf1g000000",
        tool_spec_id="t00l5pec0000",
        skill_id="abc123def456",
    )
    assert store.path == (
        tmp_path
        / "calibration"
        / "ollama__gemma4__12b__c0nf1g000000"
        / "t00l5pec0000"
        / "abc123def456.jsonl"
    )
    store.append(make_records(1)[0])
    assert store.path.exists()


@pytest.mark.parametrize("field", ["suite", "policy_id", "config_id", "tool_spec_id", "skill_id"])
@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".."])
def test_for_run_rejects_path_like_components(tmp_path: Path, field: str, bad: str) -> None:
    parts = {
        "suite": "calibration",
        "policy_id": "scripted:a",
        "config_id": "c0nf1g000000",
        "tool_spec_id": "t00l5pec0000",
        "skill_id": "abc123def456",
    }
    parts[field] = bad
    with pytest.raises(ValueError):
        EpisodeStore.for_run(tmp_path, **parts)


def test_record_keeps_original_policy_id_and_full_config(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "r.jsonl")
    record = make_records(1)[0]
    store.append(record)
    stored = EpisodeStore(tmp_path / "r.jsonl").records()[0]
    assert stored.key.policy_id == "scripted:t"
    assert stored.policy_config["type"] == "scripted"
    assert stored.policy_config["calls"][1] == {"tool": "get_logs", "args": None}


def test_missing_file_is_an_empty_store(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "none.jsonl")
    assert store.keys() == set()
    assert store.records() == []


def test_truncated_last_line_is_dropped_with_warning_and_file_repaired(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = EpisodeStore(path)
    records = make_records(3)
    for record in records:
        store.append(record)
    path.write_bytes(path.read_bytes()[:-25])

    with pytest.warns(UserWarning, match="truncated"):
        repaired = EpisodeStore(path)
    assert repaired.records() == records[:2]
    assert path.read_bytes().endswith(b"\n")

    repaired.append(records[2])
    assert EpisodeStore(path).records() == records


def test_corrupted_middle_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = EpisodeStore(path)
    for record in make_records(3):
        store.append(record)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = "{not json\n"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(StoreCorruptedError):
        EpisodeStore(path)


def test_complete_last_line_missing_only_newline_is_kept(tmp_path: Path) -> None:
    """내용은 다 쓰이고 줄바꿈만 빠진 줄은 깨진 줄이 아니다. 버리지 않고 줄바꿈을 보충한다."""
    path = tmp_path / "r.jsonl"
    store = EpisodeStore(path)
    records = make_records(2)
    for record in records:
        store.append(record)
    path.write_bytes(path.read_bytes()[:-1])

    reopened = EpisodeStore(path)
    assert reopened.records() == records
    reopened.append(make_records(3)[2])
    assert len(EpisodeStore(path).records()) == 3


# ---------------------------------------------------------------------------
# 읽기 전용 열기(repair=False): 실행 중인 파일을 읽어도 바꾸지 않는다
# ---------------------------------------------------------------------------


def write_with_partial_last_line(path: Path, records: list[EpisodeRecord]) -> bytes:
    """쓰는 중인 파일처럼 마지막 줄이 중간에서 끊긴 상태를 만든다."""
    store = EpisodeStore(path)
    for record in records:
        store.append(record)
    data = path.read_bytes()[:-25]
    path.write_bytes(data)
    return data


def test_read_only_skips_partial_last_line_without_touching_file(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    records = make_records(3)
    before = write_with_partial_last_line(path, records)

    with pytest.warns(UserWarning, match="incomplete last line"):
        reader = EpisodeStore(path, repair=False)
        assert reader.records() == records[:2]
    assert path.read_bytes() == before


def test_read_only_keeps_complete_line_missing_newline_without_touching_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r.jsonl"
    store = EpisodeStore(path)
    records = make_records(2)
    for record in records:
        store.append(record)
    before = path.read_bytes()[:-1]
    path.write_bytes(before)

    assert EpisodeStore(path, repair=False).records() == records
    assert path.read_bytes() == before


def test_read_only_still_rejects_corrupted_middle_line(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    store = EpisodeStore(path)
    for record in make_records(3):
        store.append(record)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = "{not json\n"
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(StoreCorruptedError):
        EpisodeStore(path, repair=False)


def test_read_only_store_refuses_to_append(tmp_path: Path) -> None:
    reader = EpisodeStore(tmp_path / "r.jsonl", repair=False)
    with pytest.raises(PermissionError, match="read-only"):
        reader.append(make_records(1)[0])
    assert not (tmp_path / "r.jsonl").exists()


def test_writer_still_repairs_after_a_reader_looked(tmp_path: Path) -> None:
    """읽기 전용으로 본 뒤에도, 쓰는 쪽이 다시 열면 기존처럼 복구한다."""
    path = tmp_path / "r.jsonl"
    records = make_records(3)
    write_with_partial_last_line(path, records)
    with pytest.warns(UserWarning):
        EpisodeStore(path, repair=False)
    with pytest.warns(UserWarning, match="truncated"):
        writer = EpisodeStore(path)
    writer.append(records[2])
    assert EpisodeStore(path).records() == records
