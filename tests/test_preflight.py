"""세션마다 하는 실행 전 점검 (Ticket 005 §2). 하나라도 실패하면 실행하지 않는다."""

import copy
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from evidencegate.analysis import load_pool
from evidencegate.eval_constants import MIN_FREE_DISK_BYTES
from evidencegate.machine import Machine
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.preflight import (
    check_config_id,
    check_disk,
    check_engine,
    check_frozen_pool,
    check_machine,
    check_parallel,
    check_plan,
    check_preregistration_7_8,
    check_preregistration_l3,
    preflight_ok,
    reference_policy_config,
    run_preflight,
)
from evidencegate.run_plan import build_plan, plan_bytes, plan_sha256
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
M3 = Machine("Mac15,3", "Apple M3")

# S0 Dev-A 기록(Ticket 004)의 정책 설정. 이 dict의 content_id가 config 7c66ee5bf13c다.
REFERENCE: dict[str, Any] = {
    "type": "ollama",
    "model": "gpt-oss:20b",
    "model_digest": "f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe",
    "quantization": "MXFP4",
    "ollama_version": "0.40.3",
    "instructions_sha256": "332b776ce67826fba8829790fc46090df243c2f9f38602e1c8c72faea4bfaed3",
    "options": {"temperature": 0.7, "num_ctx": 8192},
    "think": "low",
    "stop": {"max_tool_calls": 8, "max_consecutive_text_turns": 2},
    "format_error_policy": {"feedback": "error_as_tool_result", "max_consecutive_format_errors": 3},
    "seed_rule": "derive_llm_seed(family, incident_seed, repeat): sha256, first 8 bytes mod 2^31",
    "machine": {"hw_model": "Mac15,3", "chip": "Apple M3"},
    "parallel": {"client_concurrency": 2, "server_num_parallel": 2},
}


def changed(path: str, value: object) -> dict[str, Any]:
    out = copy.deepcopy(REFERENCE)
    node = out
    *parents, leaf = path.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value
    return out


# ---------------------------------------------------------------------------
# 엔진 동일성과 config
# ---------------------------------------------------------------------------


def test_engine_identity_passes_for_the_same_config() -> None:
    assert check_engine(copy.deepcopy(REFERENCE), REFERENCE).ok


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model_digest", "0" * 64),
        ("ollama_version", "0.41.0"),
        ("quantization", "Q4_K_M"),
        ("instructions_sha256", "1" * 64),
        ("options.temperature", 0.8),
        ("options.num_ctx", 4096),
        ("think", "medium"),
    ],
)
def test_engine_identity_fails_and_names_the_difference(path: str, value: object) -> None:
    result = check_engine(changed(path, value), REFERENCE)
    assert not result.ok
    assert f"{path}:" in result.detail  # 중첩 키도 전체 경로로


def test_config_id_matches_the_fixed_condition() -> None:
    """S0 Dev-A의 정책 설정 그대로면 config_id가 7c66ee5bf13c다."""
    assert check_config_id(copy.deepcopy(REFERENCE)).ok
    assert not check_config_id(changed("options.temperature", 0.8)).ok


def test_machine_must_be_mac15_3() -> None:
    assert check_machine(M3).ok
    assert not check_machine(Machine("Mac14,2", "Apple M2")).ok


@pytest.mark.parametrize(
    ("parallel", "ok"),
    [
        ({"client_concurrency": 2, "server_num_parallel": 2}, True),
        ({"client_concurrency": 1, "server_num_parallel": 2}, False),
        ({"client_concurrency": 2, "server_num_parallel": 4}, False),
        (None, False),
    ],
)
def test_parallel_settings(parallel: dict[str, int] | None, ok: bool) -> None:
    config = copy.deepcopy(REFERENCE)
    if parallel is None:
        del config["parallel"]
    else:
        config["parallel"] = parallel
    assert check_parallel(config).ok is ok


def record(policy_config: dict[str, Any], seed: int) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey(
            "ollama:gpt-oss:20b",
            "7c66ee5bf13c",
            "e9b4996f7529",
            "bf65e569ae37",
            "bad_deployment",
            seed,
            0,
            "L1",
        ),
        policy_config=policy_config,
        llm_seed=0,
        correct_disposition=True,
        evidence_safe=True,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-10-15T00:00:00+00:00",
        duration_s=1.0,
    )


def s0_dev_a_store(runs: Path, configs: list[dict[str, Any]]) -> None:
    store = EpisodeStore.for_run(
        runs,
        suite="dev_a",
        policy_id="ollama:gpt-oss:20b",
        config_id="7c66ee5bf13c",
        tool_spec_id="e9b4996f7529",
        skill_id="bf65e569ae37",
    )
    for i, c in enumerate(configs):
        store.append(record(c, 100_000 + i))


def test_reference_comes_from_the_s0_dev_a_store(tmp_path: Path) -> None:
    s0_dev_a_store(tmp_path, [REFERENCE, REFERENCE])
    assert reference_policy_config(tmp_path) == REFERENCE


def test_reference_needs_one_config_across_the_s0_dev_a_store(tmp_path: Path) -> None:
    s0_dev_a_store(tmp_path, [REFERENCE, changed("think", "medium")])
    with pytest.raises(ValueError, match="policy_config"):
        reference_policy_config(tmp_path)


def test_reference_needs_the_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="S0 Dev-A"):
        reference_policy_config(tmp_path)


# ---------------------------------------------------------------------------
# 동결 풀, 7.8 해시, 실행 계획, 디스크
# ---------------------------------------------------------------------------


def test_frozen_pool_and_preregistration_pass_for_the_repo() -> None:
    assert check_frozen_pool(REPO_ROOT).ok
    assert check_preregistration_7_8(REPO_ROOT).ok


def test_frozen_pool_fails_when_a_candidate_changes(tmp_path: Path) -> None:
    shutil.copytree(REPO_ROOT / "skills", tmp_path / "skills")
    path = tmp_path / "skills" / "candidates" / "cand-000" / "SKILL.md"
    path.write_text(path.read_text() + " ")
    result = check_frozen_pool(tmp_path)
    assert not result.ok and "cand-000" in result.detail


def test_preregistration_fails_when_the_copy_changes(tmp_path: Path) -> None:
    shutil.copytree(REPO_ROOT / "preregistration", tmp_path / "preregistration")
    copy_path = tmp_path / "preregistration" / "ticket-004-section-7.8.txt"
    copy_path.write_text(copy_path.read_text() + " ")
    assert not check_preregistration_7_8(tmp_path).ok


def write_plan(tmp_path: Path) -> Path:
    plan = build_plan(load_pool(REPO_ROOT), REPO_ROOT)
    path = tmp_path / "plan.json"
    path.write_bytes(plan_bytes(plan))
    (tmp_path / "plan.json.sha256").write_text(f"{plan_sha256(plan)}  plan.json\n")
    return path


def test_plan_check_passes_for_the_recorded_plan(tmp_path: Path) -> None:
    result = check_plan(write_plan(tmp_path), REPO_ROOT)
    assert result.ok
    assert plan_sha256(build_plan(load_pool(REPO_ROOT), REPO_ROOT)) in result.detail


def test_plan_check_fails_when_the_file_does_not_match_its_hash(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"version":1', b'"version":2'))
    assert not check_plan(path, REPO_ROOT).ok


def test_plan_check_fails_when_the_plan_differs_from_a_rebuild(tmp_path: Path) -> None:
    """파일과 해시가 맞아도 지금 풀로 다시 만든 계획과 다르면 실패."""
    import hashlib

    path = write_plan(tmp_path)
    data = path.read_bytes().replace(b'"repeat":0', b'"repeat":9', 1)
    path.write_bytes(data)
    (tmp_path / "plan.json.sha256").write_text(f"{hashlib.sha256(data).hexdigest()}  plan.json\n")
    result = check_plan(path, REPO_ROOT)
    assert not result.ok and "rebuild" in result.detail


def test_plan_check_fails_without_the_hash_file(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    (tmp_path / "plan.json.sha256").unlink()
    assert not check_plan(path, REPO_ROOT).ok


Usage = namedtuple("Usage", "total used free")


def test_disk_check_uses_the_threshold(tmp_path: Path) -> None:
    assert check_disk(tmp_path, usage=lambda _: Usage(0, 0, MIN_FREE_DISK_BYTES)).ok
    assert not check_disk(tmp_path, usage=lambda _: Usage(0, 0, MIN_FREE_DISK_BYTES - 1)).ok


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------


def full(tmp_path: Path, **overrides: Any) -> list[Any]:
    runs = tmp_path / "runs"
    s0_dev_a_store(runs, [REFERENCE])
    kwargs: dict[str, Any] = {
        "root": REPO_ROOT,
        "runs_dir": runs,
        "current_config": copy.deepcopy(REFERENCE),
        "machine": M3,
        "plan_path": write_plan(tmp_path),
        "disk_usage": lambda _: Usage(0, 0, 10 * MIN_FREE_DISK_BYTES),
    }
    kwargs.update(overrides)
    return run_preflight(**kwargs)


def test_run_preflight_passes_when_everything_matches(tmp_path: Path) -> None:
    checks = full(tmp_path)
    assert preflight_ok(checks)
    assert [c.name for c in checks] == [
        "engine",
        "config_id",
        "machine",
        "parallel",
        "frozen_pool",
        "preregistration_7_8",
        "preregistration_l3",
        "plan",
        "disk",
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"current_config": changed("ollama_version", "0.41.0")},
        {"machine": Machine("Mac14,2", "Apple M2")},
        {"current_config": changed("parallel.server_num_parallel", 1)},
        {"disk_usage": lambda _: Usage(0, 0, 0)},
    ],
)
def test_run_preflight_fails_when_any_check_fails(tmp_path: Path, override: dict[str, Any]) -> None:
    assert not preflight_ok(full(tmp_path, **override))


def test_run_preflight_reports_a_missing_reference_as_a_failed_check(tmp_path: Path) -> None:
    checks = full(tmp_path, runs_dir=tmp_path / "empty")
    engine = checks[0]
    assert engine.name == "engine" and not engine.ok and "S0 Dev-A" in engine.detail


def test_plan_check_fails_when_only_the_recorded_hash_is_wrong(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    (tmp_path / "plan.json.sha256").write_text("0" * 64 + "  plan.json\n")
    result = check_plan(path, REPO_ROOT)
    assert not result.ok and "recorded" in result.detail


def test_l3_preregistration_passes_for_the_repo() -> None:
    """Ticket 006 사전 등록이 Ticket 005 본 실행 전에 고정돼 있어야 한다(2026-10-22 사용자 결정)."""
    result = check_preregistration_l3(REPO_ROOT)
    assert result.ok and "e368d9ed" in result.detail


def test_l3_preregistration_fails_when_the_copy_changes_or_is_missing(tmp_path: Path) -> None:
    shutil.copytree(REPO_ROOT / "preregistration", tmp_path / "preregistration")
    copy_path = tmp_path / "preregistration" / "ticket-006-l3-sections.txt"
    copy_path.write_text(copy_path.read_text() + " ")
    assert not check_preregistration_l3(tmp_path).ok
    copy_path.unlink()
    assert not check_preregistration_l3(tmp_path).ok
