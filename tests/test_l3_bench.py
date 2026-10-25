"""Ticket 006 L3-check 준비(§3, §6 순서 4): L3 split, 확인 블록, 실행 전 점검, 운영 지표 요약.

L3-check는 S0 x 30개(반복 0)이고 운영 지표만 본다. CD·ES는 어디에도 내지 않는다.
"""

import copy
import dataclasses
import io
import json
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from evidencegate import eval_constants as k
from evidencegate.bench_runner import block_store
from evidencegate.eval_data import split_cases
from evidencegate.l3_bench import (
    l3_check_block,
    l3_preflight,
    l3_split_cases,
    main,
    ops_summary,
    render_ops,
)
from evidencegate.machine import Machine
from evidencegate.policy import PolicyTrace, StopReason, TurnRecord
from evidencegate.preflight import Check, preflight_ok
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.suites import CALIBRATION_SEED_OFFSET
from opsworld.state import Family
from test_bench_runner import FakePolicy, Probe
from test_preflight import REFERENCE, s0_dev_a_store

REPO_ROOT = Path(__file__).resolve().parents[1]
M3 = Machine("Mac15,3", "Apple M3")
Usage = namedtuple("Usage", "total used free")


# ---------------------------------------------------------------------------
# L3 split (§3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "per_family", "seeds"),
    [
        ("l3_check", 10, range(410_000, 410_010)),
        ("l3_oracle", 100, range(400_000, 400_100)),
    ],
)
def test_l3_splits_match_the_ticket(split: str, per_family: int, seeds: range) -> None:
    cases = l3_split_cases(split)
    assert len(cases) == 3 * per_family
    assert {c[1] for c in cases} == {"L3"}
    for family in Family:
        assert sorted(c[2] for c in cases if c[0] == family.value) == list(seeds)
    assert list(cases) == sorted(cases)  # 칸(family), incident_seed 순


def test_l3_split_seeds_do_not_overlap_other_splits() -> None:
    l3 = {c[2] for s in ("l3_check", "l3_oracle") for c in l3_split_cases(s)}
    others = {c[2] for s in k.EVAL_SPLITS for c in split_cases(s)}
    assert not l3 & others
    assert not any(CALIBRATION_SEED_OFFSET <= s < CALIBRATION_SEED_OFFSET + 100_000 for s in l3)
    assert not {c[2] for c in l3_split_cases("l3_check")} & {
        c[2] for c in l3_split_cases("l3_oracle")
    }


def test_unknown_l3_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown L3 split"):
        l3_split_cases("oracle")


def test_ticket_005_splits_stay_l1_l2() -> None:
    """L3 split은 Ticket 005 분석의 split 목록에 들어가지 않는다(고정 결정 6)."""
    assert k.EVAL_SPLITS == ("dev_a", "dev_b", "oracle")
    assert not set(k.L3_SPLITS) & set(k.EVAL_SPLITS)


# ---------------------------------------------------------------------------
# L3-check 블록
# ---------------------------------------------------------------------------


def test_l3_check_block_is_s0_repeat_0_over_the_30_check_cases() -> None:
    block = l3_check_block(REPO_ROOT)
    assert (block.name, block.group, block.split, block.repeat) == ("S0", "s0", "l3_check", 0)
    assert block.skill_id == k.S0_SKILL_ID
    assert block.skill_path == "skills/S0/SKILL.md"
    assert block.cases == l3_split_cases("l3_check")


def test_l3_check_block_refuses_a_changed_s0(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "S0" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text((REPO_ROOT / "skills" / "S0" / "SKILL.md").read_text() + "\nextra\n")
    with pytest.raises(ValueError, match="S0"):
        l3_check_block(tmp_path)


# ---------------------------------------------------------------------------
# 실행 전 점검: Ticket 005와 같되 Ticket 005 계획 파일 점검은 없다
# ---------------------------------------------------------------------------


def preflight(tmp_path: Path, **overrides: Any) -> list[Check]:
    runs = tmp_path / "runs"
    s0_dev_a_store(runs, [REFERENCE])
    kwargs: dict[str, Any] = {
        "root": REPO_ROOT,
        "runs_dir": runs,
        "current_config": copy.deepcopy(REFERENCE),
        "machine": M3,
        "disk_usage": lambda _: Usage(0, 0, 10 * k.MIN_FREE_DISK_BYTES),
    }
    kwargs.update(overrides)
    return l3_preflight(**kwargs)


def test_l3_preflight_passes_and_includes_the_l3_preregistration(tmp_path: Path) -> None:
    checks = preflight(tmp_path)
    assert preflight_ok(checks), checks
    assert [c.name for c in checks] == [
        "engine",
        "config_id",
        "machine",
        "parallel",
        "frozen_pool",
        "preregistration_7_8",
        "preregistration_l3",
        "disk",
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"machine": Machine("Mac14,2", "Apple M2")},
        {"current_config": {**REFERENCE, "ollama_version": "0.41.0"}},
        {"disk_usage": lambda _: Usage(0, 0, 0)},
    ],
)
def test_l3_preflight_fails_when_any_check_fails(tmp_path: Path, override: dict[str, Any]) -> None:
    assert not preflight_ok(preflight(tmp_path, **override))


def test_l3_preflight_fails_when_the_l3_preregistration_copy_changes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for part in ("preregistration", "skills", "docs"):
        if (REPO_ROOT / part).exists():
            shutil.copytree(REPO_ROOT / part, root / part)
    copy_file = root / "preregistration" / "ticket-006-l3-sections.txt"
    copy_file.write_bytes(copy_file.read_bytes() + b" ")
    checks = {c.name: c for c in preflight(tmp_path, root=root)}
    assert not checks["preregistration_l3"].ok


# ---------------------------------------------------------------------------
# 운영 지표 요약: CD·ES 없음
# ---------------------------------------------------------------------------


def rec(
    seed: int,
    *,
    started: str,
    duration: float,
    turns: int = 3,
    format_errors: int = 0,
    action: str | None = "observe_only",
    stop: StopReason = StopReason.ACTION,
) -> EpisodeRecord:
    turn = TurnRecord("get_service_metrics", {}, "{}", False)
    return EpisodeRecord(
        key=EpisodeKey("p", "c", "t", "s", "bad_deployment", seed, 0, "L3"),
        policy_config={},
        llm_seed=0,
        correct_disposition=True,
        evidence_safe=False,
        action=action,
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((turn,) * turns, format_errors, stop),
        started_at=started,
        duration_s=duration,
    )


def test_ops_summary_counts_operational_metrics_only() -> None:
    records = [
        rec(1, started="2026-10-25T00:00:00+00:00", duration=10.0, turns=4),
        rec(2, started="2026-10-25T00:00:00+00:00", duration=12.0, turns=6, format_errors=2),
        rec(
            3,
            started="2026-10-25T00:00:10+00:00",
            duration=8.0,
            turns=8,
            action=None,
            stop=StopReason.MAX_STEPS,
        ),
        rec(4, started="2026-10-25T00:00:12+00:00", duration=8.0, turns=2),
    ]
    s = ops_summary(records, planned=30, server_errors=1)
    assert (s.episodes, s.planned) == (4, 30)
    assert (s.format_error_episodes, s.format_errors) == (1, 2)
    assert s.no_disposition == 1
    assert s.stop_reasons == {"action": 3, "max_steps": 1}
    assert s.mean_tool_calls == 5.0 and s.max_tool_calls == 8
    assert s.mean_duration_s == 9.5
    assert s.wall_s == 20.0  # 첫 시작 00:00:00 ~ 마지막 끝 00:00:20
    assert s.s_per_episode == 5.0
    assert s.main_run_estimate_s == 5.0 * k.L3_MAIN_EPISODES
    assert s.server_errors == 1


def test_ops_summary_and_its_rendering_have_no_cd_or_es() -> None:
    records = [rec(1, started="2026-10-25T00:00:00+00:00", duration=10.0)]
    s = ops_summary(records, planned=30, server_errors=0)
    text = render_ops(s, reference=None)
    fields = set(vars(s))
    assert not fields & {"correct_disposition", "evidence_safe", "cd", "es", "actions"}
    for word in ("CD", "ES", "correct", "evidence", "observe_only", "rollback", "scale"):
        assert word not in text


def test_ops_summary_needs_records() -> None:
    with pytest.raises(ValueError, match="no L3-check records"):
        ops_summary([], planned=30, server_errors=0)


def test_rendering_shows_the_main_run_estimate_and_the_reference() -> None:
    records = [
        rec(1, started="2026-10-25T00:00:00+00:00", duration=6.0, turns=4),
        rec(2, started="2026-10-25T00:00:06+00:00", duration=6.0, turns=4),
    ]
    s = ops_summary(records, planned=2, server_errors=0)
    reference = ops_summary(
        [rec(9, started="2026-10-20T00:00:00+00:00", duration=5.0, turns=3)],
        planned=1,
        server_errors=0,
    )
    text = render_ops(s, reference=reference)
    assert "7200 episodes" in text and "6.0 s/episode" in text and "12h00m" in text
    assert "S0 L1/L2 Oracle r0" in text and "mean tool calls 3.00" in text


# ---------------------------------------------------------------------------
# 명령: check는 Ticket 006 세션 로그에, Ticket 005 로그는 건드리지 않음
# ---------------------------------------------------------------------------


def cli(tmp_path: Path, args: list[str], *, checks_ok: bool = True) -> tuple[int, str]:
    out = io.StringIO()
    code = main(
        [*args, "--runs-dir", str(tmp_path / "runs")],
        policy_factory=FakePolicy,
        preflight=lambda policy: [Check("engine", checks_ok, "fake")],
        probe=lambda: Probe(True),
        status_policy=FakePolicy(),
        out=out,
        err=out,
    )
    return code, out.getvalue()


def test_check_runs_the_30_cases_once_and_logs_to_ticket_006(tmp_path: Path) -> None:
    code, text = cli(tmp_path, ["check"])
    assert code == 0, text
    store = block_store(tmp_path / "runs", FakePolicy(), l3_check_block(REPO_ROOT))
    records = store.records()
    assert sorted((r.key.family, r.key.level, r.key.incident_seed) for r in records) == list(
        l3_split_cases("l3_check")
    )
    assert store.path.parts[-4] == "l3_check"
    assert (tmp_path / "runs" / k.L3_SESSION_LOG).exists()
    assert not (tmp_path / "runs" / k.SESSION_LOG).exists()
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "runs" / k.L3_SESSION_LOG).read_text().splitlines()
    ]
    assert events[0] == "probe" and "session_end" in events
    assert "30/30 episodes" in text and "estimate" in text
    for word in ("CD", "ES ", "correct_disposition", "evidence_safe"):
        assert word not in text

    code, again = cli(tmp_path, ["check"])
    assert code == 0 and "0 episodes" in again
    assert len(store.records()) == 30


def test_check_runs_nothing_when_preflight_fails(tmp_path: Path) -> None:
    code, text = cli(tmp_path, ["check"], checks_ok=False)
    assert code == 1 and "preflight failed" in text
    assert not (tmp_path / "runs" / "l3_check").exists()
    assert (tmp_path / "runs" / k.L3_SESSION_LOG).exists()


def test_summary_without_records_says_so(tmp_path: Path) -> None:
    code, text = cli(tmp_path, ["summary"])
    assert code == 1 and "no L3-check records" in text


def test_summary_after_check(tmp_path: Path) -> None:
    cli(tmp_path, ["check"])
    code, text = cli(tmp_path, ["summary"])
    assert code == 0 and "30/30 episodes" in text


def test_summary_counts_only_repeat_0_of_the_check_cases(tmp_path: Path) -> None:
    """같은 저장소에 다른 반복이나 다른 케이스가 있어도 세지 않는다."""
    cli(tmp_path, ["check"])
    store = block_store(tmp_path / "runs", FakePolicy(), l3_check_block(REPO_ROOT))
    first = store.records()[0]
    store.append(dataclasses.replace(first, key=dataclasses.replace(first.key, repeat=1)))
    store.append(dataclasses.replace(first, key=dataclasses.replace(first.key, incident_seed=7)))
    code, text = cli(tmp_path, ["summary"])
    assert code == 0 and "30/30 episodes" in text
