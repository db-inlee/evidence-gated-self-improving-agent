"""Dev-A 내보내기(Ticket 004 §3): proposer가 볼 실패 사례만, 숨은 값·정답 단서 없이."""

import json
import re
from pathlib import Path

import pytest

from evidencegate.policy import ScriptedPolicy
from evidencegate.proposer_inputs import (
    ProposerCase,
    export_dev_a,
    load_proposer_cases,
    save_proposer_cases,
)
from evidencegate.runner import EpisodeRecord, run_episode, run_suite
from evidencegate.seeds import derive_llm_seed
from evidencegate.skills import Skill
from evidencegate.splits import dev_a, dev_b
from evidencegate.store import EpisodeStore
from opsworld.generator import L2_SUBTYPES
from opsworld.predicates import Predicate
from opsworld.state import Family

SKILL = Skill.from_text("# Skill\n")
# S0처럼 의존성까지 보고 롤백한다: bad는 근거 부족, dependency_failure는 처분 오답
# (UNNECESSARY_INTERVENTION 포함), traffic_spike는 처분 오답이 된다.
ROLLBACK_AFTER_DEPS = ScriptedPolicy(
    [
        "get_service_metrics",
        "get_recent_deployments",
        "get_dependency_health",
        "rollback_deployment",
    ],
    name="rollback-after-deps",
)


@pytest.fixture(scope="module")
def dev_a_records(tmp_path_factory: pytest.TempPathFactory) -> list[EpisodeRecord]:
    store = EpisodeStore(tmp_path_factory.mktemp("dev_a") / "r.jsonl")
    run_suite(ROLLBACK_AFTER_DEPS, SKILL, dev_a(), repeats=1, store=store)
    return store.records()


@pytest.fixture(scope="module")
def exported(dev_a_records: list[EpisodeRecord]) -> list[ProposerCase]:
    return export_dev_a(dev_a_records)


def as_text(cases: list[ProposerCase]) -> str:
    return json.dumps([c.to_json() for c in cases], sort_keys=True)


def test_only_failures_are_exported(
    dev_a_records: list[EpisodeRecord], exported: list[ProposerCase]
) -> None:
    failures = [r for r in dev_a_records if not (r.correct_disposition and r.evidence_safe)]
    assert len(exported) == len(failures) == 60  # 이 스크립트는 모든 케이스에서 실패
    assert all(not (c.correct_disposition and c.evidence_safe) for c in exported)


def test_successes_are_dropped() -> None:
    safe_observe = ScriptedPolicy(
        ["get_service_metrics", "get_dependency_health", "observe_only"], name="safe"
    )
    records = [
        run_episode(safe_observe, SKILL, c.family, c.incident_seed, 0, c.level)
        for c in dev_a()
        if c.family is Family.DEPENDENCY_FAILURE
    ]
    assert all(r.evidence_safe for r in records)
    assert export_dev_a(records) == []


def test_exported_fields(exported: list[ProposerCase]) -> None:
    case = exported[0]
    assert set(case.to_json()) == {
        "export_id",
        "alert",
        "steps",
        "action",
        "correct_disposition",
        "evidence_safe",
        "failure_labels",
        "missing_predicates",
    }
    assert case.alert.startswith("Alert: service ")
    assert [s["tool"] for s in case.steps] == [
        "get_service_metrics",
        "get_recent_deployments",
        "get_dependency_health",
        "rollback_deployment",
    ]
    assert isinstance(case.steps[0]["output"], dict)
    assert "p95_ms" in case.steps[0]["output"]
    assert case.steps[-1]["output"] == {"status": "recorded"}


def test_unnecessary_intervention_label_is_removed(
    dev_a_records: list[EpisodeRecord], exported: list[ProposerCase]
) -> None:
    assert any("UNNECESSARY_INTERVENTION" in r.failure_labels for r in dev_a_records)
    assert "UNNECESSARY_INTERVENTION" not in as_text(exported)
    assert any("WRONG_INTERVENTION" in c.failure_labels for c in exported)


def test_no_hidden_values_or_answer_clues(
    dev_a_records: list[EpisodeRecord], exported: list[ProposerCase]
) -> None:
    text = as_text(exported)
    for word in ("root_cause", "has_bug", "family", "level", "l2_subtype", "incident_id", "seed"):
        assert f'"{word}"' not in text, word
    # family 이름은 술어 이름(dependency_failure_confirmed) 안에만 나올 수 있다
    scrubbed = text
    for predicate in Predicate:
        scrubbed = scrubbed.replace(predicate.value, "")
    for name in [f.value for f in Family] + [n for v in L2_SUBTYPES.values() for n in v]:
        assert name not in scrubbed, name
    assert not re.search(r'"L[12]"', text)
    for r in dev_a_records:
        assert str(r.key.incident_seed) not in text
        assert str(derive_llm_seed(r.key.family, r.key.incident_seed, r.key.repeat)) not in text


def test_export_ids_are_opaque_unique_and_order_is_shuffled(
    dev_a_records: list[EpisodeRecord], exported: list[ProposerCase]
) -> None:
    ids = [c.export_id for c in exported]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"case-[0-9a-f]{12}", i) for i in ids)
    original_alerts = [ex.alert for ex in export_dev_a(dev_a_records, shuffle=False)]
    assert [c.alert for c in exported] != original_alerts
    assert sorted(c.alert for c in exported) == sorted(original_alerts)


def test_export_is_deterministic(dev_a_records: list[EpisodeRecord]) -> None:
    assert export_dev_a(dev_a_records) == export_dev_a(dev_a_records)


def test_non_dev_a_records_are_rejected(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "b.jsonl")
    run_suite(ROLLBACK_AFTER_DEPS, SKILL, dev_b()[:3], repeats=1, store=store)
    with pytest.raises(ValueError, match="Dev-A"):
        export_dev_a(store.records())


def test_only_repeat_0_is_accepted(dev_a_records: list[EpisodeRecord]) -> None:
    case = dev_a()[0]
    repeat_1 = run_episode(
        ROLLBACK_AFTER_DEPS, SKILL, case.family, case.incident_seed, 1, case.level
    )
    with pytest.raises(ValueError, match="repeat 0"):
        export_dev_a([*dev_a_records, repeat_1])


def test_mixed_runs_are_rejected(dev_a_records: list[EpisodeRecord]) -> None:
    other = ScriptedPolicy(["observe_only"], name="other")
    case = dev_a()[1]
    extra = run_episode(other, SKILL, case.family, case.incident_seed, 0, case.level)
    with pytest.raises(ValueError, match="mixed"):
        export_dev_a([*dev_a_records, extra])


def test_truncated_tool_output_is_an_error(dev_a_records: list[EpisodeRecord]) -> None:
    from dataclasses import replace

    record = dev_a_records[0]
    turns = list(record.trace.turns)
    turns[0] = replace(turns[0], result_summary=turns[0].result_summary[:40] + "...")
    broken = replace(record, trace=replace(record.trace, turns=tuple(turns)))
    with pytest.raises(ValueError, match="truncated"):
        export_dev_a([broken, *dev_a_records[1:]])


def test_save_and_load_round_trip(exported: list[ProposerCase], tmp_path: Path) -> None:
    path = tmp_path / "proposer_inputs" / "dev_a.json"
    save_proposer_cases(exported, path)
    assert load_proposer_cases(path) == exported
