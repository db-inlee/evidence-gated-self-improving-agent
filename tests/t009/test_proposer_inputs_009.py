"""Ticket 009 Dev-A 내보내기(사전 등록 2·2.1절): v1 규칙 그대로, 009 Dev-A 케이스만."""

import json
import random
import re
from dataclasses import replace
from pathlib import Path

import pytest

from evidencegate import proposer_inputs as v1_inputs
from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.policy import ScriptedPolicy
from evidencegate.proposer_inputs import ProposerCase, load_proposer_cases
from evidencegate.runner import EpisodeRecord, run_episode, run_suite
from evidencegate.seeds import derive_llm_seed
from evidencegate.skills import Skill
from evidencegate.splits import dev_a as v1_dev_a
from evidencegate.store import EpisodeStore
from evidencegate.t009 import proposer_inputs as pi
from evidencegate.t009.splits import split_cases
from opsworld.generator import L2_SUBTYPES
from opsworld.predicates import Predicate
from opsworld.state import Family, Level

SKILL = Skill.from_text("# Skill\n")
# bad는 근거 부족(evidence-only), dependency_failure·traffic_spike는 처분 오답(wrong)
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
def records(tmp_path_factory: pytest.TempPathFactory) -> list[EpisodeRecord]:
    store = EpisodeStore(tmp_path_factory.mktemp("dev_a_009") / "r.jsonl")
    run_suite(ROLLBACK_AFTER_DEPS, SKILL, pi.dev_a_cases(), repeats=1, store=store)
    return store.records()


@pytest.fixture(scope="module")
def exported(records: list[EpisodeRecord]) -> list[ProposerCase]:
    return pi.export_dev_a_009(records)


def as_text(cases: list[ProposerCase]) -> str:
    return json.dumps([c.to_json() for c in cases], sort_keys=True)


def test_dev_a_cases_are_the_009_split() -> None:
    cases = pi.dev_a_cases()
    assert len(cases) == 60
    assert sorted((c.family.value, c.level.value, c.incident_seed) for c in cases) == split_cases(
        "t009_dev_a"
    )


def test_export_matches_the_v1_rules(records: list[EpisodeRecord]) -> None:
    """실패만, (family, seed, level) 정렬, v1 `_to_case`, v1 섞기 seed. v1과 같은 규칙이다."""
    got = pi.export_dev_a_009(records)
    failures = sorted(
        (r for r in records if not (r.correct_disposition and r.evidence_safe)),
        key=lambda r: (r.key.family, r.key.incident_seed, r.key.level),
    )
    expected = [v1_inputs._to_case(r) for r in failures]
    random.Random(v1_inputs.EXPORT_SHUFFLE_SEED).shuffle(expected)
    assert got == expected
    assert len(got) == 60  # 이 스크립트는 모든 케이스에서 실패
    assert got == pi.export_dev_a_009(records)  # 결정적
    assert all(re.fullmatch(r"case-[0-9a-f]{12}", c.export_id) for c in got)
    assert len({c.export_id for c in got}) == len(got)
    unshuffled = pi.export_dev_a_009(records, shuffle=False)
    assert [c.alert for c in got] != [c.alert for c in unshuffled]
    assert sorted(c.alert for c in got) == sorted(c.alert for c in unshuffled)


def test_successes_are_dropped(records: list[EpisodeRecord]) -> None:
    passed = [replace(r, correct_disposition=True, evidence_safe=True) for r in records[:5]]
    assert len(pi.export_dev_a_009([*passed, *records[5:]])) == 55


def test_no_hidden_values_or_answer_clues(
    records: list[EpisodeRecord], exported: list[ProposerCase]
) -> None:
    text = as_text(exported)
    for word in ("root_cause", "has_bug", "family", "level", "l2_subtype", "incident_id", "seed"):
        assert f'"{word}"' not in text, word
    scrubbed = text
    for predicate in Predicate:
        scrubbed = scrubbed.replace(predicate.value, "")
    for name in [f.value for f in Family] + [n for v in L2_SUBTYPES.values() for n in v]:
        assert name not in scrubbed, name
    assert not re.search(r'"L[12]"', text)
    assert "UNNECESSARY_INTERVENTION" not in text
    for r in records:
        assert str(r.key.incident_seed) not in text
        assert str(derive_llm_seed(r.key.family, r.key.incident_seed, r.key.repeat)) not in text


@pytest.mark.parametrize("split", ["t009_dev_b", "t009_oracle", "t009_calibration"])
def test_other_009_splits_are_rejected(split: str) -> None:
    family, level, seed = split_cases(split)[0]
    record = run_episode(ROLLBACK_AFTER_DEPS, SKILL, Family(family), seed, 0, Level(level))
    with pytest.raises(ValueError, match="not a Ticket 009 Dev-A case"):
        pi.export_dev_a_009([record])


def test_v1_dev_a_records_are_rejected() -> None:
    c = v1_dev_a()[0]
    record = run_episode(ROLLBACK_AFTER_DEPS, SKILL, c.family, c.incident_seed, 0, c.level)
    with pytest.raises(ValueError, match="not a Ticket 009 Dev-A case"):
        pi.export_dev_a_009([record])


def test_only_repeat_0_and_one_run_and_complete(records: list[EpisodeRecord]) -> None:
    c = pi.dev_a_cases()[0]
    repeat_1 = run_episode(ROLLBACK_AFTER_DEPS, SKILL, c.family, c.incident_seed, 1, c.level)
    with pytest.raises(ValueError, match="repeat 0"):
        pi.export_dev_a_009([*records[1:], repeat_1])
    observe = ScriptedPolicy(["observe_only"], name="o")
    other = run_episode(observe, SKILL, c.family, c.incident_seed, 0, c.level)
    with pytest.raises(ValueError, match="mixed"):
        pi.export_dev_a_009([*records[1:], other])
    with pytest.raises(ValueError, match="missing"):
        pi.export_dev_a_009(records[1:])
    with pytest.raises(ValueError, match="duplicate"):
        pi.export_dev_a_009([*records, records[0]])


def case(cd: bool, es: bool, i: int) -> ProposerCase:
    return ProposerCase(f"case-{i:012x}", "Alert", (), "observe_only", cd, es, (), ())


def test_strata_counts() -> None:
    cases = [case(True, False, i) for i in range(3)] + [
        case(False, False, 10 + i) for i in range(5)
    ]
    assert pi.strata(cases) == {"failures": 8, "evidence_only": 3, "wrong": 5}


def test_sampling_check_runs_all_calls_or_reports_the_problem() -> None:
    ok = [case(True, False, i) for i in range(3)] + [case(False, False, 10 + i) for i in range(5)]
    check = pi.check_sampling(ok, calls=80)
    assert check.ok and check.calls == 80
    assert check.distinct_cases_shown == 8
    thin = [case(True, False, 0)] + [case(False, False, 10 + i) for i in range(8)]
    bad = pi.check_sampling(thin, calls=80)
    assert not bad.ok and "stratum too small" in bad.detail
    few = [case(True, False, i) for i in range(2)] + [case(False, False, 10 + i) for i in range(3)]
    assert not pi.check_sampling(few, calls=80).ok  # 6개를 뽑을 수 없음


def write_store(runs: Path, records: list[EpisodeRecord]) -> None:
    store = EpisodeStore.for_run(
        runs,
        suite="t009_dev_a",
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=S0_SKILL_ID,
    )
    for r in records:
        store.append(
            replace(
                r,
                key=replace(
                    r.key,
                    policy_id=POLICY_ID,
                    config_id=CONFIG_ID,
                    tool_spec_id=TOOL_SPEC_ID,
                    skill_id=S0_SKILL_ID,
                ),
            )
        )


def test_main_writes_the_export_and_refuses_a_different_file(
    records: list[EpisodeRecord], tmp_path: Path
) -> None:
    write_store(tmp_path / "runs", records)
    out = tmp_path / "inputs.json"
    args = ["--runs-dir", str(tmp_path / "runs"), "--out", str(out)]
    assert pi.main(args) == 0
    loaded = load_proposer_cases(out)
    assert len(loaded) == 60
    digest = out.with_name(out.name + ".sha256").read_text().split()
    assert digest[1] == out.name and len(digest[0]) == 64
    assert pi.main(args) == 0  # 같은 바이트면 그대로
    out.write_text("[]\n")
    assert pi.main(args) == 1


def test_main_stops_when_a_stratum_is_thin(records: list[EpisodeRecord], tmp_path: Path) -> None:
    # bad_deployment(근거 부족)를 성공으로 바꿔 evidence-only 층을 비운다
    fixed = [
        replace(r, correct_disposition=True, evidence_safe=True)
        if r.key.family == Family.BAD_DEPLOYMENT.value
        else r
        for r in records
    ]
    write_store(tmp_path / "runs", fixed)
    out = tmp_path / "inputs.json"
    assert pi.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out)]) == 1


def test_default_output_is_outside_the_evaluation_paths() -> None:
    path = pi.OUT_PATH.as_posix()
    assert path.startswith("runs/proposer_inputs-009/")
    for word in ("dev_b", "oracle", "calibration"):
        assert word not in path
