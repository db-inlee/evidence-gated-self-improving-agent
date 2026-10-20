"""평가 데이터 읽기·검증과 짝 표 (Ticket 005 §1 A, §8-1·5·6). 합성 레코드만 쓴다."""

from collections.abc import Iterable

import pytest

from evidencegate.eval_data import (
    CELLS,
    cell_of,
    check_run_conditions,
    majority,
    outcomes,
    paired_gate,
    paired_oracle,
    repeat_outcomes,
    split_cases,
    split_of,
)
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord

CD = "correct_disposition"
ES = "evidence_safe"


def rec(
    skill: str,
    case: tuple[str, str, int],
    repeat: int = 0,
    *,
    cd: bool = True,
    es: bool = True,
    config: str = "7c66ee5bf13c",
    tools: str = "e9b4996f7529",
    policy: str = "ollama:gpt-oss:20b",
    stop: StopReason = StopReason.ACTION,
) -> EpisodeRecord:
    family, level, seed = case
    return EpisodeRecord(
        key=EpisodeKey(policy, config, tools, skill, family, seed, repeat, level),
        policy_config={},
        llm_seed=0,
        correct_disposition=cd,
        evidence_safe=es,
        action="observe_only" if stop is StopReason.ACTION else None,
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, stop),
        started_at="2026-10-21T00:00:00+00:00",
        duration_s=1.0,
    )


def full(
    skill: str, split: str, repeat: int = 0, *, cd: Iterable[bool] | None = None
) -> list[EpisodeRecord]:
    cases = split_cases(split)
    values = list(cd) if cd is not None else [True] * len(cases)
    return [rec(skill, c, repeat, cd=v, es=not v) for c, v in zip(cases, values, strict=True)]


# ---------------------------------------------------------------------------
# 케이스·칸·split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("split", "size"), [("dev_a", 60), ("dev_b", 210), ("oracle", 300)])
def test_split_cases_have_the_preregistered_sizes_and_order(split: str, size: int) -> None:
    cases = split_cases(split)
    assert len(cases) == size == len(set(cases))
    assert cases == sorted(cases)  # (family, level, incident_seed) 정렬(§8-1)
    per_cell = size // 6
    assert all(sum(1 for c in cases if cell_of(c) == cell) == per_cell for cell in CELLS)


def test_cells_are_family_by_level_in_sorted_order() -> None:
    assert CELLS == (
        ("bad_deployment", "L1"),
        ("bad_deployment", "L2"),
        ("dependency_failure", "L1"),
        ("dependency_failure", "L2"),
        ("traffic_spike", "L1"),
        ("traffic_spike", "L2"),
    )


def test_split_of_each_case() -> None:
    assert split_of(("bad_deployment", "L1", 100_003)) == "dev_a"
    assert split_of(("traffic_spike", "L2", 200_034)) == "dev_b"
    assert split_of(("dependency_failure", "L1", 300_049)) == "oracle"


@pytest.mark.parametrize(
    "case",
    [("bad_deployment", "L1", 900_000), ("bad_deployment", "L1", 100_010), ("nope", "L1", 100_000)],
)
def test_case_outside_every_split_is_an_error(case: tuple[str, str, int]) -> None:
    with pytest.raises(ValueError, match="split"):
        split_of(case)


def test_unknown_split_is_an_error() -> None:
    with pytest.raises(ValueError, match="split"):
        split_cases("dev_c")


# ---------------------------------------------------------------------------
# 실행 조건
# ---------------------------------------------------------------------------


def test_run_conditions_pass_for_one_condition() -> None:
    check_run_conditions(full("cand", "dev_a") + full("s0", "dev_a"))


@pytest.mark.parametrize("field", ["config", "tools", "policy"])
def test_mixed_run_conditions_are_an_error(field: str) -> None:
    records = full("cand", "dev_a")
    records.append(rec("s0", split_cases("dev_a")[0], **{field: "other"}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mixed"):
        check_run_conditions(records)


def test_expected_run_condition_is_enforced() -> None:
    records = full("cand", "dev_a")
    with pytest.raises(ValueError, match="config"):
        check_run_conditions(records, config_id="24bf38dea859")
    check_run_conditions(records, config_id="7c66ee5bf13c", tool_spec_id="e9b4996f7529")


def test_no_records_is_an_error() -> None:
    with pytest.raises(ValueError, match="no records"):
        check_run_conditions([])


# ---------------------------------------------------------------------------
# 결과 표: 누락·중복이면 멈춘다
# ---------------------------------------------------------------------------


def test_outcomes_cover_the_split_in_case_order() -> None:
    cases = split_cases("dev_a")
    values = [i % 2 == 0 for i in range(60)]
    table = outcomes(
        full("cand", "dev_a", cd=values), skill_id="cand", split="dev_a", metric=CD, repeat=0
    )
    assert list(table) == cases
    assert list(table.values()) == [int(v) for v in values]


def test_outcomes_for_es() -> None:
    values = [i % 3 == 0 for i in range(60)]
    table = outcomes(
        full("cand", "dev_a", cd=values), skill_id="cand", split="dev_a", metric=ES, repeat=0
    )
    assert list(table.values()) == [int(not v) for v in values]


def test_outcomes_ignore_other_skills_repeats_and_splits() -> None:
    records = full("cand", "dev_a") + full("s0", "dev_a") + full("cand", "dev_a", repeat=1)
    records += full("cand", "dev_b")
    table = outcomes(records, skill_id="cand", split="dev_a", metric=CD, repeat=0)
    assert len(table) == 60


def test_missing_case_is_an_error() -> None:
    records = full("cand", "dev_a")[1:]
    with pytest.raises(ValueError, match="missing"):
        outcomes(records, skill_id="cand", split="dev_a", metric=CD, repeat=0)


def test_duplicate_case_is_an_error() -> None:
    records = full("cand", "dev_a")
    records.append(records[5])
    with pytest.raises(ValueError, match="duplicate"):
        outcomes(records, skill_id="cand", split="dev_a", metric=CD, repeat=0)


def test_unknown_metric_is_an_error() -> None:
    with pytest.raises(ValueError, match="metric"):
        outcomes(full("cand", "dev_a"), skill_id="cand", split="dev_a", metric="cbu", repeat=0)


def test_episode_without_disposition_counts_as_recorded() -> None:
    """§8-6: 처분 없이 끝난 에피소드는 기록된 값(CD = ES = 거짓) 그대로."""
    records = full("cand", "dev_a")
    first = split_cases("dev_a")[0]
    records[0] = rec("cand", first, cd=False, es=False, stop=StopReason.MAX_STEPS)
    table = outcomes(records, skill_id="cand", split="dev_a", metric=CD, repeat=0)
    assert table[first] == 0


# ---------------------------------------------------------------------------
# 다수결과 반복
# ---------------------------------------------------------------------------


def test_majority_of_five() -> None:
    case = ("bad_deployment", "L1", 300_000)
    table = {r: {case: v} for r, v in enumerate([1, 1, 0, 0, 1])}
    assert majority(table) == {case: 1}
    table = {r: {case: v} for r, v in enumerate([1, 0, 0, 0, 1])}
    assert majority(table) == {case: 0}


def test_majority_needs_an_odd_number_of_repeats() -> None:
    case = ("bad_deployment", "L1", 300_000)
    with pytest.raises(ValueError, match="odd"):
        majority({0: {case: 1}, 1: {case: 0}})


def test_majority_needs_the_same_cases_in_every_repeat() -> None:
    a, b = ("bad_deployment", "L1", 300_000), ("bad_deployment", "L1", 300_001)
    with pytest.raises(ValueError, match="same cases"):
        majority({0: {a: 1}, 1: {a: 1}, 2: {b: 1}})


def test_repeat_outcomes_need_every_listed_repeat() -> None:
    records = [r for rep in range(4) for r in full("s0", "oracle", repeat=rep)]
    with pytest.raises(ValueError, match="missing"):
        repeat_outcomes(records, skill_id="s0", split="oracle", metric=CD, repeats=(0, 1, 2, 3, 4))
    table = repeat_outcomes(
        records + full("s0", "oracle", repeat=4),
        skill_id="s0",
        split="oracle",
        metric=CD,
        repeats=(0, 1, 2, 3, 4),
    )
    assert sorted(table) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 짝 표
# ---------------------------------------------------------------------------


def test_paired_gate_uses_repeat_0_for_both() -> None:
    cases = split_cases("dev_b")
    cand = [i % 4 != 0 for i in range(210)]
    base = [i % 2 == 0 for i in range(210)]
    records = full("cand", "dev_b", cd=cand) + full("s0", "dev_b", cd=base)
    records += full("s0", "dev_b", repeat=1, cd=[False] * 210)  # 반복 1은 쓰지 않는다
    table = paired_gate(
        records, candidate_skill_id="cand", s0_skill_id="s0", split="dev_b", metric=CD
    )
    assert table.cases == tuple(cases)
    assert table.cells == tuple(cell_of(c) for c in cases)
    assert table.candidate == tuple(int(v) for v in cand)
    assert table.baseline == tuple(int(v) for v in base)
    assert table.deltas == tuple(int(a) - int(b) for a, b in zip(cand, base, strict=True))


def test_paired_gate_rejects_oracle_split() -> None:
    records = full("cand", "oracle") + full("s0", "oracle")
    with pytest.raises(ValueError, match="oracle"):
        paired_gate(records, candidate_skill_id="cand", s0_skill_id="s0", split="oracle", metric=CD)


def test_paired_gate_stops_when_s0_misses_a_case() -> None:
    records = full("cand", "dev_a") + full("s0", "dev_a")[:-1]
    with pytest.raises(ValueError, match="missing"):
        paired_gate(records, candidate_skill_id="cand", s0_skill_id="s0", split="dev_a", metric=CD)


def test_paired_oracle_uses_candidate_repeat_0_and_s0_majority() -> None:
    """§8-5: 후보는 반복 0, 기준값은 S0 반복 0~4 다수결."""
    n = 300
    s0 = [full("s0", "oracle", repeat=r, cd=[(i + r) % 5 < 3 for i in range(n)]) for r in range(5)]
    cand = full("cand", "oracle", cd=[i % 2 == 0 for i in range(n)])
    cand += full("cand", "oracle", repeat=1, cd=[False] * n)  # 반복 1은 쓰지 않는다
    records = cand + [r for rep in s0 for r in rep]
    table = paired_oracle(records, candidate_skill_id="cand", s0_skill_id="s0", metric=CD)
    assert table.candidate == tuple(int(i % 2 == 0) for i in range(n))
    # 케이스 i의 다섯 반복 중 (i + r) % 5 < 3인 것은 정확히 3개 → 다수결 1
    assert table.baseline == (1,) * n
    assert table.split == "oracle"


def test_paired_oracle_needs_all_five_s0_repeats() -> None:
    records = full("cand", "oracle") + [
        r for rep in range(4) for r in full("s0", "oracle", repeat=rep)
    ]
    with pytest.raises(ValueError, match="missing"):
        paired_oracle(records, candidate_skill_id="cand", s0_skill_id="s0", metric=CD)


def test_cell_indices_group_cases_by_cell() -> None:
    records = full("cand", "dev_b") + full("s0", "dev_b")
    table = paired_gate(
        records, candidate_skill_id="cand", s0_skill_id="s0", split="dev_b", metric=CD
    )
    groups = table.cell_indices()
    assert [len(g) for g in groups] == [35] * 6
    assert sorted(i for g in groups for i in g) == list(range(210))
    for cell, group in zip(CELLS, groups, strict=True):
        assert all(table.cells[i] == cell for i in group)
