"""proposer 후보에만 적용하는 §5 검사 세 가지: 줄 수, 구조, 용어 (Ticket 004 §5, 2026-10-16)."""

from pathlib import Path

import pytest

from evidencegate.candidate_checks import (
    changed_lines,
    check_proposer_candidate,
    validate_proposer_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
S0 = (REPO_ROOT / "skills" / "S0" / "SKILL.md").read_text(encoding="utf-8")
CONTROLS = {
    p.parent.name: p.read_text(encoding="utf-8")
    for p in sorted((REPO_ROOT / "skills" / "controls").glob("*/SKILL.md"))
}

# 첫 실측 후보(2026-10-15, 풀에 넣지 않음). 검사안을 정할 때 본 실제 출력이다.
PROBE_GPT_5_4 = """# Incident Response Skill

1. Confirm the reported service degradation using `get_service_metrics`.
2. Check for recent deployments using `get_recent_deployments`.
3. Check dependency health using `get_dependency_health`.
4. Check traffic using `get_traffic_metrics` before taking any disposition action.
5. If any dependency is degraded and its issue began before or around the service regression, \
do not change the service; use `observe_only`.
6. If a recent deployment exists, verify that the deployment happened before the regression began.
7. If traffic is elevated versus baseline and CPU is high, treat traffic as a likely cause and use \
`scale_service`.
8. If traffic is not elevated, dependencies are healthy, and a recent deployment happened before \
the regression began, use `rollback_deployment`.
9. If traffic is not elevated and there is no qualifying recent deployment, use `observe_only`.
10. If the evidence is mixed or does not clearly support rollback or scaling, use `observe_only`.

Evidence card:
- Required before `rollback_deployment`: service degradation confirmed; dependency health checked \
and not indicating an upstream cause; traffic checked and ruled out as the primary cause; a recent \
deployment confirmed to have occurred before the regression.
- Required before `scale_service`: service degradation confirmed; traffic checked and shown \
elevated versus baseline; CPU usage reviewed as supporting evidence; dependency health checked.
- Required before `observe_only`: dependency issue explains the regression, or available evidence \
does not clearly support rollback or scaling.
"""
PROBE_GPT_5_4_MINI = """# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. Check dependency health.
4. Only roll back a deployment when a recent deployment exists, the deployment clearly happened \
before the regression began, dependencies are healthy, and traffic changes do not explain the \
regression.
5. If a dependency is degraded, do not change the service; observe and keep monitoring.
6. If traffic is the likely cause or traffic has not yet been ruled out, do not roll back; \
inspect traffic metrics instead.
7. Otherwise inspect resource usage and traffic.
"""

S0_STEPS = [line for line in S0.splitlines() if line[:1].isdigit()]


def skill(*steps: str) -> str:
    body = "\n".join(f"{i}. {s.split('. ', 1)[1]}" for i, s in enumerate(steps, start=1))
    return f"# Incident Response Skill\n\n{body}\n"


NEW_STEP = "0. Check traffic before rolling back."


# ---------------------------------------------------------------------------
# 통제 후보와 실측 후보 (보고서 2026-10-15-07의 표)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in CONTROLS if n != "neutral-rephrase"])
def test_controls_except_neutral_pass_all_three(name: str) -> None:
    assert check_proposer_candidate(CONTROLS[name], S0) == ()


def test_neutral_rephrase_fails_only_the_line_count() -> None:
    """설계상 모든 줄을 바꾼 통제 후보. 그래서 세 검사는 proposer 후보에만 적용한다."""
    assert check_proposer_candidate(CONTROLS["neutral-rephrase"], S0) == ("line_count",)
    assert changed_lines(CONTROLS["neutral-rephrase"], S0) == (6, 6)


def test_control_line_counts_match_the_report() -> None:
    counts = {name: changed_lines(text, S0) for name, text in CONTROLS.items()}
    assert counts == {
        "mixed-scale-on-traffic": (1, 0),
        "neg-rollback-any": (1, 2),
        "neg-skip-metrics": (0, 1),
        "neg-timing-shortcut": (1, 0),
        "neutral-rephrase": (6, 6),
        "pos-timing-only": (1, 1),
        "pos-traffic-check": (1, 1),
    }


def test_first_probe_gpt_5_4_fails_all_three() -> None:
    assert check_proposer_candidate(PROBE_GPT_5_4, S0) == ("line_count", "structure", "terms")


def test_first_probe_mini_fails_the_line_count() -> None:
    assert changed_lines(PROBE_GPT_5_4_MINI, S0) == (2, 1)
    assert check_proposer_candidate(PROBE_GPT_5_4_MINI, S0) == ("line_count",)


# ---------------------------------------------------------------------------
# 줄 수: 추가·수정 ≤ 1, 삭제 ≤ 2
# ---------------------------------------------------------------------------


def test_unchanged_skill_has_no_changed_lines() -> None:
    assert changed_lines(S0, S0) == (0, 0)


def test_renumbering_and_punctuation_are_not_changes() -> None:
    text = S0.replace("1. Confirm", "1) Confirm").replace(
        "roll the deployment back.", "roll the deployment back"
    )
    assert changed_lines(text, S0) == (0, 0)


def test_one_added_step_passes_two_fail() -> None:
    one = skill(*S0_STEPS[:3], NEW_STEP, *S0_STEPS[3:])
    two = skill(*S0_STEPS[:3], NEW_STEP, "0. Check traffic again.", *S0_STEPS[3:])
    assert changed_lines(one, S0) == (1, 0)
    assert check_proposer_candidate(one, S0) == ()
    assert check_proposer_candidate(two, S0) == ("line_count",)


def test_two_deleted_steps_pass_three_fail() -> None:
    two = skill(*S0_STEPS[:2], *S0_STEPS[4:])
    three = skill(*S0_STEPS[:2], *S0_STEPS[5:])
    assert changed_lines(two, S0) == (0, 2)
    assert check_proposer_candidate(two, S0) == ()
    assert changed_lines(three, S0) == (0, 3)
    assert check_proposer_candidate(three, S0) == ("line_count",)


def test_one_edit_with_two_deletions_passes() -> None:
    text = skill(S0_STEPS[0], NEW_STEP, *S0_STEPS[3:5])
    assert changed_lines(text, S0) == (1, 3)  # 마지막 단계 삭제까지 3줄
    text = skill(S0_STEPS[0], NEW_STEP, *S0_STEPS[3:])
    assert changed_lines(text, S0) == (1, 2)
    assert check_proposer_candidate(text, S0) == ()


# ---------------------------------------------------------------------------
# 구조: 제목 + 1부터 연속인 번호 단계만
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        S0 + "\n## Notes\n",
        S0 + "\n- Check traffic first.\n",
        S0 + "\nRationale: traffic matters.\n",
        S0.replace("# Incident Response Skill", "# Incident Skill"),
        S0.replace("6. Otherwise", "7. Otherwise"),
        S0.replace("1. Confirm", "Confirm"),
    ],
    ids=["heading", "bullet", "prose", "title", "numbering-gap", "unnumbered-step"],
)
def test_structure_violations(text: str) -> None:
    assert "structure" in check_proposer_candidate(text, S0)


def test_blank_lines_between_steps_are_fine() -> None:
    text = S0.replace("\n2. Check", "\n\n2. Check")
    assert "structure" not in check_proposer_candidate(text, S0)


# ---------------------------------------------------------------------------
# 용어: 채점 용어, 실패 라벨 이름, 술어 이름
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    [
        "Evidence Card",
        "hypothesis",
        "expected effect",
        "failure type",
        "evidence_safe",
        "Evidence-Safe",
        "correct disposition",
        "predicate",
        "PREMATURE_ACTION",
        "wrong_intervention",
        "UNNECESSARY_INTERVENTION",
        "NO_DISPOSITION",
        "traffic_ruled_out",
        "deploy_before_regression",
        "capacity_saturated",
        "incident_confirmed",
    ],
)
def test_grading_terms_are_rejected(term: str) -> None:
    text = S0.replace("Check dependency health.", f"Check dependency health ({term}).")
    assert check_proposer_candidate(text, S0) == ("terms",)


# ---------------------------------------------------------------------------
# §5 나머지 유효성 검사(proposer 후보 전체 검증): validate_proposer_candidate
# ---------------------------------------------------------------------------

EXPORT_IDS = frozenset(f"case-{i:012x}" for i in range(10))
SHOWN = sorted(EXPORT_IDS)[:6]
VALID_SKILL = skill(*S0_STEPS[:3], NEW_STEP, *S0_STEPS[3:])


def card(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "candidate_id": "cand-003",
        "failure_type": "PREMATURE_ACTION",
        "source_cases": list(SHOWN),
        "hypothesis": "Checking traffic first avoids wrong rollbacks.",
        "expected_effect": {"cd": "up", "es": "up"},
    }
    data.update(overrides)
    return data


def validate(text: str = VALID_SKILL, evidence: dict[str, object] | None = None) -> tuple[str, ...]:
    return validate_proposer_candidate(text, evidence or card(), S0, EXPORT_IDS)


def with_step(step: str) -> str:
    """S0 4단계 앞에 단계 하나를 더한다(줄 수 검사는 통과)."""
    return skill(*S0_STEPS[:3], f"0. {step}", *S0_STEPS[3:])


def test_valid_candidate_passes_everything() -> None:
    assert validate() == ()


def test_reasonable_proposer_edits_pass() -> None:
    """2차 실측 두 후보와 같은 형태의 편집은 통과한다."""
    edited = S0.replace(
        "roll the deployment back.",
        "check traffic metrics to rule out increased traffic before rolling the deployment back.",
    )
    assert validate(edited) == ()
    assert validate(with_step("Check traffic metrics to determine whether load is elevated.")) == ()


def test_unchanged_skill_is_no_change() -> None:
    assert validate(S0) == ("no_change",)


def test_non_ascii() -> None:
    assert validate(with_step("Check traffic — then decide.")) == ("non_ascii",)


def test_title_must_be_the_s0_title() -> None:
    reasons = validate(VALID_SKILL.replace("# Incident Response Skill", "# Incident Response"))
    assert "title" in reasons


def test_length_cap_is_twice_s0() -> None:
    filler = "Check traffic " + "and traffic " * 200
    assert "length" in validate(with_step(filler))
    assert len(VALID_SKILL) <= 2 * len(S0)


@pytest.mark.parametrize(
    "step",
    ["Check traffic if latency rose by 50 percent.", "Scale if traffic is above 150% of baseline."],
)
def test_numbers_outside_step_numbers(step: str) -> None:
    assert "numbers" in validate(with_step(step))


@pytest.mark.parametrize(
    "name",
    ["traffic spike", "Bad-Deployment", "dependency_failure", "CPU saturated", "deploy in window"],
)
def test_family_and_subtype_names(name: str) -> None:
    assert "family_names" in validate(with_step(f"Consider a {name} first."))


@pytest.mark.parametrize(
    "text",
    [
        "Check checkout-api first.",
        "Check payment-provider first.",
        "Roll back deploy-68ba6c.",
        "Roll back vX.Y and v.Z.",
    ],
)
def test_service_dependency_and_deployment_ids(text: str) -> None:
    if text.startswith("Roll back vX"):
        assert "ids" not in validate(with_step(text))  # 숫자 없는 v는 버전이 아님
    else:
        assert "ids" in validate(with_step(text))


def test_version_strings_are_ids_and_numbers() -> None:
    assert {"ids", "numbers"} <= set(validate(with_step("Roll back to v1.2.3.")))


@pytest.mark.parametrize(
    "word",
    [
        "log",
        "logs",
        "restart",
        "failover",
        "escalate",
        "page",
        "notify",
        "ticket",
        "wait",
        "database",
        "queue",
        "cache",
        "feature flag",
        "roll forward",
        "redeploy",
        "hotfix",
        "reboot",
        "drain",
    ],
)
def test_out_of_scope_step_words(word: str) -> None:
    assert validate(with_step(f"If unsure, {word} the service.")) == ("forbidden_words",)


@pytest.mark.parametrize("text", ["Check the catalog.", "Inspect the dialog.", "Avoid waiting."])
def test_forbidden_words_match_whole_words_only(text: str) -> None:
    """금지어는 단어 단위: catalog·dialog는 log가 아니다. waiting은 wait와 다른 단어다."""
    assert "forbidden_words" not in validate(with_step(text))


@pytest.mark.parametrize(
    "evidence",
    [
        card(candidate_id="c3"),
        card(failure_type="UNNECESSARY_INTERVENTION"),
        card(source_cases=["case-ffffffffffff", *SHOWN[1:]]),
        card(source_cases=SHOWN[:5]),
        card(hypothesis="  "),
        card(expected_effect={"cd": "up"}),
        card(expected_effect={"cd": "up", "es": "sideways"}),
        {k: v for k, v in card().items() if k != "hypothesis"},
        card(extra="x"),
    ],
    ids=[
        "candidate-id",
        "failure-type",
        "unknown-source",
        "five-sources",
        "blank-hypothesis",
        "effect-missing",
        "effect-enum",
        "missing-field",
        "extra-field",
    ],
)
def test_evidence_card_schema(evidence: dict[str, object]) -> None:
    assert validate(evidence=evidence) == ("evidence_card",)


def test_reasons_come_in_a_fixed_order() -> None:
    text = with_step("Restart checkout-api if traffic spike exceeds 50%.").replace(
        "# Incident Response Skill", "# Skill"
    )
    assert validate(text, card(hypothesis="")) == (
        "title",
        "numbers",
        "family_names",
        "ids",
        "forbidden_words",
        "line_count",  # 제목 줄도 바뀌어 추가·수정 2줄
        "structure",
        "evidence_card",
    )


def test_first_probe_candidates_are_invalid_in_full_validation() -> None:
    assert "line_count" in validate(PROBE_GPT_5_4_MINI)
    assert {"line_count", "structure", "terms"} <= set(validate(PROBE_GPT_5_4))


def padded_to(length: int) -> str:
    """길이가 정확히 length인 유효 형태의 후보(한 단계 추가)."""
    base = with_step("Check x.")
    return base.replace("Check x.", "Check " + "x" * (length - len(base) + 1) + ".")


def test_length_boundary_is_exactly_twice_s0() -> None:
    limit = 2 * len(S0)
    assert len(padded_to(limit)) == limit
    assert "length" not in validate(padded_to(limit))
    assert "length" in validate(padded_to(limit + 1))


def test_percent_sign_alone_is_rejected() -> None:
    assert validate(with_step("Scale only if traffic rose by many %.")) == ("numbers",)
