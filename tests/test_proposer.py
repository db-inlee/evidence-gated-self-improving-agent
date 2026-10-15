"""OpenAI proposer(Ticket 004 §4). 가짜 전송(httpx MockTransport)으로만 테스트한다."""

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.proposer import (
    Proposer,
    ProposerConfig,
    ProposerStopped,
    list_models,
    sample_cases,
)
from evidencegate.proposer_inputs import ProposerCase
from evidencegate.splits import dev_b, oracle
from opsworld.generator import L2_SUBTYPES
from opsworld.predicates import Predicate
from opsworld.state import Family

REPO_ROOT = Path(__file__).resolve().parents[1]
S0_TEXT = (REPO_ROOT / "skills" / "S0" / "SKILL.md").read_text(encoding="utf-8")
FAKE_KEY = "sk-test-SECRET1234567890abcdef"

GOOD_SKILL = S0_TEXT.replace(
    "6. Otherwise inspect resource usage and traffic.",
    "6. Otherwise inspect resource usage and traffic.\n7. Check traffic before rolling back.",
)


def make_case(i: int, *, correct: bool, safe: bool = False) -> ProposerCase:
    return ProposerCase(
        export_id=f"case-{i:012x}",
        alert=f"Alert: service svc-{i} is degraded (latency/error SLO breach). Investigate.",
        steps=(
            {
                "tool": "get_service_metrics",
                "args": {"service_id": f"svc-{i}"},
                "output": {"p95_ms": 900},
            },
            {
                "tool": "rollback_deployment",
                "args": {"service_id": f"svc-{i}"},
                "output": {"status": "recorded"},
            },
        ),
        action="rollback_deployment",
        correct_disposition=correct,
        evidence_safe=safe,
        failure_labels=("PREMATURE_ACTION",)
        if correct
        else ("PREMATURE_ACTION", "WRONG_INTERVENTION"),
        missing_predicates=("traffic_ruled_out",),
    )


# Dev-A 실패와 같은 구성: 근거 부족 19, 처분 오답 21
CASES = [make_case(i, correct=True) for i in range(19)] + [
    make_case(100 + i, correct=False) for i in range(21)
]


def completion(
    content: str,
    *,
    prompt: int = 3000,
    out: int = 400,
    reasoning: int = 0,
    finish: str = "stop",
    refusal: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": out,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def good_content(
    hypothesis: str = "Checking traffic before rollback avoids wrong rollbacks.",
) -> str:
    return json.dumps(
        {
            "skill_md": GOOD_SKILL,
            "failure_type": "WRONG_INTERVENTION",
            "hypothesis": hypothesis,
            "expected_effect": {"cd": "up", "es": "up"},
        }
    )


class FakeOpenAI:
    def __init__(self, replies: list[httpx.Response | dict[str, Any] | Exception]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-x"}, {"id": "gpt-x-mini"}]})
        self.bodies.append(json.loads(request.content))
        reply = self.replies.pop(0) if self.replies else completion(good_content())
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    def client(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self.handler), base_url="https://api.openai.test/v1"
        )


def config(**overrides: Any) -> ProposerConfig:
    base: dict[str, Any] = {
        "model": "gpt-x",
        "max_completion_tokens": 2000,
        "input_usd_per_mtok": 1.0,
        "output_usd_per_mtok": 4.0,
        "retry_wait_s": 0.0,
    }
    base.update(overrides)
    return ProposerConfig(**base)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)


def make(fake: FakeOpenAI, tmp_path: Path, **overrides: Any) -> Proposer:
    return Proposer(
        config(**overrides),
        cases=CASES,
        s0_text=S0_TEXT,
        ledger_path=tmp_path / "ledger.jsonl",
        client=fake.client(),
    )


# ---------------------------------------------------------------------------
# 층화 표본
# ---------------------------------------------------------------------------


def test_sample_is_stratified_deterministic_and_varies_by_call() -> None:
    for call in range(30):
        sample = sample_cases(CASES, call, k=6, min_per_stratum=2)
        assert len(sample) == 6 and len({c.export_id for c in sample}) == 6
        assert sum(c.correct_disposition and not c.evidence_safe for c in sample) >= 2
        assert sum(not c.correct_disposition for c in sample) >= 2
        assert sample == sample_cases(CASES, call, k=6, min_per_stratum=2)
    assert sample_cases(CASES, 0, 6, 2) != sample_cases(CASES, 1, 6, 2)


def test_sample_fails_when_a_stratum_is_too_small() -> None:
    with pytest.raises(ValueError, match="stratum"):
        sample_cases(CASES[:19] + CASES[19:20], 0, k=6, min_per_stratum=2)


# ---------------------------------------------------------------------------
# 요청 형식과 금지 정보
# ---------------------------------------------------------------------------


def test_request_uses_structured_output_and_configured_limits(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content())])
    make(fake, tmp_path).propose(call_index=0)
    body = fake.bodies[0]
    assert body["model"] == "gpt-x"
    assert body["max_completion_tokens"] == 2000
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in body and "reasoning_effort" not in body
    assert fake.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"


def test_optional_sampling_parameters_are_sent_only_when_set(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content())])
    make(fake, tmp_path, temperature=0.7, reasoning_effort="low").propose(call_index=0)
    assert fake.bodies[0]["temperature"] == 0.7
    assert fake.bodies[0]["reasoning_effort"] == "low"


def test_prompt_contains_s0_tools_cases_and_previous_hypotheses(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content())])
    make(fake, tmp_path).propose(call_index=3, previous_hypotheses=["Add a traffic check."])
    text = json.dumps(fake.bodies[0]["messages"])
    assert json.dumps(S0_TEXT)[1:-1] in text
    for tool in ("get_traffic_metrics", "observe_only", "scale_service"):
        assert tool in text
    for case in sample_cases(CASES, 3, 6, 2):
        assert case.export_id in text
    assert "Add a traffic check." in text


def control_distinct_lines() -> list[str]:
    s0_lines = set(S0_TEXT.splitlines())
    lines = []
    for path in (REPO_ROOT / "skills" / "controls").glob("*/SKILL.md"):
        lines += [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and line not in s0_lines
        ]
    return lines


def test_request_never_carries_forbidden_information(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content())])
    make(fake, tmp_path).propose(call_index=0, previous_hypotheses=["x"])
    text = json.dumps(fake.bodies[0])
    scrubbed = text
    for predicate in Predicate:
        scrubbed = scrubbed.replace(predicate.value, "")
    for word in [f.value for f in Family] + [n for v in L2_SUBTYPES.values() for n in v]:
        assert word not in scrubbed, word
    for key in ("root_cause", "has_bug", "l2_subtype", "incident_id"):
        assert key not in text
    assert "UNNECESSARY_INTERVENTION" not in text
    for case in [*dev_b(), *oracle()]:
        assert re.search(rf"\b{case.incident_seed}\b", text) is None
    for line in control_distinct_lines():
        assert line not in text, line


# ---------------------------------------------------------------------------
# 응답 파싱과 형식 오류
# ---------------------------------------------------------------------------


def test_parses_candidate_and_fills_id_and_source_cases(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content("H1"))])
    result = make(fake, tmp_path).propose(call_index=5)
    assert result.candidate is not None
    card = result.candidate.evidence
    assert result.candidate.skill_md == GOOD_SKILL
    assert card["candidate_id"] == "cand-005"
    assert card["source_cases"] == [c.export_id for c in sample_cases(CASES, 5, 6, 2)]
    assert card["failure_type"] == "WRONG_INTERVENTION"
    assert card["hypothesis"] == "H1"
    assert card["expected_effect"] == {"cd": "up", "es": "up"}


@pytest.mark.parametrize(
    "reply",
    [
        completion("not json"),
        completion(json.dumps({"skill_md": GOOD_SKILL})),  # 필드 누락
        completion(
            json.dumps(
                {
                    "skill_md": GOOD_SKILL,
                    "failure_type": "MADE_UP",
                    "hypothesis": "h",
                    "expected_effect": {"cd": "up", "es": "up"},
                }
            )
        ),
        # 내용이 정상 JSON이어도 거절·출력 상한 도달은 형식 오류다
        completion(good_content(), refusal="I can't help with that."),
        completion(good_content(), finish="length"),
    ],
    ids=["not-json", "missing-field", "bad-enum", "refusal", "truncated"],
)
def test_format_errors_are_recorded_not_raised_until_rate_limit(
    reply: dict[str, Any], tmp_path: Path
) -> None:
    fake = FakeOpenAI([reply])
    proposer = make(fake, tmp_path, max_format_error_rate=1.0)
    result = proposer.propose(call_index=0)
    assert result.candidate is None
    assert result.format_error is not None
    assert proposer.format_errors == 1


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        (completion("not json"), "parse_failure"),
        (completion(""), "parse_failure"),
        (completion(json.dumps({"skill_md": GOOD_SKILL})), "field_mismatch"),
        (
            completion(
                json.dumps(
                    {
                        "skill_md": GOOD_SKILL,
                        "failure_type": "MADE_UP",
                        "hypothesis": "h",
                        "expected_effect": {"cd": "up", "es": "up"},
                    }
                )
            ),
            "field_mismatch",
        ),
        (completion(good_content(), refusal="I can't help with that."), "refusal"),
        (completion(good_content(), finish="length"), "truncated"),
    ],
    ids=["not-json", "empty", "missing-field", "bad-enum", "refusal", "truncated"],
)
def test_format_error_kind_is_in_result_and_ledger(
    reply: dict[str, Any], kind: str, tmp_path: Path
) -> None:
    proposer = make(FakeOpenAI([reply]), tmp_path, max_format_error_rate=1.0)
    result = proposer.propose(call_index=0)
    assert result.format_error_kind == kind
    assert read_ledger(tmp_path)[-1]["error_kind"] == kind


def test_good_call_has_no_format_error_kind(tmp_path: Path) -> None:
    proposer = make(FakeOpenAI([completion(good_content())]), tmp_path)
    assert proposer.propose(call_index=0).format_error_kind is None
    assert read_ledger(tmp_path)[-1]["error_kind"] is None


def test_format_stop_reports_kind_and_response_excerpt(tmp_path: Path) -> None:
    """멈출 때 오류 종류와 응답 앞부분을 보여 준다. 앞부분은 길이가 제한되고 키는 지워진다."""
    content = "Sorry, here is " + "x" * 2000 + f" {FAKE_KEY}"
    proposer = make(FakeOpenAI([completion(content)]), tmp_path)
    with pytest.raises(ProposerStopped) as excinfo:
        proposer.propose(call_index=0)
    message = str(excinfo.value)
    assert "parse_failure" in message
    assert "Sorry, here is xxx" in message
    assert len(message) < 1000
    assert FAKE_KEY not in message
    stopped = excinfo.value.result
    assert stopped is not None
    assert (stopped.call_index, stopped.format_error_kind) == (0, "parse_failure")


def test_non_format_stops_carry_no_result(tmp_path: Path) -> None:
    proposer = make(FakeOpenAI([]), tmp_path, budget_usd=0.005)
    with pytest.raises(ProposerStopped) as excinfo:
        proposer.propose(call_index=0)
    assert excinfo.value.result is None


def test_format_error_on_the_first_call_already_exceeds_the_rate(tmp_path: Path) -> None:
    """문자 그대로의 규칙: 첫 호출이 형식 오류면 1/1 = 100% > 5%라 바로 멈춘다."""
    fake = FakeOpenAI([completion("bad")])
    with pytest.raises(ProposerStopped, match="format error rate"):
        make(fake, tmp_path).propose(call_index=0)


def test_format_error_rate_exactly_at_limit_does_not_stop(tmp_path: Path) -> None:
    """정상 19번 뒤 형식 오류 1번: 1/20 = 5%는 '5%를 넘음'이 아니다."""
    replies: list[Any] = [completion(good_content()) for _ in range(19)] + [completion("bad")]
    proposer = make(FakeOpenAI(replies), tmp_path)
    for i in range(20):
        proposer.propose(call_index=i)
    assert (proposer.format_errors, proposer.calls) == (1, 20)


# ---------------------------------------------------------------------------
# 비용 기록과 상한
# ---------------------------------------------------------------------------


def read_ledger(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]


def test_each_call_records_tokens_and_cost(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content(), prompt=3000, out=500, reasoning=200)])
    proposer = make(fake, tmp_path)
    proposer.propose(call_index=0)
    (entry,) = read_ledger(tmp_path)
    assert entry["input_tokens"] == 3000
    assert entry["output_tokens"] == 500
    assert entry["reasoning_tokens"] == 200
    expected = 3000 / 1e6 * 1.0 + 500 / 1e6 * 4.0
    assert entry["cost_usd"] == pytest.approx(expected)
    assert entry["cumulative_usd"] == pytest.approx(expected)
    assert proposer.cumulative_usd == pytest.approx(expected)
    assert entry["status"] == "ok"


def test_budget_check_before_call_stops_without_sending(tmp_path: Path) -> None:
    fake = FakeOpenAI([])
    # 다음 호출의 최대 비용: 출력 상한 2000 토큰 x $4/M = $0.008(+입력 추정) > 상한 $0.005
    proposer = make(fake, tmp_path, budget_usd=0.005)
    with pytest.raises(ProposerStopped, match="budget"):
        proposer.propose(call_index=0)
    assert fake.bodies == []


def test_cumulative_cost_persists_across_instances(tmp_path: Path) -> None:
    fake = FakeOpenAI([completion(good_content(), prompt=1_000_000, out=0)])
    make(fake, tmp_path).propose(call_index=0)  # $1.00
    second = make(FakeOpenAI([]), tmp_path, budget_usd=1.005)
    assert second.cumulative_usd == pytest.approx(1.0)
    with pytest.raises(ProposerStopped, match="budget"):
        second.propose(call_index=1)


def test_max_next_call_cost_uses_output_limit_and_input_estimate(tmp_path: Path) -> None:
    proposer = make(FakeOpenAI([]), tmp_path)
    estimate = proposer.max_next_call_cost(call_index=0)
    output_part = 2000 / 1e6 * 4.0
    assert estimate > output_part
    assert estimate < output_part + 0.05


# ---------------------------------------------------------------------------
# 서버 오류, 잔액 부족, 인증
# ---------------------------------------------------------------------------


def test_server_errors_are_retried_up_to_three_times(tmp_path: Path) -> None:
    err = httpx.Response(500, json={"error": {"message": "server error"}})
    fake = FakeOpenAI([err, err, err, completion(good_content())])
    result = make(fake, tmp_path).propose(call_index=0)
    assert result.candidate is not None
    assert len(fake.bodies) == 4
    statuses = [e["status"] for e in read_ledger(tmp_path)]
    assert statuses == ["server_error", "server_error", "server_error", "ok"]


def test_fourth_server_error_stops(tmp_path: Path) -> None:
    err = httpx.Response(503, json={"error": {"message": "overloaded"}})
    fake = FakeOpenAI([err, err, err, err])
    with pytest.raises(ProposerStopped, match="server error"):
        make(fake, tmp_path).propose(call_index=0)
    assert len(fake.bodies) == 4


def test_connection_errors_are_retried(tmp_path: Path) -> None:
    fake = FakeOpenAI([httpx.ConnectError("boom"), completion(good_content())])
    assert make(fake, tmp_path).propose(call_index=0).candidate is not None


def test_insufficient_quota_stops_immediately(tmp_path: Path) -> None:
    quota = httpx.Response(
        429, json={"error": {"code": "insufficient_quota", "message": "You exceeded your quota."}}
    )
    fake = FakeOpenAI([quota, completion(good_content())])
    with pytest.raises(ProposerStopped, match="quota"):
        make(fake, tmp_path).propose(call_index=0)
    assert len(fake.bodies) == 1


# ---------------------------------------------------------------------------
# API 키 비노출
# ---------------------------------------------------------------------------


def test_key_never_appears_in_errors_ledger_config_or_results(tmp_path: Path) -> None:
    echo = httpx.Response(
        401,
        json={
            "error": {"message": f"Incorrect API key provided: {FAKE_KEY}. Also sk-proj-abc***xyz."}
        },
    )
    fake = FakeOpenAI([completion(good_content()), echo])
    proposer = make(fake, tmp_path)
    result = proposer.propose(call_index=0)
    with pytest.raises(ProposerStopped) as excinfo:
        proposer.propose(call_index=1)
    texts = [
        str(excinfo.value),
        repr(excinfo.value),
        (tmp_path / "ledger.jsonl").read_text(),
        json.dumps(proposer.proposer_config()),
        repr(proposer.config),
        repr(result),
    ]
    for text in texts:
        assert FAKE_KEY not in text
        assert not re.search(r"sk-[A-Za-z0-9]", text), text


def test_key_without_sk_prefix_is_also_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """키 형식에 기대지 않고 환경 변수 값 자체를 지운다."""
    plain = "plainsecretvalue0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", plain)
    echo = httpx.Response(401, json={"error": {"message": f"Bad key: {plain}"}})
    proposer = make(FakeOpenAI([echo]), tmp_path)
    with pytest.raises(ProposerStopped) as excinfo:
        proposer.propose(call_index=0)
    assert plain not in str(excinfo.value)
    assert plain not in (tmp_path / "ledger.jsonl").read_text()


def test_missing_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ProposerStopped, match="OPENAI_API_KEY is not set"):
        make(FakeOpenAI([]), tmp_path)


def test_list_models(tmp_path: Path) -> None:
    fake = FakeOpenAI([])
    assert list_models(fake.client()) == ["gpt-x", "gpt-x-mini"]
    assert fake.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"


def test_proposer_config_records_reproducibility_fields(tmp_path: Path) -> None:
    cfg = make(FakeOpenAI([]), tmp_path).proposer_config()
    for field in (
        "model",
        "max_completion_tokens",
        "input_usd_per_mtok",
        "output_usd_per_mtok",
        "prompt_sha256",
        "cases_per_call",
        "min_per_stratum",
        "budget_usd",
        "max_retries",
        "max_format_error_rate",
    ):
        assert field in cfg
    assert re.fullmatch(r"[0-9a-f]{64}", cfg["prompt_sha256"])
