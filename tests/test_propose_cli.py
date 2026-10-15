"""proposer 진입점(실측과 풀 생성 공통). 가짜 전송으로만 테스트한다."""

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.propose_cli import main
from evidencegate.proposer_inputs import ProposerCase, save_proposer_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
S0_PATH = REPO_ROOT / "skills" / "S0" / "SKILL.md"
FAKE_KEY = "sk-test-SECRET1234567890abcdef"


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)


def make_case(i: int, *, correct: bool) -> ProposerCase:
    return ProposerCase(
        export_id=f"case-{i:012x}",
        alert="Alert: service svc-1 is degraded. Investigate.",
        steps=(),
        action="rollback_deployment",
        correct_disposition=correct,
        evidence_safe=False,
        failure_labels=("PREMATURE_ACTION",),
        missing_predicates=("traffic_ruled_out",),
    )


def content(hypothesis: str, *, extra: str = "") -> str:
    return json.dumps(
        {
            "skill_md": S0_PATH.read_text() + "\n7. Check traffic before rolling back.\n" + extra,
            "failure_type": "WRONG_INTERVENTION",
            "hypothesis": hypothesis,
            "expected_effect": {"cd": "up", "es": "up"},
        }
    )


def completion(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }


class FakeOpenAI:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-x"}, {"id": "gpt-x-mini"}]})
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))

    def client(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self.handler), base_url="https://api.openai.test/v1"
        )


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    path = tmp_path / "inputs.json"
    cases = [make_case(i, correct=True) for i in range(5)]
    cases += [make_case(i, correct=False) for i in range(5, 10)]
    save_proposer_cases(cases, path)
    return path


def propose_args(tmp_path: Path, inputs: Path, *extra: str) -> list[str]:
    return [
        "propose",
        "--model",
        "gpt-x",
        "--input-usd-per-mtok",
        "1.0",
        "--output-usd-per-mtok",
        "4.0",
        "--inputs",
        str(inputs),
        "--skill",
        str(S0_PATH),
        "--out-dir",
        str(tmp_path / "out"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        *extra,
    ]


def run(argv: list[str], fake: FakeOpenAI) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, client=fake.client(), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_models_lists_ids(tmp_path: Path) -> None:
    code, out, _ = run(["models"], FakeOpenAI([]))
    assert code == 0
    assert out.split() == ["gpt-x", "gpt-x-mini"]


def test_one_call_writes_a_result_file_and_ledger(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion(content("H0"))])
    code, out, _ = run(propose_args(tmp_path, inputs), fake)
    assert code == 0
    saved = json.loads((tmp_path / "out" / "call-000.json").read_text())
    assert saved["call_index"] == 0
    assert saved["candidate"]["evidence"]["hypothesis"] == "H0"
    assert saved["proposer_config"]["model"] == "gpt-x"
    assert saved["format_error"] is None
    assert "cand-000" in out
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


def test_calls_continue_from_existing_results_with_previous_hypotheses(
    tmp_path: Path, inputs: Path
) -> None:
    """같은 명령을 다시 돌리면 다음 call_index부터 이어서 부르고, 앞선 hypothesis를 보낸다."""
    fake = FakeOpenAI([completion(content("H0")), completion(content("H1"))])
    run(propose_args(tmp_path, inputs), fake)
    code, _, _ = run(propose_args(tmp_path, inputs), fake)
    assert code == 0
    assert (tmp_path / "out" / "call-001.json").exists()
    assert "- H0" in fake.bodies[1]["messages"][1]["content"]


def test_calls_option_makes_several_calls(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion(content(f"H{i}")) for i in range(3)])
    code, _, _ = run(propose_args(tmp_path, inputs, "--calls", "3"), fake)
    assert code == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "call-000.json",
        "call-001.json",
        "call-002.json",
    ]
    assert "- H0\n- H1" in fake.bodies[2]["messages"][1]["content"]


def test_format_stop_saves_the_call_and_exits_2(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion("not json")])
    code, _, err = run(propose_args(tmp_path, inputs), fake)
    assert code == 2
    assert "parse_failure" in err
    saved = json.loads((tmp_path / "out" / "call-000.json").read_text())
    assert saved["format_error_kind"] == "parse_failure"
    assert saved["candidate"] is None


def test_budget_stop_exits_2_without_a_request(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([])
    code, _, err = run(propose_args(tmp_path, inputs, "--budget-usd", "0.001"), fake)
    assert code == 2
    assert "budget" in err
    assert fake.bodies == []


def test_out_dir_with_a_different_config_is_refused(tmp_path: Path, inputs: Path) -> None:
    """한 출력 폴더는 한 proposer_config만 담는다(풀은 고정 설정으로 만든다)."""
    fake = FakeOpenAI([completion(content("H0"))])
    run(propose_args(tmp_path, inputs), fake)
    args = propose_args(tmp_path, inputs)
    args[args.index("--model") + 1] = "gpt-x-mini"
    code, _, err = run(args, fake)
    assert code == 1
    assert "proposer_config" in err
    assert len(fake.bodies) == 1


def test_key_is_not_in_outputs(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion(content("H0"))])
    _, out, err = run(propose_args(tmp_path, inputs), fake)
    texts = [out, err, (tmp_path / "out" / "call-000.json").read_text()]
    for text in texts:
        assert FAKE_KEY not in text


def test_result_records_the_section_5_checks(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion(content("H0", extra="\nEvidence card:\n- x\n"))])
    code, out, _ = run(propose_args(tmp_path, inputs), fake)
    assert code == 0
    saved = json.loads((tmp_path / "out" / "call-000.json").read_text())
    assert saved["invalid_reasons"] == ["line_count", "structure", "terms"]  # 줄 3개 추가
    assert "cand-000 invalid (line_count, structure, terms)" in out


def test_valid_result_has_no_invalid_reasons(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion(content("H0"))])
    run(propose_args(tmp_path, inputs), fake)
    saved = json.loads((tmp_path / "out" / "call-000.json").read_text())
    assert saved["invalid_reasons"] == []


def test_only_valid_candidates_enter_previous_hypotheses(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI(
        [
            completion(content("H-invalid", extra="\n## Notes\n")),
            completion(content("H-valid")),
            completion(content("H2")),
        ]
    )
    run(propose_args(tmp_path, inputs, "--calls", "2"), fake)
    run(propose_args(tmp_path, inputs), fake)
    assert "- H-valid" in fake.bodies[2]["messages"][1]["content"]
    for body in fake.bodies:
        assert "H-invalid" not in body["messages"][1]["content"]


def test_reasoning_effort_is_sent(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([completion(content("H0"))])
    run(propose_args(tmp_path, inputs, "--reasoning-effort", "low"), fake)
    assert fake.bodies[0]["reasoning_effort"] == "low"
    assert "temperature" not in fake.bodies[0]
