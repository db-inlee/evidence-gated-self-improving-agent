"""propose_cli 선택 인자(Ticket 009b §1·§2·§3·§4의 3).

기본값의 v1 동작은 tests/test_propose_cli.py가 그대로 확인한다.
여기서는 새 인자만 본다. 가짜 전송만 쓴다.
"""

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
S0 = S0_PATH.read_text()
STEP4 = (
    "4. If a recent deployment exists and all dependencies are healthy, roll the deployment back."
)
FAKE_KEY = "sk-test-SECRET-options-1234567890"

# 4단계에 붙이는 서로 다른 조건(v1 줄 토큰 판정으로는 서로 중복, D1로는 따로 남음)
CLAUSES = (
    "and traffic is unchanged",
    "and CPU usage is not elevated",
    "and only one deployment is recent",
    "and the deployment preceded the regression",
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr("evidencegate.proposer.time.sleep", lambda _: None)


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


def skill_with(clause: str) -> str:
    return S0.replace(STEP4, STEP4.replace("healthy,", f"healthy, {clause},"))


def reply(hypothesis: str, skill_md: str) -> dict[str, Any]:
    text = json.dumps(
        {
            "skill_md": skill_md,
            "failure_type": "WRONG_INTERVENTION",
            "hypothesis": hypothesis,
            "expected_effect": {"cd": "up", "es": "up"},
        }
    )
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }


class FakeOpenAI:
    def __init__(self, replies: list[httpx.Response | dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        item = self.replies.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

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


def args(tmp_path: Path, inputs: Path, *extra: str) -> list[str]:
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
        "--controls-dir",
        str(REPO_ROOT / "skills" / "controls"),
        *extra,
    ]


def run(argv: list[str], fake: FakeOpenAI) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, client=fake.client(), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def saved(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "out").glob("call-*.json"))]


def user(body: dict[str, Any]) -> str:
    return str(body["messages"][1]["content"])


# --previous -------------------------------------------------------------------


def test_default_previous_list_has_pool_candidates_only(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI(
        [
            reply("H-first", skill_with(CLAUSES[0])),
            reply("H-dup", skill_with(CLAUSES[1])),
            reply("H3", skill_with(CLAUSES[2])),
        ]
    )
    run(args(tmp_path, inputs, "--calls", "3"), fake)
    assert saved(tmp_path)[1]["duplicate_of"] == "cand-000"  # v1 판정: 같은 줄
    assert "H-dup" not in user(fake.bodies[2])


def test_valid_previous_list_includes_duplicates(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI(
        [
            reply("H-first", skill_with(CLAUSES[0])),
            reply("H-dup", skill_with(CLAUSES[1])),
            reply("H3", skill_with(CLAUSES[2])),
        ]
    )
    code, _, _ = run(args(tmp_path, inputs, "--calls", "3", "--previous", "valid"), fake)
    assert code == 0
    assert saved(tmp_path)[1]["duplicate_of"] == "cand-000"
    assert "- H-first\n- H-dup" in user(fake.bodies[2])


def test_valid_previous_list_skips_the_same_change_and_invalid_ones(
    tmp_path: Path, inputs: Path
) -> None:
    same = skill_with(CLAUSES[0])
    fake = FakeOpenAI(
        [
            reply("H-a", same),
            reply("H-a-again", same.replace("Traffic", "traffic")),  # 정규화하면 같은 변경
            reply("H-invalid", same + "\n## Notes\n"),
            reply("H-last", skill_with(CLAUSES[1])),
        ]
    )
    run(args(tmp_path, inputs, "--calls", "4", "--previous", "valid"), fake)
    last = user(fake.bodies[3])
    assert "- H-a\n" in last + "\n"
    assert "H-a-again" not in last and "H-invalid" not in last


def test_valid_previous_list_is_rebuilt_when_continuing(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI(
        [
            reply("H-first", skill_with(CLAUSES[0])),
            reply("H-dup", skill_with(CLAUSES[1])),
            reply("H3", skill_with(CLAUSES[2])),
        ]
    )
    run(args(tmp_path, inputs, "--calls", "2", "--previous", "valid"), fake)
    run(args(tmp_path, inputs, "--calls", "1", "--previous", "valid"), fake)
    assert "- H-first\n- H-dup" in user(fake.bodies[2])


# --dedupe ---------------------------------------------------------------------


def test_word_dedupe_keeps_different_clauses_on_the_same_line(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([reply(f"H{i}", skill_with(c)) for i, c in enumerate(CLAUSES)])
    code, out, _ = run(
        args(tmp_path, inputs, "--calls", "4", "--dedupe", "word", "--dedupe-threshold", "0.6"),
        fake,
    )
    assert code == 0
    assert [r["in_pool"] for r in saved(tmp_path)] == [True] * 4
    assert "pool 4" in out


def test_word_dedupe_still_drops_the_same_change(tmp_path: Path, inputs: Path) -> None:
    same = skill_with(CLAUSES[0])
    fake = FakeOpenAI([reply("H0", same), reply("H1", same.replace(",", " ,"))])
    run(
        args(tmp_path, inputs, "--calls", "2", "--dedupe", "word", "--dedupe-threshold", "0.6"),
        fake,
    )
    assert saved(tmp_path)[1]["duplicate_of"] == "cand-000"


def test_word_dedupe_is_used_for_control_marks(tmp_path: Path, inputs: Path) -> None:
    control = (REPO_ROOT / "skills/controls/pos-traffic-check/SKILL.md").read_text()
    fake = FakeOpenAI([reply("H0", control)])
    run(args(tmp_path, inputs, "--dedupe", "word", "--dedupe-threshold", "0.9"), fake)
    assert saved(tmp_path)[0]["control_duplicates"] == ["pos-traffic-check"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--dedupe", "word"],
        ["--dedupe-threshold", "0.6"],
        ["--dedupe", "word", "--dedupe-threshold", "0"],
    ],
)
def test_dedupe_threshold_goes_with_word_only(
    tmp_path: Path, inputs: Path, extra: list[str]
) -> None:
    fake = FakeOpenAI([])
    code, _, err = run(args(tmp_path, inputs, *extra), fake)
    assert code == 1 and "dedupe" in err
    assert fake.bodies == []


# --stop-after-idle ------------------------------------------------------------


def test_stops_after_idle_calls_without_a_new_pool_candidate(tmp_path: Path, inputs: Path) -> None:
    same = skill_with(CLAUSES[0])
    fake = FakeOpenAI([reply(f"H{i}", same) for i in range(6)])
    code, out, _ = run(args(tmp_path, inputs, "--calls", "6", "--stop-after-idle", "3"), fake)
    assert code == 0
    assert len(fake.bodies) == 4  # 풀 1개 뒤 3회 연속 무증가
    assert "no new pool candidate in the last 3 calls" in out


def test_idle_count_carries_over_when_continuing(tmp_path: Path, inputs: Path) -> None:
    same = skill_with(CLAUSES[0])
    fake = FakeOpenAI([reply(f"H{i}", same) for i in range(6)])
    run(args(tmp_path, inputs, "--calls", "2", "--stop-after-idle", "3"), fake)
    code, out, _ = run(args(tmp_path, inputs, "--calls", "5", "--stop-after-idle", "3"), fake)
    assert code == 0 and len(fake.bodies) == 4
    assert "no new pool candidate in the last 3 calls" in out


def test_idle_counts_format_errors_and_invalid_calls(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI(
        [
            reply("H0", skill_with(CLAUSES[0])),
            reply("H-bad", skill_with(CLAUSES[1]) + "\n## Notes\n"),
        ]
    )
    run(args(tmp_path, inputs, "--calls", "5", "--stop-after-idle", "1"), fake)
    assert len(fake.bodies) == 2


# 실행 설정 기록 -------------------------------------------------------------------


def test_default_options_write_no_run_options_file(tmp_path: Path, inputs: Path) -> None:
    run(args(tmp_path, inputs), FakeOpenAI([reply("H0", skill_with(CLAUSES[0]))]))
    assert not (tmp_path / "out" / "run_options.json").exists()


def test_new_options_are_recorded_and_a_different_continue_is_refused(
    tmp_path: Path, inputs: Path
) -> None:
    extra = [
        "--previous",
        "valid",
        "--dedupe",
        "word",
        "--dedupe-threshold",
        "0.6",
        "--stop-after-idle",
        "25",
    ]
    fake = FakeOpenAI([reply("H0", skill_with(CLAUSES[0])), reply("H1", skill_with(CLAUSES[1]))])
    run(args(tmp_path, inputs, *extra), fake)
    recorded = json.loads((tmp_path / "out" / "run_options.json").read_text())
    assert recorded == {
        "previous": "valid",
        "dedupe": "word",
        "dedupe_threshold": 0.6,
        "stop_after_idle": 25,
    }
    code, _, err = run(args(tmp_path, inputs), fake)  # 기본값으로 이어 부르기
    assert code == 1 and "run_options" in err
    assert len(fake.bodies) == 1
    code, _, _ = run(args(tmp_path, inputs, *extra), fake)
    assert code == 0 and len(fake.bodies) == 2


def test_new_options_on_a_folder_from_default_runs_are_refused(
    tmp_path: Path, inputs: Path
) -> None:
    fake = FakeOpenAI([reply("H0", skill_with(CLAUSES[0]))])
    run(args(tmp_path, inputs), fake)
    code, _, err = run(args(tmp_path, inputs, "--previous", "valid"), fake)
    assert code == 1 and "run_options" in err


# --quota-message ---------------------------------------------------------------


def no_credits() -> httpx.Response:
    return httpx.Response(
        429, json={"error": {"message": "You have no credits remaining. Add credits."}}
    )


def ledger(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]


def test_quota_message_stops_at_once_as_a_quota_error(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([no_credits()])
    code, _, err = run(args(tmp_path, inputs, "--quota-message", "no credits remaining"), fake)
    assert code == 2 and "insufficient quota or balance" in err
    assert len(fake.bodies) == 1
    assert [r["status"] for r in ledger(tmp_path)] == ["quota"]


def test_without_quota_message_the_v1_retry_path_is_kept(tmp_path: Path, inputs: Path) -> None:
    fake = FakeOpenAI([no_credits() for _ in range(4)])
    code, _, err = run(args(tmp_path, inputs), fake)
    assert code == 2 and "server error after 4 attempts" in err
    assert [r["status"] for r in ledger(tmp_path)] == ["server_error"] * 4


def test_control_marks_follow_the_selected_rule(tmp_path: Path, inputs: Path) -> None:
    """v1 판정은 pos-timing-only와 중복(J 0.889), D1은 0.600이라 t = 0.7이면 표시하지 않는다."""
    text = skill_with("and the deployment happened shortly before the degradation started")
    fake = FakeOpenAI([reply("H0", text), reply("H0", text)])
    run(args(tmp_path / "v1", inputs), fake)
    assert saved(tmp_path / "v1")[0]["control_duplicates"] == ["pos-timing-only"]
    run(args(tmp_path / "word", inputs, "--dedupe", "word", "--dedupe-threshold", "0.7"), fake)
    assert saved(tmp_path / "word")[0]["control_duplicates"] == []


def test_rebuilt_valid_list_skips_the_same_change(tmp_path: Path, inputs: Path) -> None:
    same = skill_with(CLAUSES[0])
    fake = FakeOpenAI(
        [
            reply("H-a", same),
            reply("H-a-again", same.replace(",", " ,")),
            reply("H-next", skill_with(CLAUSES[1])),
        ]
    )
    run(args(tmp_path, inputs, "--calls", "2", "--previous", "valid"), fake)
    run(args(tmp_path, inputs, "--calls", "1", "--previous", "valid"), fake)
    last = user(fake.bodies[2])
    assert "- H-a\n" in last + "\n" and "H-a-again" not in last
