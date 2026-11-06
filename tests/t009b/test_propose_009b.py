"""Ticket 009b proposer 진입점(사전 등록 2.1, 티켓 §5의 4).

가짜 전송으로만 테스트한다(API 호출 없음).
"""

import hashlib
import io
import json
import string
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.proposer_inputs import ProposerCase, save_proposer_cases
from evidencegate.t009b import propose

ROOT = Path(__file__).resolve().parents[2]
S0_PATH = ROOT / "skills" / "S0" / "SKILL.md"
FAKE_KEY = "sk-test-SECRET009b-1234567890abcdef"
AC = "Now drawing from 'AC Power'"
CAF = "pid 1(caffeinate): [0x1] 00:00:01 PreventUserIdleSystemSleep named: x"


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr("evidencegate.proposer.time.sleep", lambda _: None)


def power_run(argv: Sequence[str]) -> str:
    joined = " ".join(argv)
    if joined == "pmset -g batt":
        return AC
    if joined == "pmset -g assertions":
        return CAF
    return '"AppleClamshellState" = No'


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


def word(i: int) -> str:
    letters = string.ascii_lowercase
    return "check" + letters[i // 26 % 26] + letters[i % 26]


def content(i: int, *, same: bool = False) -> str:
    k = 0 if same else i
    step = f"\n7. Before acting, {word(k)} the {word(k + 300)} signal.\n"
    return json.dumps(
        {
            "skill_md": S0_PATH.read_text() + step,
            "failure_type": "WRONG_INTERVENTION",
            "hypothesis": f"H{i}",
            "expected_effect": {"cd": "up", "es": "up"},
        }
    )


def completion(text: str, tokens: int = 1000) -> dict[str, Any]:
    return {
        "model": "gpt-5.4-2026-03-05",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": tokens // 2},
    }


class FakeOpenAI:
    def __init__(self, replies: list[httpx.Response | dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-5.4"}]})
        self.bodies.append(json.loads(request.content))
        item = self.replies.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def paths(tmp_path: Path) -> propose.Paths:
    inputs = tmp_path / "inputs.json"
    cases = [make_case(i, correct=True) for i in range(5)]
    cases += [make_case(i, correct=False) for i in range(5, 10)]
    save_proposer_cases(cases, inputs)
    digest = hashlib.sha256(inputs.read_bytes()).hexdigest()
    inputs.with_name(inputs.name + ".sha256").write_text(f"{digest}  {inputs.name}\n")
    run = tmp_path / "proposer-009b"
    return propose.Paths(
        inputs=inputs,
        ledger=run / "ledger.jsonl",
        out_dir=run / "pool",
        response_models=run / "response_models.jsonl",
        skill=S0_PATH,
        controls=ROOT / "skills" / "controls",
    )


def run(
    argv: list[str],
    fake: FakeOpenAI,
    paths: propose.Paths,
    *,
    now: Callable[[], datetime] | None = None,
    git: Callable[[Sequence[str]], int] = lambda argv: 0,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = propose.main(
        argv,
        root=ROOT,
        paths=paths,
        transport=fake.transport(),
        power_run=power_run,
        git=git,
        expected_export_sha256=hashlib.sha256(paths.inputs.read_bytes()).hexdigest(),
        now=now,
        out=out,
        err=err,
    )
    return code, out.getvalue(), err.getvalue()


def sessions(paths: propose.Paths) -> list[dict[str, Any]]:
    path = paths.ledger.parent / "sessions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# 고정 값 ------------------------------------------------------------------------


def test_fixed_values_follow_the_preregistration(paths: propose.Paths) -> None:
    argv = propose.v1_argv(paths, calls=1)
    flags = dict(zip(argv[1::2], argv[2::2], strict=True))
    assert argv[0] == "propose"
    assert flags["--model"] == "gpt-5.4" and flags["--reasoning-effort"] == "low"
    assert flags["--max-completion-tokens"] == "8000"
    assert flags["--input-usd-per-mtok"] == "2.5" and flags["--output-usd-per-mtok"] == "15.0"
    assert flags["--budget-usd"] == "5.0"
    assert flags["--max-total-calls"] == "120" and flags["--stop-when-pool"] == "40"
    assert flags["--previous"] == "valid" and flags["--dedupe"] == "word"
    assert flags["--dedupe-threshold"] == "0.55" and flags["--stop-after-idle"] == "25"
    assert flags["--quota-message"] == "no credits remaining"
    assert flags["--calls"] == "1"
    assert propose.FIXED_RUN_OPTIONS == {
        "previous": "valid",
        "dedupe": "word",
        "dedupe_threshold": 0.55,
        "stop_after_idle": 25,
    }


def test_default_paths_are_the_009b_paths() -> None:
    p = propose.default_paths(ROOT)
    assert p.ledger == ROOT / "runs/proposer-009b/ledger.jsonl"
    assert p.out_dir == ROOT / "runs/proposer-009b/pool"
    assert p.response_models == ROOT / "runs/proposer-009b/response_models.jsonl"
    assert (
        p.inputs
        == ROOT
        / "runs/proposer_inputs-009/dev_a__ollama__gpt-oss__20b__7c66ee5bf13c__bf65e569ae37.json"
    )
    assert Path("runs/proposer-009b/STOP") == propose.STOP_FILE


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "gpt-x"],
        ["--dedupe-threshold", "0.6"],
        ["--budget-usd", "9"],
        ["--previous", "pool"],
    ],
)
def test_fixed_values_cannot_be_overridden(paths: propose.Paths, extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        run(["propose", *extra], FakeOpenAI([]), paths)


# 실행과 기록 -----------------------------------------------------------------------


def test_a_session_records_options_models_and_sessions(
    paths: propose.Paths, tmp_path: Path
) -> None:
    fake = FakeOpenAI([completion(content(i)) for i in range(2)])
    code, out, err = run(["propose", "--calls", "2"], fake, paths)
    assert code == 0, err
    assert json.loads((paths.out_dir / "run_options.json").read_text()) == propose.FIXED_RUN_OPTIONS
    models = [json.loads(line) for line in paths.response_models.read_text().splitlines()]
    assert [m["model"] for m in models] == ["gpt-5.4-2026-03-05"] * 2
    events = sessions(paths)
    assert [e["event"] for e in events] == ["session_start", "session_end"]
    assert events[0]["calls_requested"] == 2 and events[0]["options"] == propose.FIXED_RUN_OPTIONS
    assert events[1]["reason"] == "calls_done" and events[1]["calls_made"] == 2
    assert "first response model: gpt-5.4-2026-03-05 (requested gpt-5.4)" in out
    body = fake.bodies[0]
    assert body["model"] == "gpt-5.4" and body["reasoning_effort"] == "low"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(), path
    assert FAKE_KEY not in out + err


def test_stop_file_ends_the_session_before_the_next_call(paths: propose.Paths) -> None:
    stop = paths.ledger.parent / "STOP"
    stop.parent.mkdir(parents=True)
    stop.touch()
    fake = FakeOpenAI([completion(content(0))])
    code, out, _ = run(["propose", "--calls", "3", "--stop-file", str(stop)], fake, paths)
    assert code == 0 and fake.bodies == []
    assert sessions(paths)[-1]["reason"] == "stop_file"
    assert "stop file" in out


def test_stop_at_ends_the_session_between_calls(paths: propose.Paths) -> None:
    times = iter(
        [datetime(2026, 11, 7, 9, 0), datetime(2026, 11, 7, 9, 5), datetime(2026, 11, 7, 9, 31)]
        + [datetime(2026, 11, 7, 9, 40)] * 10
    )
    fake = FakeOpenAI([completion(content(i)) for i in range(5)])
    code, _, _ = run(
        ["propose", "--calls", "5", "--stop-at", "09:30"], fake, paths, now=lambda: next(times)
    )
    assert code == 0 and len(fake.bodies) == 1
    end = sessions(paths)[-1]
    assert end["reason"] == "stop_at" and end["calls_made"] == 1
    assert sessions(paths)[0]["stop_at"].startswith("2026-11-07T09:30")


def test_preregistered_limits_end_the_session_as_limit(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(i, same=True)) for i in range(30)])
    code, _, _ = run(["propose", "--calls", "30"], fake, paths)
    assert code == 0
    assert len(fake.bodies) == 26  # 풀 1개 뒤 25회 연속 무증가
    end = sessions(paths)[-1]
    assert (
        end["reason"] == "limit" and "no new pool candidate in the last 25 calls" in end["detail"]
    )


def test_a_quota_error_ends_the_session_as_stopped(paths: propose.Paths) -> None:
    no_credits = httpx.Response(429, json={"error": {"message": "You have no credits remaining."}})
    fake = FakeOpenAI([no_credits])
    code, _, err = run(["propose", "--calls", "3"], fake, paths)
    assert code == 2 and "insufficient quota or balance" in err
    assert sessions(paths)[-1]["reason"] == "stopped" and sessions(paths)[-1]["exit_code"] == 2
    assert fake.requests == 1


def test_sessions_continue_from_saved_calls(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(i)) for i in range(3)])
    run(["propose", "--calls", "2"], fake, paths)
    code, _, _ = run(["propose", "--calls", "1"], fake, paths)
    assert code == 0 and (paths.out_dir / "call-002.json").exists()
    assert [e["event"] for e in sessions(paths)].count("session_start") == 2


# 실행 전 점검 ---------------------------------------------------------------------


def test_preflight_failure_sends_no_request_and_is_logged(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(0))])
    code, _, err = run(["propose"], fake, paths, git=lambda argv: 1)
    assert code == 1 and "prereg_tag" in err
    assert fake.requests == 0
    assert sessions(paths)[-1]["event"] == "preflight_failed"


def test_preflight_command_only_checks(paths: propose.Paths) -> None:
    fake = FakeOpenAI([])
    code, out, _ = run(["preflight"], fake, paths)
    assert code == 0 and fake.requests == 0
    assert "preflight ok   outputs" in out
    assert not paths.ledger.parent.exists()


def test_status_reports_calls_pool_idle_and_cost(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(0)), completion(content(0, same=True))])
    run(["propose", "--calls", "2"], fake, paths)
    code, out, _ = run(["status"], fake, paths)
    assert code == 0
    assert "calls 2, pool 1, idle 1" in out and "cost $" in out


def test_models_passes_through(paths: propose.Paths) -> None:
    fake = FakeOpenAI([])
    code, out, _ = run(["models"], fake, paths)
    assert code == 0 and out.split() == ["gpt-5.4"]


def test_stop_at_uses_the_same_clock_for_parsing_and_checking() -> None:
    now = datetime(2026, 11, 7, 23, 50)
    stop = propose.next_stop_at("00:10", now)
    assert stop == now.replace(hour=0, minute=10) + timedelta(days=1)


def test_ctrl_c_is_logged_as_interrupted(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(0))])

    def interrupt(request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    fake.handler = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        run(["propose", "--calls", "2"], fake, paths)
    end = sessions(paths)[-1]
    assert end["event"] == "session_end" and end["reason"] == "interrupted"
    assert end["calls_made"] == 0
