"""Ticket 009 proposer 진입점(사전 등록 2.1절). 가짜 전송으로만 테스트한다(API 호출 없음)."""

import io
import json
import shutil
import string
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.proposer_inputs import ProposerCase, save_proposer_cases
from evidencegate.t009 import propose

ROOT = Path(__file__).resolve().parents[2]
S0_PATH = ROOT / "skills" / "S0" / "SKILL.md"
FAKE_KEY = "sk-test-SECRET009-1234567890abcdef"


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


def word(i: int) -> str:
    """숫자 없는 고유 단어(후보 본문에는 숫자가 들어가면 안 된다)."""
    letters = string.ascii_lowercase
    return "check" + letters[i // 26 % 26] + letters[i % 26]


def content(i: int, *, change: bool = True) -> str:
    step = f"\n7. Before acting, {word(i)} the {word(i + 300)} signal.\n" if change else ""
    return json.dumps(
        {
            "skill_md": S0_PATH.read_text() + step,
            "failure_type": "WRONG_INTERVENTION",
            "hypothesis": f"H{i}",
            "expected_effect": {"cd": "up", "es": "up"},
        }
    )


def completion(
    text: str, *, model: str = "gpt-5.4-2026-03-05", tokens: int = 1000
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": tokens // 2},
    }


class FakeOpenAI:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-5.4"}, {"id": "gpt-x"}]})
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def paths(tmp_path: Path) -> propose.Paths:
    inputs = tmp_path / "inputs.json"
    cases = [make_case(i, correct=True) for i in range(5)]
    cases += [make_case(i, correct=False) for i in range(5, 10)]
    save_proposer_cases(cases, inputs)
    propose.write_sha256(inputs)
    run = tmp_path / "proposer-009"
    return propose.Paths(
        inputs=inputs,
        ledger=run / "ledger.jsonl",
        out_dir=run / "pool",
        response_models=run / "response_models.jsonl",
        skill=S0_PATH,
        controls=ROOT / "skills" / "controls",
    )


def run(
    argv: list[str], fake: FakeOpenAI, paths: propose.Paths, root: Path = ROOT
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = propose.main(argv, root=root, paths=paths, transport=fake.transport(), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


# 1. 고정 값 -------------------------------------------------------------------


def test_fixed_values_are_passed_to_the_v1_cli(paths: propose.Paths) -> None:
    argv = propose.v1_argv(paths, calls=3)
    expect = {
        "--model": "gpt-5.4",
        "--reasoning-effort": "low",
        "--max-completion-tokens": "8000",
        "--input-usd-per-mtok": "2.5",
        "--output-usd-per-mtok": "15.0",
        "--budget-usd": "5.0",
        "--max-total-calls": "80",
        "--stop-when-pool": "40",
        "--calls": "3",
        "--inputs": str(paths.inputs),
        "--ledger": str(paths.ledger),
        "--out-dir": str(paths.out_dir),
        "--skill": str(paths.skill),
        "--controls-dir": str(paths.controls),
    }
    assert argv[0] == "propose"
    pairs = dict(zip(argv[1::2], argv[2::2], strict=True))
    assert pairs == expect
    fake = FakeOpenAI([completion(content(0))])
    code, _, _ = run(["propose", "--calls", "1"], fake, paths)
    assert code == 0
    body = fake.bodies[0]
    assert body["model"] == "gpt-5.4" and body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 8000
    saved = json.loads((paths.out_dir / "call-000.json").read_text())
    assert saved["proposer_config"]["budget_usd"] == 5.0
    assert saved["proposer_config"]["prompt_sha256"] == propose.PROMPT_SHA256


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "gpt-x"],
        ["--budget-usd", "9"],
        ["--ledger", "x.jsonl"],
        ["--max-total-calls", "200"],
    ],
)
def test_fixed_values_cannot_be_overridden(paths: propose.Paths, extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        run(["propose", "--calls", "1", *extra], FakeOpenAI([]), paths)


def test_default_paths_are_the_009_paths() -> None:
    p = propose.Paths.default(ROOT)
    assert p.ledger == ROOT / "runs/proposer-009/ledger.jsonl"
    assert p.out_dir == ROOT / "runs/proposer-009/pool"
    assert p.response_models == ROOT / "runs/proposer-009/response_models.jsonl"
    assert p.inputs.parent == ROOT / "runs/proposer_inputs-009"


# 2. 응답 모델 기록, 키 비노출 -----------------------------------------------------


def test_response_model_is_recorded_and_the_key_is_nowhere(
    paths: propose.Paths, tmp_path: Path
) -> None:
    fake = FakeOpenAI([completion(content(0)), completion(content(1), model="gpt-5.4-other")])
    code, out, err = run(["propose", "--calls", "2"], fake, paths)
    assert code == 0
    rows = [json.loads(line) for line in paths.response_models.read_text().splitlines()]
    assert [r["model"] for r in rows] == ["gpt-5.4-2026-03-05", "gpt-5.4-other"]
    assert all(r["requested_model"] == "gpt-5.4" for r in rows)
    assert [r["call_index"] for r in rows] == [0, 1]
    assert set(rows[0]) == {"time", "call_index", "requested_model", "model"}
    assert "first response model: gpt-5.4-2026-03-05 (requested gpt-5.4)" in out
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(), path
    assert FAKE_KEY not in out and FAKE_KEY not in err


# 3·4·5. 멈춤 규칙 ---------------------------------------------------------------


def test_stops_when_the_pool_reaches_40(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(i)) for i in range(45)])
    code, out, _ = run(["propose", "--calls", "45"], fake, paths)
    assert code == 0
    assert len(fake.bodies) == 40
    assert "pool reached 40" in out
    assert len(list(paths.out_dir.glob("call-*.json"))) == 40


def test_stops_at_80_calls(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(i, change=False)) for i in range(85)])
    code, out, _ = run(["propose", "--calls", "85"], fake, paths)
    assert code == 0
    assert len(fake.bodies) == 80
    assert "max total calls 80 reached" in out
    code, _, _ = run(["propose", "--calls", "5"], fake, paths)  # 이어서 불러도 더 부르지 않는다
    assert code == 0 and len(fake.bodies) == 80


def test_stops_before_a_call_that_could_pass_the_5_usd_budget(paths: propose.Paths) -> None:
    # 첫 호출 비용: 1,000,000 x $2.5/M + 500,000 x $15/M = $10
    fake = FakeOpenAI([completion(content(0), tokens=1_000_000), completion(content(1))])
    code, _, err = run(["propose", "--calls", "2"], fake, paths)
    assert code == 2
    assert "budget" in err
    assert len(fake.bodies) == 1


# 6. 이전 hypothesis 목록 ---------------------------------------------------------


def test_previous_hypotheses_come_only_from_the_009_pool(paths: propose.Paths) -> None:
    fake = FakeOpenAI([completion(content(i)) for i in range(3)])
    code, _, _ = run(["propose", "--calls", "3"], fake, paths)
    assert code == 0
    users = [b["messages"][1]["content"] for b in fake.bodies]
    assert "(none yet)" in users[0]
    assert "- H0" in users[1] and "- H1" in users[2]
    v1 = [
        json.loads(p.read_text())["hypothesis"]
        for p in sorted((ROOT / "skills" / "candidates").glob("cand-*/evidence.json"))
    ]
    assert v1
    for text in users:
        for hypothesis in v1:
            assert hypothesis not in text


# 7. 호출 전 점검 -----------------------------------------------------------------


def copy_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    shutil.copytree(ROOT / "preregistration", root / "preregistration")
    return root


def test_preregistration_mismatch_stops_before_any_request(
    paths: propose.Paths, tmp_path: Path
) -> None:
    root = copy_root(tmp_path)
    prereg = root / "preregistration" / "ticket-009.txt"
    prereg.write_bytes(prereg.read_bytes() + b" ")
    fake = FakeOpenAI([completion(content(0))])
    code, _, err = run(["propose", "--calls", "1"], fake, paths, root=root)
    assert code == 1 and "preregistration" in err
    assert fake.requests == 0


def test_export_sha_mismatch_stops_before_any_request(paths: propose.Paths) -> None:
    paths.inputs.with_name(paths.inputs.name + ".sha256").write_text(f"{'0' * 64}  x\n")
    fake = FakeOpenAI([completion(content(0))])
    code, _, err = run(["propose", "--calls", "1"], fake, paths)
    assert code == 1 and "sha256" in err
    assert fake.requests == 0


@pytest.mark.parametrize("which", ["ledger", "out_dir"])
def test_v1_paths_stop_before_any_request(paths: propose.Paths, tmp_path: Path, which: str) -> None:
    root = copy_root(tmp_path)
    v1 = {"ledger": root / "runs/proposer/ledger.jsonl", "out_dir": root / "runs/proposer/pool"}
    bad = propose.Paths(**{**paths.__dict__, which: v1[which]})
    fake = FakeOpenAI([completion(content(0))])
    code, _, err = run(["propose", "--calls", "1"], fake, bad, root=root)
    assert code == 1 and "v1" in err
    assert fake.requests == 0


def test_thin_strata_stop_before_any_request(paths: propose.Paths) -> None:
    cases = [make_case(0, correct=True)] + [make_case(i, correct=False) for i in range(1, 10)]
    save_proposer_cases(cases, paths.inputs)
    propose.write_sha256(paths.inputs)
    fake = FakeOpenAI([completion(content(0))])
    code, _, err = run(["propose", "--calls", "1"], fake, paths)
    assert code == 1 and "stratum" in err
    assert fake.requests == 0


# 8. models --------------------------------------------------------------------


def test_models_passes_through_to_the_v1_cli(paths: propose.Paths) -> None:
    fake = FakeOpenAI([])
    code, out, _ = run(["models"], fake, paths)
    assert code == 0
    assert out.split() == ["gpt-5.4", "gpt-x"]
    assert not paths.response_models.exists()


def test_only_successful_completions_are_recorded(
    paths: propose.Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evidencegate.proposer.time.sleep", lambda _: None)
    fake = FakeOpenAI([completion(content(0))])
    answer = fake.handler
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:  # 첫 시도는 서버 오류, 재시도는 성공
            return httpx.Response(500, json={"error": {"message": "busy", "model": "x"}})
        return answer(request)

    fake.handler = flaky  # type: ignore[method-assign]
    code, _, _ = run(["propose", "--calls", "1"], fake, paths)
    assert code == 0
    rows = paths.response_models.read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["model"] == "gpt-5.4-2026-03-05"
