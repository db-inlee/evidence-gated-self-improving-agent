"""Ticket 014 proposer 진입점(사전 등록 3절). 가짜 전송으로만 시험한다(API 호출 없음)."""

import hashlib
import io
import json
import shutil
import string
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_pool_014 import dev_a, write_dev_a

from evidencegate.t014 import pool, propose
from evidencegate.t014.settings import fixed_run_options

ROOT = Path(__file__).resolve().parents[2]
S0_TEXT = (ROOT / "skills/S0/SKILL.md").read_text()
FAKE_KEY = "sk-test-SECRET014-1234567890abcdef"
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


def word(i: int) -> str:
    letters = string.ascii_lowercase
    return "check" + letters[i // 26 % 26] + letters[i % 26]


def content(i: int, *, text: str | None = None) -> str:
    skill = text or S0_TEXT + f"\n7. Before acting, {word(i)} the {word(i + 300)} signal.\n"
    return json.dumps(
        {
            "skill_md": skill,
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
def root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for sub in ("skills/S0", "skills/controls", "skills/candidates-009b"):
        shutil.copytree(ROOT / sub, root / sub)
    prereg = root / "preregistration/ticket-014.txt"
    prereg.parent.mkdir(parents=True)
    prereg.write_text("frozen text\n")
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    (root / "preregistration/ticket-014.sha256").write_text(f"sha256: {digest}\n")
    write_dev_a(root / "runs", dev_a())
    assert pool.export_main(root / "runs", root / propose.EXPORT_PATH) == 0
    return root


def run(
    argv: list[str],
    fake: FakeOpenAI,
    root: Path,
    *,
    now: Callable[[], datetime] | None = None,
    git: Callable[[Sequence[str]], int] = lambda argv: 0,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = propose.main(
        argv,
        root=root,
        transport=fake.transport(),
        power_run=power_run,
        git=git,
        now=now,
        out=out,
        err=err,
    )
    return code, out.getvalue(), err.getvalue()


def calls(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "runs/proposer-014/pool").glob("call-*.json"))
    return [json.loads(f.read_text()) for f in files]


def sessions(root: Path) -> list[dict[str, Any]]:
    path = root / "runs/proposer-014/sessions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fixed_values_are_the_009b_values_plus_the_exclude_dir(root: Path) -> None:
    paths = propose.default_paths(root)
    argv = propose.v1_argv(root, paths, calls=1)
    flags = dict(zip(argv[1::2], argv[2::2], strict=True))
    assert argv[0] == "propose" and flags["--calls"] == "1"
    assert flags["--model"] == "gpt-5.4" and flags["--reasoning-effort"] == "low"
    assert flags["--max-total-calls"] == "120" and flags["--stop-when-pool"] == "40"
    assert flags["--budget-usd"] == "5.0" and flags["--stop-after-idle"] == "25"
    assert flags["--previous"] == "valid" and flags["--dedupe"] == "word"
    assert flags["--dedupe-threshold"] == "0.55"
    assert flags["--exclude-dir"] == str(root / "skills/candidates-009b")
    assert flags["--inputs"] == str(root / propose.EXPORT_PATH)
    assert paths.ledger == root / "runs/proposer-014/ledger.jsonl"
    assert paths.out_dir == root / "runs/proposer-014/pool"


def test_a_session_writes_the_014_run_options_and_no_key(root: Path) -> None:
    fake = FakeOpenAI([completion(content(i)) for i in range(2)])
    code, out, err = run(["propose", "--calls", "2"], fake, root)
    assert code == 0, err
    options = json.loads((root / "runs/proposer-014/pool/run_options.json").read_text())
    assert options == fixed_run_options(root)
    assert [e["event"] for e in sessions(root)] == ["session_start", "session_end"]
    assert sessions(root)[-1]["reason"] == "calls_done"
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt", ".md"}:
            assert FAKE_KEY not in path.read_text(errors="ignore"), path
    assert FAKE_KEY not in out + err


def test_a_copy_of_a_009b_candidate_is_excluded_from_the_pool(root: Path) -> None:
    old = (root / "skills/candidates-009b/cand-000/SKILL.md").read_text()
    fake = FakeOpenAI([completion(content(0, text=old)), completion(content(1))])
    code, out, _ = run(["propose", "--calls", "2"], fake, root)
    assert code == 0
    first, second = calls(root)
    assert first["excluded_by"] == "cand-000" and first["in_pool"] is False
    assert second.get("excluded_by") is None  # 제외 폴더와 다른 후보는 빼지 않는다
    assert "excluded as duplicate of cand-000" in out


def test_preflight_failures_send_no_request(root: Path) -> None:
    fake = FakeOpenAI([completion(content(0))])
    code, _, err = run(["propose"], fake, root, git=lambda argv: 1)
    assert code == 1 and "prereg_tag" in err and fake.requests == 0
    assert sessions(root)[-1]["event"] == "preflight_failed"


def test_preflight_checks_the_preregistration_hash(root: Path) -> None:
    (root / "preregistration/ticket-014.txt").write_text("edited\n")
    code, out, _ = run(["preflight"], FakeOpenAI([]), root)
    assert code == 1 and "FAIL preregistration_014" in out


def test_preflight_checks_the_exclude_dir_and_the_export(root: Path) -> None:
    shutil.rmtree(root / "skills/candidates-009b/cand-000")
    code, out, _ = run(["preflight"], FakeOpenAI([]), root)
    assert code == 1 and "FAIL exclude_dir" in out
    export = root / propose.EXPORT_PATH
    export.write_text("[]\n")
    code, out, _ = run(["preflight"], FakeOpenAI([]), root)
    assert "FAIL export" in out


def test_preflight_passes_on_a_fresh_root(root: Path) -> None:
    code, out, _ = run(["preflight"], FakeOpenAI([]), root)
    assert code == 0, out
    assert not (root / "runs/proposer-014").exists()


def test_status_and_stop_file(root: Path) -> None:
    fake = FakeOpenAI([completion(content(0))])
    run(["propose", "--calls", "1"], fake, root)
    code, out, _ = run(["status"], fake, root)
    assert code == 0 and "calls 1, pool 1, idle 0" in out
    (root / "runs/proposer-014/STOP").touch()
    code, out, _ = run(["propose", "--calls", "1"], FakeOpenAI([]), root)
    assert code == 0 and sessions(root)[-1]["reason"] == "stop_file"
