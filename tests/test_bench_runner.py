"""Ticket 005 실행기(§2): 블록 실행, 이어서 실행, 블록 수 제한, 중단 규칙, 진행 확인."""

import io
import json
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate.bench_runner import (
    BenchStopped,
    FormatErrorWindow,
    StopRules,
    block_store,
    main,
    run_blocks,
    status_line,
)
from evidencegate.eval_data import split_cases
from evidencegate.ollama_policy import OllamaError
from evidencegate.policy import PolicyTrace, StopReason, call_tool
from evidencegate.preflight import Check
from evidencegate.run_plan import Block
from evidencegate.skills import Skill, load_skill
from opsworld.agent_env import AgentEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
S0 = load_skill(REPO_ROOT / "skills" / "S0" / "SKILL.md")
CAND = load_skill(REPO_ROOT / "skills" / "candidates" / "cand-000" / "SKILL.md")
FAST = StopRules(retry_wait_s=0.0)


class FakePolicy:
    """observe_only로 끝낸다. 정해진 (seed, 반복)에서 서버 오류를 내거나 형식 오류를 기록한다."""

    policy_id = "fake:policy"

    def __init__(
        self,
        *,
        fail: Callable[[int, int], bool] = lambda seed, attempt: False,
        format_error: Callable[[int], bool] = lambda seed: False,
    ) -> None:
        self.config: dict[str, Any] = {"type": "fake"}
        self.fail = fail
        self.format_error = format_error
        self.calls: dict[int, int] = {}
        self.lock = threading.Lock()

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        seed = env._episode.incident.seed
        with self.lock:
            attempt = self.calls.get(seed, 0)
            self.calls[seed] = attempt + 1
        if self.fail(seed, attempt):
            raise OllamaError("request to Ollama failed: connection reset")
        _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
        return PolicyTrace((turn,), int(self.format_error(seed)), StopReason.ACTION)


def block(index: int, name: str, skill: Skill, path: str, split: str, repeat: int, n: int) -> Block:
    return Block(
        index, name, "proposer", skill.skill_id, path, split, repeat, tuple(split_cases(split)[:n])
    )


def small_plan(n: int = 4) -> list[Block]:
    return [
        block(0, "cand-000", CAND, "skills/candidates/cand-000/SKILL.md", "dev_a", 0, n),
        block(1, "S0", S0, "skills/S0/SKILL.md", "oracle", 3, n),
        block(2, "cand-000", CAND, "skills/candidates/cand-000/SKILL.md", "dev_b", 0, n),
    ]


def run(tmp_path: Path, plan: list[Block], policy: FakePolicy, **kw: Any) -> Any:
    kw.setdefault("rules", FAST)
    return run_blocks(
        plan, policy, root=REPO_ROOT, runs_dir=tmp_path / "runs", out=io.StringIO(), **kw
    )


def stored(tmp_path: Path, plan: list[Block], policy: FakePolicy) -> list[Any]:
    return [r for b in plan for r in block_store(tmp_path / "runs", policy, b).records()]


# ---------------------------------------------------------------------------
# 블록 실행과 이어서 실행
# ---------------------------------------------------------------------------


def test_every_block_lands_in_its_split_store_with_its_repeat(tmp_path: Path) -> None:
    plan = small_plan()
    summary = run(tmp_path, plan, FakePolicy())
    assert (summary.episodes_run, summary.blocks_run) == (12, 3)
    for b in plan:
        records = block_store(tmp_path / "runs", FakePolicy(), b).records()
        assert sorted((r.key.family, r.key.level, r.key.incident_seed) for r in records) == sorted(
            b.cases
        )
        assert {r.key.repeat for r in records} == {b.repeat}
        assert {r.key.skill_id for r in records} == {b.skill_id}
        assert block_store(tmp_path / "runs", FakePolicy(), b).path.parts[-4] == b.split


def test_running_again_runs_nothing(tmp_path: Path) -> None:
    plan = small_plan()
    run(tmp_path, plan, FakePolicy())
    again = run(tmp_path, plan, FakePolicy())
    assert (again.episodes_run, again.blocks_run, again.blocks_skipped) == (0, 0, 3)
    assert len(stored(tmp_path, plan, FakePolicy())) == 12


def test_partially_stored_block_runs_only_the_missing_cases(tmp_path: Path) -> None:
    plan = small_plan()
    run(tmp_path, plan[:1], FakePolicy())
    summary = run(tmp_path, plan, FakePolicy())
    assert summary.episodes_run == 8
    assert len(stored(tmp_path, plan, FakePolicy())) == 12


def test_max_blocks_counts_only_blocks_that_had_work(tmp_path: Path) -> None:
    plan = small_plan()
    first = run(tmp_path, plan, FakePolicy(), max_blocks=1)
    assert (first.blocks_run, first.episodes_run) == (1, 4)
    second = run(tmp_path, plan, FakePolicy(), max_blocks=1)
    assert (second.blocks_run, second.blocks_skipped, second.episodes_run) == (1, 1, 4)
    assert len(stored(tmp_path, plan, FakePolicy())) == 8


def test_two_workers_store_every_episode_once(tmp_path: Path) -> None:
    plan = small_plan(n=10)
    summary = run(tmp_path, plan, FakePolicy(), workers=2)
    records = stored(tmp_path, plan, FakePolicy())
    assert summary.episodes_run == 30 == len(records)
    assert (
        len(
            {
                (r.key.skill_id, r.key.family, r.key.level, r.key.incident_seed, r.key.repeat)
                for r in records
            }
        )
        == 30
    )


def test_skill_file_must_match_the_block(tmp_path: Path) -> None:
    bad = [
        Block(
            0,
            "x",
            "proposer",
            "000000000000",
            "skills/S0/SKILL.md",
            "dev_a",
            0,
            tuple(split_cases("dev_a")[:2]),
        )
    ]
    with pytest.raises(ValueError, match="skill_id"):
        run(tmp_path, bad, FakePolicy())


# ---------------------------------------------------------------------------
# 형식 오류 창: 에피소드 단위, 최근 200개, 20개째부터, > 5%
# ---------------------------------------------------------------------------


def test_window_does_not_judge_before_20_episodes() -> None:
    w = FormatErrorWindow(StopRules())
    for _ in range(19):
        w.add(True)
    assert not w.exceeded()
    w.add(True)
    assert w.exceeded()


def test_window_10_of_200_continues_and_11_stops() -> None:
    w = FormatErrorWindow(StopRules())
    for i in range(200):
        w.add(i < 10)
    assert (w.errors, w.size) == (10, 200) and not w.exceeded()
    w.add(True)  # 맨 앞(오류)이 빠지고 새 오류가 들어와 여전히 10/200
    assert (w.errors, w.size) == (10, 200) and not w.exceeded()
    w2 = FormatErrorWindow(StopRules())
    for i in range(200):
        w2.add(i < 11)
    assert (w2.errors, w2.size) == (11, 200) and w2.exceeded()


def test_window_rate_is_exact_before_200() -> None:
    """처음 200개 전에는 지금까지의 개수로: 2/40 = 5%는 넘지 않음, 3/40은 넘음."""
    w = FormatErrorWindow(StopRules())
    for i in range(40):
        w.add(i < 2)
    assert w.rate == Fraction(2, 40) and not w.exceeded()
    w.add(True)
    assert w.exceeded()  # 3/41 > 1/20


def test_format_error_rate_stops_the_run(tmp_path: Path) -> None:
    plan = small_plan(n=30)
    with pytest.raises(BenchStopped, match="format error") as info:
        run(tmp_path, plan, FakePolicy(format_error=lambda seed: True))
    assert len(stored(tmp_path, plan, FakePolicy())) == 20  # 20개째에서 판정
    assert info.value.last is not None and "correct" not in json.dumps(info.value.last)


def test_window_includes_episodes_stored_in_earlier_sessions(tmp_path: Path) -> None:
    plan = small_plan(n=12)
    run(
        tmp_path,
        plan[:1],
        FakePolicy(format_error=lambda seed: True),
        rules=StopRules(retry_wait_s=0.0, min_episodes=100),
    )
    with pytest.raises(BenchStopped, match="format error"):
        run(
            tmp_path,
            plan,
            FakePolicy(format_error=lambda seed: True),
            rules=StopRules(retry_wait_s=0.0, min_episodes=20),
        )
    # 앞 세션 12개 + 이번 세션 8개 = 20개째에서 멈춘다
    assert len(stored(tmp_path, plan, FakePolicy())) == 20


# ---------------------------------------------------------------------------
# 서버 오류: 같은 에피소드 3회까지 재시도, 연속 3개 에피소드 실패면 멈춤
# ---------------------------------------------------------------------------


def test_server_error_is_retried_and_the_episode_is_stored(tmp_path: Path) -> None:
    plan = small_plan(n=3)
    policy = FakePolicy(
        fail=lambda seed, attempt: attempt < 3
    )  # 처음 3번 실패, 4번째(재시도 3회째) 성공
    summary = run(tmp_path, plan, policy)
    assert summary.episodes_run == 9 and summary.server_errors == 9 * 3
    assert len(stored(tmp_path, plan, FakePolicy())) == 9


def test_three_consecutive_failed_episodes_stop_the_run(tmp_path: Path) -> None:
    plan = small_plan(n=5)
    policy = FakePolicy(fail=lambda seed, attempt: True)
    with pytest.raises(BenchStopped, match="consecutive"):
        run(tmp_path, plan, policy, workers=1)
    assert len(stored(tmp_path, plan, FakePolicy())) == 0
    assert sum(policy.calls.values()) == 3 * 4  # 에피소드 3개 x (처음 1회 + 재시도 3회)


def test_with_two_workers_at_most_the_in_flight_episode_runs_after_the_stop(tmp_path: Path) -> None:
    """동시 2: 세 번째 실패 때 이미 돌고 있던 에피소드 하나는 끝까지 돈다(새로 시작하지는 않음)."""
    plan = small_plan(n=10)
    policy = FakePolicy(fail=lambda seed, attempt: True)
    with pytest.raises(BenchStopped, match="consecutive"):
        run(tmp_path, plan, policy, workers=2)
    assert len(policy.calls) <= 4


def test_a_success_resets_the_consecutive_failure_count(tmp_path: Path) -> None:
    """실패, 실패, 성공, 실패, 실패, 성공 ... 은 멈추지 않는다."""
    dev_a = split_cases("dev_a")[:6]
    failing = {dev_a[i][2] for i in (0, 1, 3, 4)}
    plan = [block(0, "cand-000", CAND, "skills/candidates/cand-000/SKILL.md", "dev_a", 0, 6)]
    summary = run(tmp_path, plan, FakePolicy(fail=lambda seed, attempt: seed in failing))
    assert (summary.episodes_run, summary.episodes_failed) == (2, 4)


def test_failed_episodes_run_again_on_resume(tmp_path: Path) -> None:
    dev_a = split_cases("dev_a")[:4]
    plan = [block(0, "cand-000", CAND, "skills/candidates/cand-000/SKILL.md", "dev_a", 0, 4)]
    run(tmp_path, plan, FakePolicy(fail=lambda seed, attempt: seed == dev_a[0][2]))
    assert len(stored(tmp_path, plan, FakePolicy())) == 3
    run(tmp_path, plan, FakePolicy())
    assert len(stored(tmp_path, plan, FakePolicy())) == 4


# ---------------------------------------------------------------------------
# 진행 확인: CD·ES는 내지 않는다
# ---------------------------------------------------------------------------


def test_status_line_reports_progress_without_outcomes(tmp_path: Path) -> None:
    plan = small_plan()
    run(
        tmp_path,
        plan[:2],
        FakePolicy(format_error=lambda seed: seed % 2 == 0),
        rules=StopRules(retry_wait_s=0.0, min_episodes=100),
    )
    line = status_line(plan, tmp_path / "runs", FakePolicy())
    assert "8/12 episodes" in line and "2/3 blocks" in line
    assert "format errors (last 200)" in line and "server errors" in line
    for word in ("correct", "evidence", "CD", "ES", "disposition"):
        assert word not in line
    assert "\n" not in line


def test_progress_output_has_no_outcomes(tmp_path: Path) -> None:
    out = io.StringIO()
    run_blocks(
        small_plan(), FakePolicy(), root=REPO_ROOT, runs_dir=tmp_path / "runs", out=out, rules=FAST
    )
    text = out.getvalue()
    assert "[12/12]" in text or "12/12" in text
    for word in ("correct", "evidence_safe", "disposition"):
        assert word not in text


def test_session_log_records_start_end_and_server_errors(tmp_path: Path) -> None:
    plan = small_plan(n=2)
    run(tmp_path, plan, FakePolicy(fail=lambda seed, attempt: attempt < 1))
    log = [
        json.loads(line)
        for line in (tmp_path / "runs" / "ticket005" / "sessions.jsonl").read_text().splitlines()
    ]
    events = [e["event"] for e in log]
    assert events[0] == "session_start" and events[-1] == "session_end"
    assert events.count("server_error") == 6
    assert log[-1]["episodes_run"] == 6


# ---------------------------------------------------------------------------
# 명령: 점검·탐침 실패면 실행하지 않음, 파일럿은 블록 2개
# ---------------------------------------------------------------------------


class Probe:
    def __init__(self, parallel: bool) -> None:
        self.parallel = parallel
        self.ratio = 1.3 if parallel else 1.95
        self.ratios = (self.ratio,)
        self.single_s = 5.0
        self.concurrent_s = 6.5
        self.max_ratio = 1.68


def cli(
    tmp_path: Path,
    args: list[str],
    *,
    checks_ok: bool = True,
    parallel: bool = True,
    plan: list[Block] | None = None,
) -> tuple[int, str]:
    out = io.StringIO()
    code = main(
        [*args, "--runs-dir", str(tmp_path / "runs")],
        policy_factory=FakePolicy,
        preflight=lambda policy: [Check("engine", checks_ok, "fake")],
        probe=lambda: Probe(parallel),
        plan=plan or small_plan(),
        out=out,
        err=out,
    )
    return code, out.getvalue()


def test_failed_preflight_runs_nothing(tmp_path: Path) -> None:
    code, text = cli(tmp_path, ["run"], checks_ok=False)
    assert code == 1 and "preflight failed" in text
    assert not (tmp_path / "runs" / "dev_a").exists()


def test_failed_parallel_probe_runs_nothing(tmp_path: Path) -> None:
    code, text = cli(tmp_path, ["run"], parallel=False)
    assert code == 1 and "parallel" in text
    assert not (tmp_path / "runs" / "dev_a").exists()


def test_pilot_runs_two_blocks_and_stops(tmp_path: Path) -> None:
    code, text = cli(tmp_path, ["run", "--max-blocks", "2"])
    assert code == 0 and "max blocks reached" in text
    assert len(stored(tmp_path, small_plan(), FakePolicy())) == 8


def test_stop_exits_2_with_the_reason(tmp_path: Path) -> None:
    out = io.StringIO()
    code = main(
        ["run", "--runs-dir", str(tmp_path / "runs")],
        policy_factory=lambda: FakePolicy(fail=lambda seed, attempt: True),
        preflight=lambda policy: [Check("engine", True, "fake")],
        probe=lambda: Probe(True),
        plan=small_plan(),
        out=out,
        err=out,
        rules=FAST,
    )
    assert code == 2 and "stopped:" in out.getvalue() and "consecutive" in out.getvalue()


def test_status_command_needs_no_policy_connection(tmp_path: Path) -> None:
    out = io.StringIO()
    code = main(
        ["status", "--runs-dir", str(tmp_path / "runs")],
        plan=small_plan(),
        out=out,
        err=out,
        status_policy=FakePolicy(),
    )
    assert code == 0 and "0/12 episodes" in out.getvalue()


def test_pilot_flag_runs_only_the_first_two_plan_blocks_even_when_repeated(tmp_path: Path) -> None:
    """파일럿은 계획의 첫 두 블록만. 다시 쳐도 다음 블록으로 넘어가지 않는다."""
    code, text = cli(tmp_path, ["run", "--pilot"])
    assert code == 0
    assert len(stored(tmp_path, small_plan()[:2], FakePolicy())) == 8
    code, text = cli(tmp_path, ["run", "--pilot"])
    assert code == 0 and "0 episodes" in text
    assert len(stored(tmp_path, small_plan(), FakePolicy())) == 8


def test_blocks_of_one_store_with_different_repeats_are_kept_apart(tmp_path: Path) -> None:
    """S0 Oracle 반복 0~4는 한 저장소 파일을 함께 쓴다. 반복 0이 끝나도 반복 1은 따로 돈다."""
    plan = [
        block(0, "S0", S0, "skills/S0/SKILL.md", "oracle", 0, 3),
        block(1, "S0", S0, "skills/S0/SKILL.md", "oracle", 1, 3),
    ]
    run(tmp_path, plan[:1], FakePolicy())
    summary = run(tmp_path, plan, FakePolicy())
    assert (summary.blocks_skipped, summary.blocks_run, summary.episodes_run) == (1, 1, 3)
    records = block_store(tmp_path / "runs", FakePolicy(), plan[0]).records()
    assert sorted(r.key.repeat for r in records) == [0, 0, 0, 1, 1, 1]
