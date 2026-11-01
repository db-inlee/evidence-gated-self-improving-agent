"""실행기 프로세스 단위 동시 실행 (2026-11-01 결정 5). 가짜 작업자로 돌린다(LLM 없음).

작업자는 spawn된 프로세스에서 돌아야 하므로 이 모듈의 최상위 클래스로 둔다(pickle 가능).
"""

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.agent import EpisodeTrace, PolicySettings, StepTrace
from alfworld_pilot.runner import (
    BLOCK_SIZE,
    TIME_FIELDS,
    Episode,
    RunSettings,
    RunStopped,
    plan_episodes,
    run_plan,
)


@dataclass(frozen=True)
class FakeWorker:
    """seed만으로 정해지는 궤적. 순서가 섞이도록 seed에 따라 조금씩 기다린다."""

    crash_keys: frozenset[str] = frozenset()

    def __call__(self, episode: Episode) -> EpisodeTrace:
        if episode.key in self.crash_keys:
            raise ConnectionError(f"ollama down at {episode.key}")
        time.sleep((episode.llm_seed % 4) * 0.03)
        n = 1 + episode.llm_seed % 3
        steps = tuple(
            StepTrace(f"go to shelf {i}", f"obs {episode.llm_seed} {i}", True) for i in range(n)
        )
        return EpisodeTrace(
            steps=steps,
            won=episode.llm_seed % 2 == 0,
            stop_reason="won" if episode.llm_seed % 2 == 0 else "max_steps",
            first_observation=f"room {episode.rel}",
            format_errors=0,
            text_turns=0,
            llm_calls=n,
            prompt_tokens=(10,) * n,
            completion_tokens=(1,) * n,
            truncated_calls=0,
            max_omitted=0,
            wall_s=0.01,
        )


@dataclass
class PowerCounter:
    calls: int = 0

    def __call__(self) -> None:
        self.calls += 1


def games(n: int) -> list[dict[str, Any]]:
    return [
        {"rel": f"train/pick_and_place_simple-X-{i}/t", "task_type": "pick_and_place_simple"}
        for i in range(n)
    ]


CONFIG = {"policy": PolicySettings().config(), "skill_sha256": "s" * 64}


def go(
    out: Path,
    plan: list[Episode],
    concurrency: int,
    worker: FakeWorker,
    power: PowerCounter | None = None,
) -> None:
    run_plan(
        plan,
        out,
        run_episode=worker,
        check_power=power or PowerCounter(),
        run_config=CONFIG,
        run_settings=RunSettings(concurrency=concurrency, server_num_parallel=2),
    )


def read(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.read_text().splitlines()]


def content(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        ({k: v for k, v in r.items() if k not in TIME_FIELDS} for r in records),
        key=lambda r: r["key"],
    )


def test_concurrency_one_and_two_give_the_same_records(tmp_path: Path) -> None:
    plan = plan_episodes(games(BLOCK_SIZE + 3), repeats=1)
    go(tmp_path / "c1.jsonl", plan, 1, FakeWorker())
    go(tmp_path / "c2.jsonl", plan, 2, FakeWorker())
    one, two = read(tmp_path / "c1.jsonl"), read(tmp_path / "c2.jsonl")
    assert len(two) == len(plan)
    assert content(one) == content(two)
    assert {"wall_s", "started_at", "finished_at", "run"} <= set(TIME_FIELDS)


def test_concurrency_is_a_run_setting_not_a_policy_setting(tmp_path: Path) -> None:
    out = tmp_path / "e.jsonl"
    go(out, plan_episodes(games(2), repeats=1), 2, FakeWorker())
    r = read(out)[0]
    assert r["run"] == {"concurrency": 2, "server_num_parallel": 2}
    assert "concurrency" not in json.dumps(r["config"])
    assert r["finished_at"] >= r["started_at"]


def test_power_is_still_checked_at_start_and_after_each_block(tmp_path: Path) -> None:
    plan = plan_episodes(games(2 * BLOCK_SIZE), repeats=1)
    power = PowerCounter()
    go(tmp_path / "e.jsonl", plan, 2, FakeWorker(), power)
    assert power.calls == 1 + 2


def test_one_worker_error_stops_the_run_and_resume_has_no_duplicates_or_gaps(
    tmp_path: Path,
) -> None:
    out = tmp_path / "e.jsonl"
    plan = plan_episodes(games(2 * BLOCK_SIZE + 2), repeats=1)
    bad = plan[2].key
    with pytest.raises(RunStopped, match="ollama down"):
        go(out, plan, 2, FakeWorker(crash_keys=frozenset({bad})))
    first = [r["key"] for r in read(out)] if out.exists() else []
    assert bad not in first
    # 멈춘 블록 뒤의 에피소드는 시작하지 않는다
    assert set(first) <= {e.key for e in plan[:BLOCK_SIZE]}
    go(out, plan, 2, FakeWorker())
    keys = [r["key"] for r in read(out)]
    assert len(keys) == len(set(keys))
    assert set(keys) == {e.key for e in plan}


def test_resume_may_change_concurrency_but_not_the_policy_config(tmp_path: Path) -> None:
    out = tmp_path / "e.jsonl"
    plan = plan_episodes(games(4), repeats=1)
    go(out, plan[:2], 1, FakeWorker())
    go(out, plan, 2, FakeWorker())
    assert [r["run"]["concurrency"] for r in sorted(read(out), key=lambda r: r["key"])].count(
        2
    ) == 2


def test_a_broken_line_stops_resume(tmp_path: Path) -> None:
    out = tmp_path / "e.jsonl"
    plan = plan_episodes(games(2), repeats=1)
    go(out, plan[:1], 1, FakeWorker())
    with out.open("a") as f:
        f.write('{"key": "partial')
    with pytest.raises(RunStopped, match="line 2"):
        go(out, plan, 2, FakeWorker())


def test_concurrency_cannot_exceed_the_declared_server_parallelism() -> None:
    with pytest.raises(ValueError):
        RunSettings(concurrency=2, server_num_parallel=1)
    with pytest.raises(ValueError):
        RunSettings(concurrency=0, server_num_parallel=1)


@dataclass(frozen=True)
class SlowWorker:
    def __call__(self, episode: Episode) -> EpisodeTrace:
        time.sleep(0.4)
        return FakeWorker()(episode)


def test_two_workers_really_overlap(tmp_path: Path) -> None:
    out = tmp_path / "e.jsonl"
    run_plan(
        plan_episodes(games(BLOCK_SIZE), repeats=1),
        out,
        run_episode=SlowWorker(),
        check_power=PowerCounter(),
        run_config=CONFIG,
        run_settings=RunSettings(concurrency=2, server_num_parallel=2),
    )
    spans = sorted((r["started_at"], r["finished_at"]) for r in read(out))
    assert any(b[0] < a[1] for a, b in itertools.pairwise(spans))


def test_real_worker_is_picklable() -> None:
    import pickle

    from alfworld_pilot.runner import AlfworldWorker

    worker = AlfworldWorker(
        skill_text="s", settings=PolicySettings(), json_root="/x", base_url="http://h"
    )
    assert pickle.loads(pickle.dumps(worker)) == worker


def test_a_repeated_episode_line_stops_resume(tmp_path: Path) -> None:
    out = tmp_path / "e.jsonl"
    plan = plan_episodes(games(2), repeats=1)
    go(out, plan[:1], 1, FakeWorker())
    line = out.read_text()
    out.write_text(line + line)
    with pytest.raises(RunStopped, match="repeats"):
        go(out, plan, 2, FakeWorker())
