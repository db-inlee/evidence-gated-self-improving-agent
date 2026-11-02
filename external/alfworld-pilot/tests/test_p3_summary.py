"""P3 요약 (Ticket 008 §4 P3). 가짜 기록으로 시험한다."""

from typing import Any

import pytest

from alfworld_pilot.p3 import (
    load_seconds,
    memory_summary,
    principle_9_case,
    stats,
    summarize,
)


def record(
    key: str,
    task_type: str,
    *,
    won: bool,
    started: float,
    finished: float,
    prompts: tuple[int, ...] = (100, 200),
    steps: int = 2,
    stop: str = "won",
    truncated: int = 0,
    take_seen: bool = True,
) -> dict[str, Any]:
    first = "You are in the middle of a room. Looking quickly around you, you see a shelf 1."
    seen = "You arrive at shelf 1. On the shelf 1, you see a pen 1." if take_seen else "x"
    trace = [{"command": "go to shelf 1", "observation": seen, "admissible": True}]
    trace += [
        {
            "command": "take pen 1 from shelf 1",
            "observation": "You pick up the pen 1 from the shelf 1." if won else "Nothing happens.",
            "admissible": won,
        }
    ] * (steps - 1)
    return {
        "key": key,
        "rel": key.split("#")[0],
        "task_type": task_type,
        "repeat": 0,
        "won": won,
        "stop_reason": stop,
        "steps": trace,
        "first_observation": first,
        "format_errors": 1,
        "text_turns": 0,
        "llm_calls": len(prompts),
        "prompt_tokens": list(prompts),
        "completion_tokens": [10] * len(prompts),
        "truncated_calls": truncated,
        "max_omitted": 0,
        "wall_s": finished - started,
        "started_at": started,
        "finished_at": finished,
        "run": {"concurrency": 2, "server_num_parallel": 2},
    }


def test_stats_mean_median_max() -> None:
    assert stats([1, 2, 6]) == {"n": 3, "mean": 3.0, "median": 2, "min": 1, "max": 6}
    assert stats([]) == {"n": 0, "mean": None, "median": None, "min": None, "max": None}


def test_principle_9_cases() -> None:
    assert principle_9_case(0, 12) == "zero_successes"
    assert principle_9_case(12, 12) == "all_successes"
    assert principle_9_case(5, 12) == "not_applicable"


def test_load_seconds_reads_the_server_log_lines() -> None:
    log = (
        'time=... msg="waiting for llama-server to start responding"\n'
        'time=2026-11-02T23:22:40 level=INFO msg="llama-server started in 28.48 seconds"\n'
        "unrelated\n"
    )
    assert load_seconds(log) == [28.48]
    assert load_seconds("nothing") == []


def test_memory_summary_reads_pressure_and_ollama_ps() -> None:
    log = (
        "2026-11-02 23:23:00\n"
        "NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL\n"
        "gpt-oss:20b    17052f91a42e    16 GB    100% GPU     16384      59 minutes from now\n"
        "System-wide memory free percentage: 41%\n"
        "2026-11-02 23:23:30\n"
        "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
        "System-wide memory free percentage: 63%\n"
    )
    m = memory_summary(log)
    assert m["samples"] == 2
    assert m["min_free_percent"] == 41 and m["max_free_percent"] == 63
    assert m["ollama_ps"] == ["gpt-oss:20b 16 GB 100% GPU 16384"]


def test_summary_throughput_with_and_without_load() -> None:
    records = [
        record(
            "train/a/t#0", "pick_and_place_simple", won=True, started=0.0, finished=100.0, steps=3
        ),
        record(
            "train/b/t#0",
            "pick_and_place_simple",
            won=False,
            started=0.0,
            finished=60.0,
            stop="max_steps",
            steps=3,
            prompts=(100, 7000, 300),
            truncated=1,
        ),
        record("train/c/t#0", "look_at_obj_in_light", won=True, started=60.0, finished=120.0),
        record(
            "train/d/t#0",
            "look_at_obj_in_light",
            won=True,
            started=100.0,
            finished=160.0,
            take_seen=False,
        ),
    ]
    s = summarize(records, load_s=[20.0], expert_steps={"train/a/t": 1, "train/c/t": 10})
    assert s["episodes"] == 4 and s["won"] == 3
    assert s["won_by_type"] == {"look_at_obj_in_light": 2, "pick_and_place_simple": 1}
    t = s["throughput"]
    assert t["wall_clock_s"] == 160.0
    assert t["s_bar_including_load"] == 40.0
    assert t["s_bar_excluding_load"] == 35.0  # (160 - 20) / 4
    assert t["load_s"] == 20.0
    assert t["effective_concurrency"] == pytest.approx(280 / 160)
    assert s["prompt_tokens"]["max"] == 7000
    assert s["prompt_tokens_over_limit"] == {"limit": 6144, "exceeded": True, "calls": 1}
    assert s["stop_reasons"] == {"max_steps": 1, "won": 3}
    assert s["truncated_calls"] == 1 and s["format_errors"] == 4
    assert s["principle_9"] == "not_applicable"
    assert s["per_episode"]["llm_calls"]["max"] == 3
    # E-a: d는 shelf 1 내용을 보지 않고 꺼냈다. E-c: a는 3걸음 > 2 x 1, c는 2 <= 20
    assert s["e_a"] == {"won": 3, "holds": 2}
    assert s["e_b"] == {"admissible_steps": 8, "steps": 10}  # b의 실패한 take 2개
    assert s["e_c"] == {"defined": 2, "holds": 1}


def test_summary_without_a_load_line_reports_none() -> None:
    s = summarize(
        [record("train/a/t#0", "x", won=True, started=0.0, finished=10.0)],
        load_s=[],
        expert_steps={},
    )
    assert s["throughput"]["load_s"] is None
    assert s["throughput"]["s_bar_excluding_load"] is None


def test_one_load_reported_by_two_waiting_requests_counts_once() -> None:
    # P3 실제 로그: 동시 요청 2개가 같은 적재를 각각 "started in"으로 적는다
    one_load = (
        'msg="starting llama-server" cmd="..."\n'
        'msg="llama-server started in 13.39 seconds"\n'
        'msg="loaded runners" count=1\n'
        'msg="llama-server started in 13.41 seconds"\n'
    )
    assert load_seconds(one_load) == [13.41]
    two_loads = (
        one_load + 'msg="starting llama-server"\nmsg="llama-server started in 9.5 seconds"\n'
    )
    assert load_seconds(two_loads) == [13.41, 9.5]
