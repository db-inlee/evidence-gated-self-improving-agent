"""Ticket 012 §9의 3: 0단계(S0-prime 기준률과 같은 seed 흔들림).

기준은 `plans/ticket-012-criteria.json`이다.

    caffeinate -dims uv run python -m evidencegate.t012.stage0 preflight
    PYTHONUNBUFFERED=1 caffeinate -dims uv run python -m evidencegate.t012.stage0 run \\
        [--stop-at HH:MM] [--stop-file PATH] [--max-blocks N]
    uv run python -m evidencegate.t012.stage0 status
    uv run python -m evidencegate.t012.stage0 report

- 하네스는 기본값이다(config_id 7c66ee5bf13c). 바꾸는 것은 스킬뿐이다(S0-prime, 수정본 S0-prime-r1).
- 스킬마다 블록 셋: Dev-A 60(반복 0)을 같은 설정(동시 2)·같은 seed로 세 번 돈다. 같은 키는 한 저장
  파일에 두 번 쓸 수 없으므로, 회차마다 split을 따로 둔다(`RUN_SPLITS`, 모두 `t012_stage0` 접두사).
  009 Dev-A 기록은 읽지도 쓰지도 않는다.
- 순서: S0-prime 세 번 → 1회째로 기준률 판정 → 미달이면 허용된 수정 한 번(S0-prime-r1)으로
  세 번 → 다시 판정.
- 기준률, 같은 seed 불일치(1회째 대 재실행 둘 = 120쌍), k 산식(Clopper-Pearson 상한, 아래 한계,
  꼬리 확률, 상한)은 모두 기준 파일에서 읽는다. 기준 파일 sha256이 고정값과 다르면 거부한다.
- 시도마다 판정: pass(기준률 통과, k ≤ 상한) / revise(S0-prime 기준률 미달, 수정 한 번 남음) /
  close(수정본도 미달이거나 k가 상한을 넘음). 결과 파일의 `verdict`는 마지막 시도의 판정이다.
- 블록 경계마다 정지 파일·`--stop-at`·전원 점검(010 탐침과 같다). 형식 오류·서버 오류 규칙은 v1
  그대로다(기본 하네스 블록). 실행 출력에는 CD·ES를 내지 않는다. `report`만 기준률을 낸다.
- `report`는 필요한 블록이 모두 끝났을 때만 `results/ticket-012/stage0.json`을 쓴다. 같은 기록이면
  같은 바이트이고, 다른 파일이 이미 있으면 덮어쓰지 않는다.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate import bench_runner
from evidencegate.analysis import REPO_ROOT
from evidencegate.bench_runner import BenchStopped, SessionLog, block_store, run_blocks
from evidencegate.eval_constants import RUN_WORKERS
from evidencegate.exact import clopper_pearson
from evidencegate.harness import ACTION_TOOLS
from evidencegate.ids import content_id
from evidencegate.machine import Machine, detect_machine
from evidencegate.ollama_policy import (
    PARALLEL_PROBE_MAX_RATIO,
    OllamaError,
    OllamaPolicy,
    OllamaSettings,
    probe_parallelism,
)
from evidencegate.policy import Policy
from evidencegate.power import Run, check_power, system_run
from evidencegate.preflight import (
    Check,
    check_config_id,
    check_disk,
    check_engine,
    check_machine,
    check_parallel,
    preflight_ok,
    reference_policy_config,
)
from evidencegate.run_plan import Block
from evidencegate.runner import EpisodeRecord
from evidencegate.skills import load_skill
from evidencegate.store import EpisodeStore
from evidencegate.t009.bench import block_check, next_stop_at, server_version
from evidencegate.t009.preflight import check_ollama_version, check_power_now
from evidencegate.t009.splits import split_cases
from evidencegate.t010.probe import describe

MODEL = "gpt-oss:20b"
CRITERIA_PATH = Path("plans/ticket-012-criteria.json")
CRITERIA_SHA256 = "63a8f73ee8bac878c51007105036aa1bdeaed5d1f8a1678ce3893611efc5fe94"
CRITERIA_TAG = "t012-criteria"
SOURCE_SPLIT = "t009_dev_a"  # 케이스만 쓴다(009 기록은 열지 않음)
RUN_SPLITS = ("t012_stage0", "t012_stage0_rerun1", "t012_stage0_rerun2")
ATTEMPTS = ("S0_prime", "S0_prime_r1")  # 두 번째는 허용된 수정 한 번
REPEAT = 0
LOG_DIR = "ticket012/stage0"
SESSION_LOG = f"{LOG_DIR}/sessions.jsonl"
RESULT_PATH = Path("results/ticket-012/stage0.json")
UPPER_BOUNDS = {"clopper_pearson_two_sided_95": Fraction(1, 20)}

Git = Callable[[Sequence[str]], int]


def system_git(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), capture_output=True, check=False).returncode


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_criteria(root: Path, expected_sha256: str = CRITERIA_SHA256) -> dict[str, Any]:
    path = root / CRITERIA_PATH
    actual = _sha(path)
    sidecar = path.with_name(path.name + ".sha256").read_text().split()[0]
    if actual != expected_sha256 or sidecar != expected_sha256:
        raise ValueError(
            f"{CRITERIA_PATH} sha256 {actual} (sidecar {sidecar}) != {expected_sha256}"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def stage_blocks(criteria: Mapping[str, Any], key: str) -> tuple[Block, ...]:
    skill = criteria["skills"][key]
    cases = tuple(split_cases(SOURCE_SPLIT))
    return tuple(
        Block(i, key, "stage0", skill["skill_id"], skill["path"], split, REPEAT, cases)
        for i, split in enumerate(RUN_SPLITS)
    )


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------

Key = tuple[str, str, int]


def _key(r: EpisodeRecord) -> Key:
    return (r.key.family, r.key.level, r.key.incident_seed)


def _complete(records: Sequence[EpisodeRecord]) -> dict[Key, EpisodeRecord]:
    expected = set(split_cases(SOURCE_SPLIT))
    by_key = {_key(r): r for r in records}
    if set(by_key) != expected or len(records) != len(expected):
        raise ValueError(f"{len(records)} records do not match the {len(expected)} Dev-A cases")
    return by_key


def base_rate(records: Sequence[EpisodeRecord], criteria: Mapping[str, Any]) -> dict[str, Any]:
    rule = criteria["base_rate"]
    by_key = _complete(records)
    cells: Counter[str] = Counter()
    for (family, level, _), r in sorted(by_key.items()):
        cells[f"{family}/{level}"] += int(r.correct_disposition)
    names = sorted({f"{f}/{lv}" for f, lv, _ in by_key})
    if len(names) != rule["cells"]:
        raise ValueError(f"{len(names)} cells, criteria expect {rule['cells']}")
    in_range = sum(
        1 for n in names if rule["cd_successes_min"] <= cells[n] <= rule["cd_successes_max"]
    )
    return {
        "cells": {n: cells[n] for n in names},
        "in_range": in_range,
        "needed": rule["cells_needed"],
        "pass": in_range >= rule["cells_needed"],
    }


def disagreements(runs: Sequence[Sequence[EpisodeRecord]]) -> dict[str, int]:
    """1회째와 재실행 각각의 처분(action, 없음도 값 하나) 불일치. 짝이 맞지 않으면 거부."""
    first, *reruns = (_complete(r) for r in runs)
    configs = {content_id(r.policy_config) for run in (first, *reruns) for r in run.values()}
    if len(configs) != 1:
        raise ValueError(f"runs carry {len(configs)} different policy configs")
    differing = pairs = 0
    for run in reruns:
        for key, r in run.items():
            if r.llm_seed != first[key].llm_seed:
                raise ValueError(f"{key}: llm_seed {r.llm_seed} != {first[key].llm_seed}")
            pairs += 1
            differing += int(r.action != first[key].action)
    return {"pairs": pairs, "differing": differing}


def _tail(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def k_rule(counts: Mapping[str, int], criteria: Mapping[str, Any]) -> dict[str, Any]:
    rule = criteria["k_rule"]
    if rule["upper_bound"] not in UPPER_BOUNDS:
        raise ValueError(f"unknown upper_bound {rule['upper_bound']!r}")
    if counts["pairs"] != rule["pairs"]:
        raise ValueError(f"{counts['pairs']} pairs, criteria expect {rule['pairs']}")
    interval = clopper_pearson(
        counts["differing"], counts["pairs"], UPPER_BOUNDS[rule["upper_bound"]]
    )
    assert interval is not None
    upper = interval[1]
    used = max(upper, float(Fraction(rule["floor"])))
    tail = float(Fraction(rule["tail_below"]))
    n = rule["episodes"]
    k = next((k for k in range(n + 2) if _tail(k, n, used) < tail), n + 1)
    return {
        "p_upper": round(upper, 6),
        "p_used": round(used, 6),
        "k": k,
        "k_max": rule["k_max"],
        "over_cap": k > rule["k_max"],
    }


def describe_run(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """기술용(판정에 쓰지 않음)."""
    first_action = sum(
        1 for r in records if r.trace.turns and r.trace.turns[0].tool in ACTION_TOOLS
        and not r.trace.turns[0].format_error
    )  # fmt: skip
    counts = Counter(len(r.trace.turns) for r in records)
    return {
        **describe(records),
        "first_call_is_disposition": first_action,
        "tool_call_counts": {str(k): v for k, v in sorted(counts.items())},
    }


def attempt(key: str, runs: Sequence[Sequence[EpisodeRecord]], criteria: Mapping[str, Any]) -> Any:
    base = base_rate(runs[0], criteria)
    counts = disagreements(runs)
    k = k_rule(counts, criteria)
    revision = criteria["revision"]
    if not base["pass"]:
        outcome = "revise" if key != revision["skill"] else "close"
    else:
        outcome = "close" if k["over_cap"] else "pass"
    return {
        "skill": key,
        "skill_id": criteria["skills"][key]["skill_id"],
        "runs": list(RUN_SPLITS),
        "base_rate": base,
        "disagreement": counts,
        "k": k,
        "outcome": outcome,
        "descriptive": {split: describe_run(r) for split, r in zip(RUN_SPLITS, runs, strict=True)},
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRun:
    blocks_run: int
    stopped_by: str | None


def _records(runs_dir: Path, policy: Policy, block: Block) -> list[EpisodeRecord]:
    path = block_store(runs_dir, policy, block).path
    return EpisodeStore(path, repair=False).records() if path.exists() else []


def run_stage0(
    policy: Policy,
    *,
    root: Path,
    runs_dir: Path,
    out: TextIO,
    criteria: Mapping[str, Any],
    max_blocks: int | None = None,
    check_power: Callable[[], None] | None = None,
    stop_at: datetime | None = None,
    stop_file: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    session_fields: Mapping[str, Any] | None = None,
) -> StageRun:
    log = SessionLog(runs_dir, SESSION_LOG)
    log.write("stage0_start", criteria_sha256=CRITERIA_SHA256, **dict(session_fields or {}))
    blocks_run = 0
    try:
        for key in ATTEMPTS:
            blocks = stage_blocks(criteria, key)
            for block in blocks:
                if len(_records(runs_dir, policy, block)) >= len(block.cases):
                    continue
                if max_blocks is not None and blocks_run >= max_blocks:
                    log.write("stage0_end", reason="max_blocks", blocks_run=blocks_run)
                    print(f"max blocks reached ({max_blocks}); stopping", file=out, flush=True)
                    return StageRun(blocks_run, "max_blocks")
                print(f"{key} {block.split}", file=out, flush=True)
                summary = run_blocks(
                    [block], policy, root=root, runs_dir=runs_dir, out=out,
                    session_log=SESSION_LOG, check_power=check_power, stop_at=stop_at,
                    stop_file=stop_file, clock=clock, log_interrupt=True,
                    session_fields={"skill": key, "split": block.split},
                )  # fmt: skip
                if summary.stopped_by is not None:
                    log.write("stage0_end", reason=summary.stopped_by, blocks_run=blocks_run)
                    return StageRun(blocks_run, summary.stopped_by)
                blocks_run += summary.blocks_run
                if len(_records(runs_dir, policy, block)) < len(block.cases):
                    raise BenchStopped(
                        f"{key} {block.split}: episodes failed; run the same command again", None
                    )
            runs = [_records(runs_dir, policy, b) for b in blocks]
            outcome = attempt(key, runs, criteria)["outcome"]
            log.write("stage0_attempt", skill=key, outcome=outcome)
            if outcome != "revise":
                break
            print(f"{key}: base rate below the criteria; running the one revision", file=out)
    except BenchStopped as stop:
        log.write("stage0_end", reason=stop.reason, blocks_run=blocks_run)
        raise
    except KeyboardInterrupt:
        log.write("stage0_end", reason="interrupted", blocks_run=blocks_run)
        raise
    log.write("stage0_end", reason="done", blocks_run=blocks_run)
    return StageRun(blocks_run, None)


# ---------------------------------------------------------------------------
# 보고
# ---------------------------------------------------------------------------


class _Offline:
    """기록 경로 계산용(정책을 돌리지 않음)."""

    def __init__(self, policy_id: str, config: Mapping[str, Any]) -> None:
        self.policy_id = policy_id
        self.config = dict(config)

    def run(self, *args: Any) -> Any:  # pragma: no cover
        raise RuntimeError("report does not run episodes")


def build_result(
    runs_dir: Path, criteria: Mapping[str, Any], policy_id: str, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    """필요한 블록이 모두 끝났으면 결과, 아니면 None."""
    policy = _Offline(policy_id, config)
    attempts: list[dict[str, Any]] = []
    for key in ATTEMPTS:
        runs = [_records(runs_dir, policy, b) for b in stage_blocks(criteria, key)]
        if any(len(r) < len(split_cases(SOURCE_SPLIT)) for r in runs):
            return None
        attempts.append(attempt(key, runs, criteria))
        if attempts[-1]["outcome"] != "revise":
            break
    last = attempts[-1]
    return {
        "ticket": "012",
        "stage": "stage0",
        "criteria_sha256": CRITERIA_SHA256,
        "criteria_tag": CRITERIA_TAG,
        "policy_id": policy_id,
        "config_id": content_id(dict(config)),
        "engine": {k: config.get(k) for k in ("model", "model_digest", "ollama_version")},
        "attempts": attempts,
        "verdict": last["outcome"],
        "skill_for_stage1": last["skill"] if last["outcome"] == "pass" else None,
    }


def result_bytes(result: Mapping[str, Any]) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def report(
    *,
    runs_dir: Path,
    root: Path,
    criteria: Mapping[str, Any],
    policy_id: str,
    config: Mapping[str, Any],
    out: TextIO,
) -> int:
    result = build_result(runs_dir, criteria, policy_id, config)
    if result is None:
        print("incomplete: the stage-0 blocks are not all finished; no verdict", file=out)
        return 1
    data = result_bytes(result)
    path = root / RESULT_PATH
    if path.exists() and path.read_bytes() != data:
        print(f"error: {path} exists and differs; not overwritten", file=out)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    for a in result["attempts"]:
        b, k = a["base_rate"], a["k"]
        print(
            f"{a['skill']}: base rate {b['in_range']}/{len(b['cells'])} cells in range "
            f"(need {b['needed']}), same-seed differing {a['disagreement']['differing']}/"
            f"{a['disagreement']['pairs']}, k {k['k']} (cap {k['k_max']}) -> {a['outcome']}",
            file=out,
        )
    print(f"verdict: {result['verdict']}; stage 1 skill: {result['skill_for_stage1']}", file=out)
    print(f"wrote {path} (sha256 {hashlib.sha256(data).hexdigest()})", file=out)
    return 0


# ---------------------------------------------------------------------------
# 실행 전 점검
# ---------------------------------------------------------------------------


def stage0_preflight(
    *,
    root: Path,
    runs_dir: Path,
    base_config: Mapping[str, Any],
    machine: Machine,
    git: Git = system_git,
    power_run: Run = system_run,
    disk_usage: Callable[[Path], Any] | None = None,
) -> list[Check]:
    try:
        engine = check_engine(base_config, reference_policy_config(runs_dir))
    except ValueError as error:
        engine = Check("engine", False, f"cannot read the S0 Dev-A reference: {error}")
    try:
        criteria: dict[str, Any] | None = load_criteria(root)
        crit = Check("criteria", True, f"{CRITERIA_PATH} sha256 {CRITERIA_SHA256}")
    except (ValueError, OSError) as error:
        criteria, crit = None, Check("criteria", False, str(error))
    code = git(["git", "-C", str(root), "merge-base", "--is-ancestor", CRITERIA_TAG, "HEAD"])
    tag = Check("criteria_tag", code == 0, f"{CRITERIA_TAG} ancestor of HEAD: git exit {code}")
    if criteria is None:
        skills = Check("skills", False, "criteria not loaded")
    else:
        wrong = [
            k for k in ATTEMPTS
            if not (root / criteria["skills"][k]["path"]).exists()
            or load_skill(root / criteria["skills"][k]["path"]).skill_id
            != criteria["skills"][k]["skill_id"]
        ]  # fmt: skip
        skills = Check("skills", not wrong, f"skill files differ from the criteria: {wrong}"
                       if wrong else "S0-prime and S0-prime-r1 match the criteria")  # fmt: skip
    paths = Check(
        "stage0_paths",
        all(s.startswith("t012_stage0") and s != SOURCE_SPLIT for s in RUN_SPLITS),
        f"records under runs/{', runs/'.join(RUN_SPLITS)}",
    )
    disk_path = runs_dir if runs_dir.exists() else root
    return [
        engine,
        check_config_id(base_config),
        check_machine(machine),
        check_parallel(base_config),
        check_ollama_version(base_config),
        check_power_now(power_run),
        check_disk(disk_path, usage=disk_usage) if disk_usage else check_disk(disk_path),
        crit,
        tag,
        skills,
        paths,
    ]


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def _settings() -> OllamaSettings:
    return OllamaSettings.for_model(MODEL, concurrency=RUN_WORKERS, server_num_parallel=RUN_WORKERS)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: httpx.Client | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t012-stage0")
    parser.add_argument("command", choices=("preflight", "run", "status", "report"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--stop-at", default=None)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--ollama-url", default=bench_runner.DEFAULT_OLLAMA_URL)
    args = parser.parse_args(argv)
    try:
        criteria = load_criteria(args.root)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=err)
        return 1

    if args.command in ("status", "report"):
        try:
            config = reference_policy_config(args.runs_dir)
        except ValueError as error:
            print(f"error: {error}", file=err)
            return 1
        policy_id = f"ollama:{MODEL}"
        if args.command == "report":
            return report(runs_dir=args.runs_dir, root=args.root, criteria=criteria,
                          policy_id=policy_id, config=config, out=out)  # fmt: skip
        offline = _Offline(policy_id, config)
        for key in ATTEMPTS:
            for b in stage_blocks(criteria, key):
                print(f"{key} {b.split}: {len(_records(args.runs_dir, offline, b))}/60", file=out)
        return 0

    try:
        stop_at = next_stop_at(args.stop_at, datetime.now().astimezone()) if args.stop_at else None
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    http = client or httpx.Client(base_url=args.ollama_url)
    machine = detect_machine()
    try:
        policy = OllamaPolicy.connect(_settings(), http, machine=machine)
    except OllamaError as error:
        print(f"error: {error}", file=err)
        return 1
    checks = stage0_preflight(root=args.root, runs_dir=args.runs_dir, base_config=policy.config,
                              machine=machine)  # fmt: skip
    for c in checks:
        print(f"preflight {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}", file=out)
    if not preflight_ok(checks):
        print("preflight failed; nothing was run", file=err)
        SessionLog(args.runs_dir, SESSION_LOG).write(
            "preflight_failed", checks=[c.__dict__ for c in checks]
        )
        return 1
    if args.command == "preflight":
        return 0
    probe = probe_parallelism(http, _settings(), max_ratio=PARALLEL_PROBE_MAX_RATIO)
    print(f"parallel probe: ratio {probe.ratio:.2f} -> {'ok' if probe.parallel else 'FAILED'}",
          file=out)  # fmt: skip
    if not probe.parallel:
        print(f"error: restart Ollama with OLLAMA_NUM_PARALLEL={RUN_WORKERS}", file=err)
        return 1
    version = server_version(args.ollama_url)
    stop_file = args.stop_file or args.runs_dir / LOG_DIR / "STOP"
    try:
        result = run_stage0(
            policy, root=args.root, runs_dir=args.runs_dir, out=out, criteria=criteria,
            max_blocks=args.max_blocks, check_power=block_check(check_power, version),
            stop_at=stop_at, stop_file=stop_file,
            session_fields={"ollama_version": version(), "stop_file": str(stop_file),
                            "stop_at": stop_at.isoformat() if stop_at else None},
        )  # fmt: skip
    except BenchStopped as stop:
        print(f"stopped: {stop.reason}", file=err)
        return 2
    print(f"done: {result.blocks_run} blocks run; stopped by {result.stopped_by}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
