"""Ticket 012 §9의 4: 1단계 손잡이 탐침(S0-prime 기준).

    caffeinate -dims uv run python -m evidencegate.t012.stage1 preflight
    PYTHONUNBUFFERED=1 caffeinate -dims uv run python -m evidencegate.t012.stage1 run \\
        [--stop-at HH:MM] [--stop-file PATH] [--max-blocks N]
    uv run python -m evidencegate.t012.stage1 status
    uv run python -m evidencegate.t012.stage1 report

- 변형: P1~P6는 010 탐침과 같은 값(`t010.probe.VARIANTS`)이다. P0는 도구 호출 한도만 기준 파일
  값(1회)으로 바꾼 동작 확인용이고 판정 집계에 넣지 않는다.
- 블록 = 변형. S0-prime 스킬 x Dev-A 60(반복 0), split `t012_stage1`. 비교 기준은 0단계 1회째 기록
  (split `t012_stage0`, S0-prime)이고 열기만 한다.
- 판정 숫자는 코드에 쓰지 않는다. k는 `results/ticket-012/stage0.json`(sha256 고정, 판정 pass,
  다음 단계 스킬 S0_prime)에서, m·필드 수·P0 기준은 기준 파일에서 읽는다.
- 형식 오류는 010과 같다: 후보 블록은 기록만 하고, 연속 3개 블록에서 형식 오류 에피소드가 1/2을
  넘으면 멈춘다. P0 블록이 끝나면 처분 차이를 세어 기준 미만이면 판정 없이 멈춘다.
- 도구 설명 후보(P3·P4)는 012 검사(다른 도구 이름 금지, 순서 지시 낱말)를 받고, 결과 파일에 문장을
  그대로 싣는다. 실행 출력에는 CD·ES를 내지 않는다.
- 결과 `results/ticket-012/stage1.json`은 모든 변형이 끝났을 때 한 번만 쓴다(같은 기록이면 같은
  바이트, 다른 파일이 있으면 덮어쓰지 않음).
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate import bench_runner
from evidencegate.analysis import REPO_ROOT
from evidencegate.bench_runner import BenchStopped, SessionLog, StopRules, block_store, run_blocks
from evidencegate.eval_constants import TOOL_SPEC_ID
from evidencegate.harness import TOOL_NAMES
from evidencegate.ids import config_id, content_id
from evidencegate.machine import detect_machine
from evidencegate.ollama_policy import (
    PARALLEL_PROBE_MAX_RATIO,
    OllamaError,
    OllamaPolicy,
    probe_parallelism,
)
from evidencegate.policy import Policy
from evidencegate.power import check_power
from evidencegate.preflight import Check, preflight_ok, reference_policy_config
from evidencegate.run_plan import Block
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009.bench import block_check, next_stop_at, server_version
from evidencegate.t009.splits import split_cases
from evidencegate.t010.probe import (
    FORMAT_ANOMALY_BLOCKS,
    FORMAT_ANOMALY_RATE,
    VARIANTS,
    Decision,
    PairKey,
    Variant,
    changed_paths,
    describe,
    format_anomaly,
    offline_policy,
    pair,
    pair_key,
    summarize,
    variant_policy,
)
from evidencegate.t012.stage0 import (
    CRITERIA_SHA256,
    RUN_SPLITS,
    load_criteria,
    stage0_preflight,
)
from evidencegate.t012.stage0 import _settings as base_settings

STAGE0_PATH = Path("results/ticket-012/stage0.json")
STAGE0_SHA256 = "9bbe3ba4ccbe572065452865ac952b40fc25b196b3da854cb6ae5036c6f34739"
STAGE1_SKILL = "S0_prime"
BASELINE_SPLIT = RUN_SPLITS[0]  # 0단계 1회째
PROBE_SPLIT = "t012_stage1"
SOURCE_SPLIT = "t009_dev_a"  # 케이스만 쓴다
REPEAT = 0
LOG_DIR = "ticket012/stage1"
SESSION_LOG = f"{LOG_DIR}/sessions.jsonl"
RESULT_PATH = Path("results/ticket-012/stage1.json")
CANDIDATE_RULES = StopRules(max_rate=Fraction(1))  # 후보 블록: 형식 오류율로 멈추지 않음


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage0(root: Path, expected_sha256: str = STAGE0_SHA256) -> dict[str, Any]:
    path = root / STAGE0_PATH
    actual = _sha(path)
    if actual != expected_sha256:
        raise ValueError(f"{STAGE0_PATH} sha256 {actual} != {expected_sha256}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def rule(criteria: Mapping[str, Any], stage0: Mapping[str, Any]) -> dict[str, Any]:
    """판정 숫자: k는 0단계 결과에서, 나머지는 기준 파일에서."""
    if stage0["verdict"] != "pass":
        raise ValueError(f"stage 0 verdict is {stage0['verdict']!r}, not pass")
    if stage0["skill_for_stage1"] != STAGE1_SKILL:
        raise ValueError(f"stage 1 skill is {stage0['skill_for_stage1']!r}, not {STAGE1_SKILL}")
    attempt = next(a for a in stage0["attempts"] if a["skill"] == STAGE1_SKILL)
    probe, p0 = criteria["probe"], criteria["probe"]["p0"]
    return {
        "k": attempt["k"]["k"],
        "m": probe["m"],
        "min_fields": probe["min_fields"],
        "p0_min": p0["min_changed"],
        "p0_max_tool_calls": p0["max_tool_calls"],
        "episodes": p0["episodes"],
    }


def variants(criteria: Mapping[str, Any]) -> tuple[Variant, ...]:
    p0 = replace(VARIANTS[0], max_tool_calls=criteria["probe"]["p0"]["max_tool_calls"])
    return (p0, *VARIANTS[1:])


def probe_blocks(criteria: Mapping[str, Any]) -> tuple[Block, ...]:
    skill = criteria["skills"][STAGE1_SKILL]
    cases = tuple(split_cases(SOURCE_SPLIT))
    return tuple(
        Block(i, v.name, "probe" if v.judged else "control", skill["skill_id"], skill["path"],
              PROBE_SPLIT, REPEAT, cases)
        for i, v in enumerate(variants(criteria))
    )  # fmt: skip


def _order_pattern(criteria: Mapping[str, Any]) -> re.Pattern[str]:
    words = criteria["order_ban"]["words"]
    joined = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in words)
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


def description_problems(tool: str, text: str, criteria: Mapping[str, Any]) -> list[str]:
    """012 §1.4 자동 검사: 순서 지시 낱말, 다른 도구 이름. 문장 판단은 쓰지 않는다."""
    problems = [f"order word {m.group(0)!r}" for m in _order_pattern(criteria).finditer(text)]
    if criteria["order_ban"]["other_tool_names_banned"]:
        problems += [f"other tool {n}" for n in TOOL_NAMES if n != tool and n in text]
    return problems


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------


def judge(
    different: Mapping[str, int], rule: Mapping[str, Any], variant_list: Sequence[Variant]
) -> Decision:
    p0 = different["P0"]
    if p0 < rule["p0_min"]:
        return Decision("no_decision", p0, (), ())
    moved = [v for v in variant_list if v.judged and different[v.name] >= rule["k"]]
    fields = tuple(dict.fromkeys(v.field for v in moved))
    ok = len(moved) >= rule["m"] and len(set(fields)) >= rule["min_fields"]
    return Decision("proceed" if ok else "reconsider", p0, tuple(v.name for v in moved), fields)


def check_variants(
    base_config: Mapping[str, Any],
    criteria: Mapping[str, Any],
    variant_list: Sequence[Variant] | None = None,
) -> Check:
    try:
        ids = {}
        for v in variant_list if variant_list is not None else variants(criteria):
            config = offline_policy(v, base_config).config
            expected = {"stop.max_tool_calls"} if v.field == "control" else {"harness"}
            changed = changed_paths(base_config, config)
            if changed != expected:
                return Check("variants", False, f"{v.name} changes {sorted(changed)}")
            ids[v.name] = content_id(config)
    except KeyError as error:
        return Check("variants", False, f"cannot build the variant configs: missing {error}")
    distinct = (
        len(set(ids.values())) == len(ids) and content_id(dict(base_config)) not in ids.values()
    )
    detail = ", ".join(f"{n} {i}" for n, i in ids.items())
    return Check("variants", distinct, detail)


# ---------------------------------------------------------------------------
# 기준 기록(0단계 1회째)
# ---------------------------------------------------------------------------


def baseline_path(
    runs_dir: Path, base_config: Mapping[str, Any], policy_id: str, criteria: Mapping[str, Any]
) -> Path:
    return EpisodeStore.for_run(
        runs_dir,
        suite=BASELINE_SPLIT,
        policy_id=policy_id,
        config_id=content_id(dict(base_config)),
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=criteria["skills"][STAGE1_SKILL]["skill_id"],
    ).path


def load_baseline(
    runs_dir: Path, base_config: Mapping[str, Any], policy_id: str, criteria: Mapping[str, Any]
) -> dict[PairKey, EpisodeRecord]:
    path = baseline_path(runs_dir, base_config, policy_id, criteria)
    records = EpisodeStore(path, repair=False).records() if path.exists() else []
    expected = {(f, lv, s, REPEAT) for f, lv, s in split_cases(SOURCE_SPLIT)}
    by_key = {pair_key(r): r for r in records}
    if set(by_key) != expected or len(records) != len(expected):
        raise ValueError(f"{len(records)} baseline records at {path}, expected {len(expected)}")
    return by_key


def check_baseline(
    runs_dir: Path, base_config: Mapping[str, Any], policy_id: str, criteria: Mapping[str, Any]
) -> Check:
    try:
        records = load_baseline(runs_dir, base_config, policy_id, criteria)
    except ValueError as error:
        return Check("baseline", False, str(error))
    return Check("baseline", True, f"{len(records)} S0-prime records of {BASELINE_SPLIT}")


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


def run_stage1(
    policies: Mapping[str, Policy],
    *,
    root: Path,
    runs_dir: Path,
    out: TextIO,
    baseline: Mapping[PairKey, EpisodeRecord],
    criteria: Mapping[str, Any],
    rule: Mapping[str, Any],
    max_blocks: int | None = None,
    check_power: Callable[[], None] | None = None,
    stop_at: datetime | None = None,
    stop_file: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    session_fields: Mapping[str, Any] | None = None,
) -> StageRun:
    log = SessionLog(runs_dir, SESSION_LOG)
    variant_list = variants(criteria)
    log.write(
        "stage1_start",
        variants={v.name: config_id(policies[v.name]) for v in variant_list},
        rule=dict(rule),
        **dict(session_fields or {}),
    )
    shares: list[Fraction] = []
    blocks_run = 0
    try:
        for variant, block in zip(variant_list, probe_blocks(criteria), strict=True):
            policy = policies[variant.name]
            records = _records(runs_dir, policy, block)
            if len(records) < len(block.cases):
                if max_blocks is not None and blocks_run >= max_blocks:
                    log.write("stage1_end", reason="max_blocks", blocks_run=blocks_run)
                    print(f"max blocks reached ({max_blocks}); stopping", file=out, flush=True)
                    return StageRun(blocks_run, "max_blocks")
                print(f"variant {variant.name} ({variant.field})", file=out, flush=True)
                summary = run_blocks(
                    [block], policy, root=root, runs_dir=runs_dir, out=out,
                    rules=CANDIDATE_RULES, session_log=SESSION_LOG, check_power=check_power,
                    stop_at=stop_at, stop_file=stop_file, clock=clock, log_interrupt=True,
                    session_fields={"variant": variant.name},
                )  # fmt: skip
                if summary.stopped_by is not None:
                    log.write("stage1_end", reason=summary.stopped_by, blocks_run=blocks_run)
                    return StageRun(blocks_run, summary.stopped_by)
                blocks_run += summary.blocks_run
                records = _records(runs_dir, policy, block)
            if len(records) < len(block.cases):
                raise BenchStopped(
                    f"{variant.name}: {len(records)} of {len(block.cases)} episodes stored; "
                    "run the same command again",
                    None,
                )
            share = Fraction(sum(r.trace.format_errors > 0 for r in records), len(records))
            shares.append(share)
            log.write("block_format_errors", variant=variant.name, share=str(share))
            if format_anomaly(shares):
                raise BenchStopped(
                    f"format error episodes exceed {FORMAT_ANOMALY_RATE} in "
                    f"{FORMAT_ANOMALY_BLOCKS} consecutive blocks (last {variant.name})",
                    None,
                )
            if not variant.judged:
                changed = summarize(variant, pair(baseline, records)).different_disposition
                log.write("p0_check", different_disposition=changed, minimum=rule["p0_min"])
                if changed < rule["p0_min"]:
                    raise BenchStopped(
                        f"P0 check failed: {changed} of {len(records)} dispositions differ "
                        f"(need {rule['p0_min']}); no probe decision",
                        None,
                    )
    except BenchStopped as stop:
        log.write("stage1_end", reason=stop.reason, blocks_run=blocks_run)
        raise
    except KeyboardInterrupt:
        log.write("stage1_end", reason="interrupted", blocks_run=blocks_run)
        raise
    log.write("stage1_end", reason="plan_done", blocks_run=blocks_run)
    return StageRun(blocks_run, None)


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------


def build_result(
    baseline: Mapping[PairKey, EpisodeRecord],
    records: Mapping[str, Sequence[EpisodeRecord]],
    criteria: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    variant_list = variants(criteria)
    missing = [v.name for v in variant_list if len(records.get(v.name, ())) < rule["episodes"]]
    if missing:
        raise ValueError(f"incomplete: variants {missing} are not finished")
    summaries = {v.name: summarize(v, pair(baseline, records[v.name])) for v in variant_list}
    decision = judge({n: s.different_disposition for n, s in summaries.items()}, rule, variant_list)
    rows = []
    for v in variant_list:
        config = records[v.name][0].policy_config
        rows.append(
            {
                **asdict(summaries[v.name]),
                "config_id": content_id(config),
                "harness": config.get("harness"),
                "max_tool_calls": config.get("stop", {}).get("max_tool_calls"),
                "moved": v.name in decision.moved,
            }
        )
    candidates = [
        {"variant": v.name, "tool": tool, "text": text,
         "problems": description_problems(tool, text, criteria)}
        for v in variant_list if v.harness is not None
        for tool, text in v.harness.tool_descriptions
    ]  # fmt: skip
    return {
        "ticket": "012",
        "stage": "stage1",
        "criteria_sha256": CRITERIA_SHA256,
        "stage0_sha256": STAGE0_SHA256,
        "rule": dict(rule),
        "baseline": {"split": BASELINE_SPLIT, **describe(list(baseline.values()))},
        "variants": rows,
        "tool_description_candidates": candidates,
        "decision": asdict(decision),
    }


def result_bytes(result: Mapping[str, Any]) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_result(root: Path, result: Mapping[str, Any]) -> bool:
    data = result_bytes(result)
    path = root / RESULT_PATH
    if path.exists() and path.read_bytes() != data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


# ---------------------------------------------------------------------------
# 실행 전 점검
# ---------------------------------------------------------------------------


def stage1_extra_checks(
    *,
    root: Path,
    runs_dir: Path,
    base_config: Mapping[str, Any],
    policy_id: str,
    criteria: Mapping[str, Any],
) -> list[Check]:
    try:
        stage0 = load_stage0(root)
        r = rule(criteria, stage0)
        result = Check("stage0_result", True, f"stage0 sha256 {STAGE0_SHA256}, pass, k {r['k']}")
    except (ValueError, OSError) as error:
        result = Check("stage0_result", False, str(error))
    paths = Check(
        "stage1_paths",
        PROBE_SPLIT not in (*RUN_SPLITS, SOURCE_SPLIT),
        f"records under runs/{PROBE_SPLIT}; baseline runs/{BASELINE_SPLIT} is only read",
    )
    return [
        result,
        check_variants(base_config, criteria),
        check_baseline(runs_dir, base_config, policy_id, criteria),
        paths,
    ]


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def _variant_records(
    runs_dir: Path, base_config: Mapping[str, Any], criteria: Mapping[str, Any]
) -> dict[str, list[EpisodeRecord]]:
    out = {}
    for v, block in zip(variants(criteria), probe_blocks(criteria), strict=True):
        out[v.name] = _records(runs_dir, offline_policy(v, base_config), block)
    return out


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t012-stage1")
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
        the_rule = rule(criteria, load_stage0(args.root))
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=err)
        return 1
    policy_id = "ollama:gpt-oss:20b"

    if args.command in ("status", "report"):
        try:
            config = reference_policy_config(args.runs_dir)
        except ValueError as error:
            print(f"error: {error}", file=err)
            return 1
        records = _variant_records(args.runs_dir, config, criteria)
        if args.command == "status":
            for v in variants(criteria):
                rows = records[v.name]
                errors = sum(r.trace.format_errors > 0 for r in rows)
                print(
                    f"{v.name} {v.field}: {len(rows)}/60, format-error episodes {errors}", file=out
                )
            return 0
        try:
            base = load_baseline(args.runs_dir, config, policy_id, criteria)
            result = build_result(base, records, criteria, the_rule)
        except ValueError as error:
            print(f"error: {error}", file=err)
            return 1
        if not write_result(args.root, result):
            print(f"error: {args.root / RESULT_PATH} exists and differs", file=err)
            return 1
        for row in result["variants"]:
            print(f"{row['name']} {row['field']}: differing {row['different_disposition']}/60, "
                  f"moved {row['moved']}, traffic {row['traffic_episodes']}", file=out)  # fmt: skip
        d = result["decision"]
        print(f"decision: {d['verdict']} (k {the_rule['k']}; moved {list(d['moved'])})", file=out)
        return 0

    try:
        stop_at = next_stop_at(args.stop_at, datetime.now().astimezone()) if args.stop_at else None
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    http = httpx.Client(base_url=args.ollama_url)
    machine = detect_machine()
    try:
        base_policy = OllamaPolicy.connect(base_settings(), http, machine=machine)
    except OllamaError as error:
        print(f"error: {error}", file=err)
        return 1
    checks = stage0_preflight(root=args.root, runs_dir=args.runs_dir,
                              base_config=base_policy.config, machine=machine)  # fmt: skip
    checks += stage1_extra_checks(root=args.root, runs_dir=args.runs_dir,
                                  base_config=base_policy.config, policy_id=policy_id,
                                  criteria=criteria)  # fmt: skip
    for c in checks:
        print(f"preflight {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}", file=out)
    if not preflight_ok(checks):
        print("preflight failed; nothing was run", file=err)
        return 1
    if args.command == "preflight":
        return 0
    probe = probe_parallelism(http, base_settings(), max_ratio=PARALLEL_PROBE_MAX_RATIO)
    print(f"parallel probe: ratio {probe.ratio:.2f} -> {'ok' if probe.parallel else 'FAILED'}",
          file=out)  # fmt: skip
    if not probe.parallel:
        return 1
    version = server_version(args.ollama_url)
    stop_file = args.stop_file or args.runs_dir / LOG_DIR / "STOP"
    policies = {v.name: variant_policy(v, http, base_policy.config) for v in variants(criteria)}
    baseline = load_baseline(args.runs_dir, base_policy.config, policy_id, criteria)
    try:
        run_result = run_stage1(
            policies, root=args.root, runs_dir=args.runs_dir, out=out, baseline=baseline,
            criteria=criteria, rule=the_rule, max_blocks=args.max_blocks,
            check_power=block_check(check_power, version), stop_at=stop_at, stop_file=stop_file,
            session_fields={"ollama_version": version(), "stop_file": str(stop_file)},
        )  # fmt: skip
    except BenchStopped as stop:
        print(f"stopped: {stop.reason}", file=err)
        return 2
    print(
        f"done: {run_result.blocks_run} variant blocks run; stopped by {run_result.stopped_by}",
        file=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
