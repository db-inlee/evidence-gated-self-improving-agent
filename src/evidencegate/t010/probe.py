"""Ticket 010 §2 Dev-A 손잡이 탐침(사전 등록 전). 보고서 2026-11-10-03의 변형·기준을 고정한다.

    caffeinate -dims uv run python -m evidencegate.t010.probe preflight
    PYTHONUNBUFFERED=1 caffeinate -dims uv run python -m evidencegate.t010.probe run \\
        [--stop-at HH:MM] [--stop-file PATH] [--max-blocks N]
    uv run python -m evidencegate.t010.probe status
    uv run python -m evidencegate.t010.probe report

- 블록 = 변형(`VARIANTS`, P0~P6 순서 고정). 블록마다 S0 스킬 x Dev-A 60(반복 0, 009와 같은 케이스·
  llm_seed)이고, 변형마다 정책(`config_id`)이 다르다. 기록은 split `t010_probe`
  (`runs/t010_probe/…`)에 둔다.
- 비교 대상은 009 Dev-A S0 기록(`S0_RECORDS`)이다. 읽기 전용(`repair=False`)으로 열기만 하고,
  sha256이 `S0_RECORDS_SHA256`과 다르면 실행·보고를 거부한다.
- 실행 전 점검(`probe_preflight`): 009의 실행 조건 점검(engine, config_id, machine, parallel,
  ollama_version, power, disk)에 S0 기록, 변형 설정, 기록 경로 점검을 더한다. 그다음 v1과 같은
  병렬 탐침을 한다. 블록 점검(전원·Ollama 버전), `--stop-at`·정지 파일, Ctrl-C 기록은 009b와 같다.
- 형식 오류(티켓 §11의 2, 후보 블록 규칙): 형식 오류율로 멈추지 않고 블록별 형식 오류 에피소드
  비율을 기록한다. 다만 연속 3개 블록에서 그 비율이 1/2을 넘으면 멈춘다(기반 이상 감지).
  서버 오류 규칙(3회 재시도, 연속 3개)은 v1 그대로다.
- P0(max_tool_calls 3)는 동작 확인용이다. P0 블록이 끝나면 처분이 S0와 다른 에피소드 수를 세고,
  `P0_MIN_EPISODES` 미만이면 다음 변형을 돌리지 않고 멈춘다(탐침 판정 없음).
- 실행·진행 확인은 CD·ES를 출력하지 않는다. `report`만 칸별 CD·ES 차이를 낸다(Dev-A, 기술용).
- `report`는 모든 변형이 60개씩 끝났을 때만 판정(`judge`)하고 `results/ticket-010/probe.json`을
  쓴다. 같은 기록이면 같은 바이트이고, 이미 있는 파일과 다르면 덮어쓰지 않는다.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate import bench_runner
from evidencegate.analysis import REPO_ROOT
from evidencegate.bench_runner import BenchStopped, SessionLog, StopRules, block_store, run_blocks
from evidencegate.eval_constants import (
    CONFIG_ID,
    POLICY_ID,
    RUN_WORKERS,
    S0_SKILL_ID,
    TOOL_SPEC_ID,
)
from evidencegate.harness import HarnessText
from evidencegate.ids import config_id, content_id
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
from evidencegate.run_plan import S0_PATH, Block, plan_sha256
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009.bench import block_check, next_stop_at, server_version
from evidencegate.t009.preflight import check_ollama_version, check_power_now
from evidencegate.t009.splits import split_cases

# ---------------------------------------------------------------------------
# 고정 값(보고서 2026-11-10-03 6·7절, 사용자 결정 2026-11-10)
# ---------------------------------------------------------------------------

MODEL = "gpt-oss:20b"
PROBE_SPLIT = "t010_probe"
SOURCE_SPLIT = "t009_dev_a"
REPEAT = 0
S0_RECORDS = (
    Path(SOURCE_SPLIT)
    / f"{POLICY_ID.replace(':', '__')}__{CONFIG_ID}"
    / TOOL_SPEC_ID
    / f"{S0_SKILL_ID}.jsonl"
)  # runs 폴더 기준
S0_RECORDS_SHA256 = "6d68923dd27ec65ffddb96be03f6a33b8331f569138f75b3c4565b586b3d1dd3"
LOG_DIR = "ticket010/probe"
SESSION_LOG = f"{LOG_DIR}/sessions.jsonl"
RESULT_PATH = Path("results/ticket-010/probe.json")
TRAFFIC_TOOL = "get_traffic_metrics"

EPISODES = 60
MOVED_MIN_EPISODES = 8  # k: 잡음(같은 seed 처분 불일치 25/600)만으로 8개 이상일 확률 0.34%
MOVED_MIN_VARIANTS = 3  # m
MOVED_MIN_FIELDS = 2
P0_MIN_EPISODES = 50
FORMAT_ANOMALY_BLOCKS = 3
FORMAT_ANOMALY_RATE = Fraction(1, 2)
S0_MAX_TOOL_CALLS = 8  # OllamaSettings 기본값(config 7c66ee5bf13c)
P0_MAX_TOOL_CALLS = 3

_KOREAN_COUNT = {2: "두", 3: "세"}
DECISION_RULE = (
    f"변형 하나가 움직였다 = Dev-A {EPISODES}개 중 처분이 S0와 다른 에피소드가 "
    f"{MOVED_MIN_EPISODES}개 이상. "
    f"진행 = P1~P6 중 {MOVED_MIN_VARIANTS}개 이상이 움직였고, 움직인 변형이 필드 "
    f"{_KOREAN_COUNT[MOVED_MIN_FIELDS]} 종류 이상"
    "(도구 순서·도구 설명·표시 방식)에 걸쳐 있음. 그렇지 않으면 본 실행 전에 방향을 다시 정한다. "
    f"P0(max_tool_calls {P0_MAX_TOOL_CALLS})는 동작 확인용이고 판정 집계에 넣지 않는다. "
    f"P0에서 처분이 S0와 다른 에피소드가 {P0_MIN_EPISODES}개 미만이면 하네스 적용을 의심하고 "
    "탐침 판정을 내지 않은 채 멈춘다."
)

# 후보 블록: 형식 오류율로 멈추지 않는다(비율은 1을 넘을 수 없다). 서버 오류 규칙은 v1 그대로.
CANDIDATE_RULES = StopRules(max_rate=Fraction(1))


@dataclass(frozen=True)
class Variant:
    name: str
    field: str  # control / tool_order / tool_descriptions / observation_renderer
    harness: HarnessText | None
    judged: bool  # 판정 집계(m)에 넣는가
    max_tool_calls: int = S0_MAX_TOOL_CALLS


VARIANTS: tuple[Variant, ...] = (
    Variant("P0", "control", None, judged=False, max_tool_calls=P0_MAX_TOOL_CALLS),
    Variant(
        "P1",
        "tool_order",
        HarnessText(
            tool_order=(
                "get_traffic_metrics",
                "get_service_metrics",
                "get_recent_deployments",
                "get_dependency_health",
            )
        ),
        judged=True,
    ),
    Variant(
        "P2",
        "tool_order",
        HarnessText(
            tool_order=(
                "get_service_metrics",
                "get_dependency_health",
                "get_recent_deployments",
                "get_traffic_metrics",
            )
        ),
        judged=True,
    ),
    Variant(
        "P3",
        "tool_descriptions",
        HarnessText(
            tool_descriptions=(
                (
                    "get_traffic_metrics",
                    "Get baseline and current request rate (requests per second) of the service. "
                    "Useful to check whether demand changed.",
                ),
            )
        ),
        judged=True,
    ),
    Variant(
        "P4",
        "tool_descriptions",
        HarnessText(
            tool_descriptions=(
                (
                    "get_dependency_health",
                    "Get the health status of each dependency of the service and the minute any "
                    "outage started (null if none). Useful to tell a problem in a dependency from "
                    "a problem in the service itself.",
                ),
            )
        ),
        judged=True,
    ),
    Variant(
        "P5", "observation_renderer", HarnessText(observation_renderer="key_value_lines"), True
    ),
    Variant("P6", "observation_renderer", HarnessText(observation_renderer="json_compact"), True),
)
_EXPECTED_CHANGE = {"control": {"stop.max_tool_calls"}}  # 그 밖의 필드는 {"harness"}


# ---------------------------------------------------------------------------
# 변형 설정과 블록
# ---------------------------------------------------------------------------


def variant_settings(variant: Variant | None) -> OllamaSettings:
    """None이면 S0(기본 하네스) 설정이다."""
    return OllamaSettings.for_model(
        MODEL,
        concurrency=RUN_WORKERS,
        server_num_parallel=RUN_WORKERS,
        max_tool_calls=variant.max_tool_calls if variant else S0_MAX_TOOL_CALLS,
        harness=variant.harness if variant else None,
    )


def variant_policy(
    variant: Variant | None, client: httpx.Client, base_config: Mapping[str, Any]
) -> OllamaPolicy:
    """S0 정책 설정(서버에서 읽은 엔진·장비 값)에서 변형 정책을 만든다. 서버를 부르지 않는다."""
    return OllamaPolicy(
        variant_settings(variant),
        client,
        model_digest=base_config["model_digest"],
        quantization=base_config["quantization"],
        ollama_version=base_config["ollama_version"],
        machine=Machine(**base_config["machine"]),
    )


def offline_policy(variant: Variant | None, base_config: Mapping[str, Any]) -> OllamaPolicy:
    """설정·기록 경로 계산용(LLM을 부르지 않음)."""
    with httpx.Client() as unused:
        return variant_policy(variant, unused, base_config)


def changed_paths(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = "") -> set[str]:
    """두 설정에서 값이 다른 경로(점으로 이음). 한쪽에만 있는 키는 그 키 경로."""
    out: set[str] = set()
    for key in set(a) | set(b):
        path = f"{prefix}{key}"
        if key not in a or key not in b:
            out.add(path)
        elif isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
            out |= changed_paths(a[key], b[key], f"{path}.")
        elif a[key] != b[key]:
            out.add(path)
    return out


def probe_blocks() -> tuple[Block, ...]:
    cases = tuple(split_cases(SOURCE_SPLIT))
    return tuple(
        Block(
            i,
            v.name,
            "probe" if v.judged else "control",
            S0_SKILL_ID,
            S0_PATH,
            PROBE_SPLIT,
            REPEAT,
            cases,
        )
        for i, v in enumerate(VARIANTS)
    )


# ---------------------------------------------------------------------------
# S0 기록, 짝 맞춤, 요약, 판정
# ---------------------------------------------------------------------------

PairKey = tuple[str, str, int, int]  # (family, level, incident_seed, repeat)


def pair_key(r: EpisodeRecord) -> PairKey:
    return (r.key.family, r.key.level, r.key.incident_seed, r.key.repeat)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path) -> list[EpisodeRecord]:
    return EpisodeStore(path, repair=False).records() if path.exists() else []


def load_s0(runs_dir: Path, expected_sha256: str = S0_RECORDS_SHA256) -> list[EpisodeRecord]:
    """009 Dev-A S0 기록(읽기 전용). 파일이 없거나 sha256이 다르면 ValueError."""
    path = runs_dir / S0_RECORDS
    if not path.exists():
        raise ValueError(f"missing the 009 Dev-A S0 records {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path} sha256 {actual} does not match {expected_sha256}")
    return read_records(path)


def check_s0_records(
    runs_dir: Path, base_config: Mapping[str, Any], expected_sha256: str = S0_RECORDS_SHA256
) -> Check:
    try:
        records = load_s0(runs_dir, expected_sha256)
    except ValueError as error:
        return Check("s0_dev_a", False, str(error))
    expected = {(f, lv, s, REPEAT) for f, lv, s in split_cases(SOURCE_SPLIT)}
    keys = [pair_key(r) for r in records]
    if len(records) != EPISODES or set(keys) != expected or len(set(keys)) != len(keys):
        return Check(
            "s0_dev_a", False, f"{len(records)} records do not match the {EPISODES} Dev-A cases"
        )
    if any(r.policy_config != dict(base_config) for r in records):
        return Check("s0_dev_a", False, "S0 records carry another policy_config than the S0 now")
    return Check("s0_dev_a", True, f"{EPISODES} S0 Dev-A records, sha256 {expected_sha256}")


def pair(
    s0: Mapping[PairKey, EpisodeRecord], records: Sequence[EpisodeRecord]
) -> list[tuple[EpisodeRecord, EpisodeRecord]]:
    """(S0, 변형) 짝. 빠짐·남음·중복·llm_seed 불일치는 ValueError."""
    seen: dict[PairKey, EpisodeRecord] = {}
    for r in records:
        key = pair_key(r)
        if key not in s0:
            raise ValueError(f"{key} is not in the S0 Dev-A records")
        if key in seen:
            raise ValueError(f"{key} appears twice")
        if r.llm_seed != s0[key].llm_seed:
            raise ValueError(f"{key}: llm_seed {r.llm_seed} != S0 {s0[key].llm_seed}")
        seen[key] = r
    missing = sorted(set(s0) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} S0 cases missing, first {missing[0]}")
    return [(s0[k], seen[k]) for k in sorted(seen)]


def tools_called(r: EpisodeRecord) -> frozenset[str]:
    """형식 오류가 아닌 턴에서 부른 도구(조치 도구 포함)."""
    return frozenset(t.tool for t in r.trace.turns if not t.format_error)


def describe(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """기술용(판정에 쓰지 않음): 트래픽 호출, 형식 오류, 정지 이유, 도구 호출 수·순서."""
    sequences = Counter(" > ".join(t.tool for t in r.trace.turns) for r in records)
    return {
        "traffic_episodes": sum(TRAFFIC_TOOL in tools_called(r) for r in records),
        "format_error_episodes": sum(r.trace.format_errors > 0 for r in records),
        "stop_reasons": dict(sorted(Counter(str(r.trace.stop_reason) for r in records).items())),
        "mean_tool_calls": sum(len(r.trace.turns) for r in records) / len(records),
        "sequences": dict(sorted(sequences.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


@dataclass(frozen=True)
class VariantSummary:
    name: str
    field: str
    judged: bool
    episodes: int
    different_disposition: int  # 판정에 쓰는 유일한 수
    different_tool_set: int
    traffic_episodes: int
    format_error_episodes: int
    stop_reasons: dict[str, int]
    mean_tool_calls: float
    sequences: dict[str, int]
    cells: dict[str, dict[str, int]]  # "family/level" -> n, cd_diff, es_diff(변형 - S0, 개수)


def summarize(
    variant: Variant, pairs: Sequence[tuple[EpisodeRecord, EpisodeRecord]]
) -> VariantSummary:
    rows = [v for _, v in pairs]
    cells: dict[str, dict[str, int]] = {}
    for s, v in pairs:
        cell = cells.setdefault(
            f"{v.key.family}/{v.key.level}", {"n": 0, "cd_diff": 0, "es_diff": 0}
        )
        cell["n"] += 1
        cell["cd_diff"] += int(v.correct_disposition) - int(s.correct_disposition)
        cell["es_diff"] += int(v.evidence_safe) - int(s.evidence_safe)
    return VariantSummary(
        name=variant.name,
        field=variant.field,
        judged=variant.judged,
        episodes=len(rows),
        different_disposition=sum(s.action != v.action for s, v in pairs),
        different_tool_set=sum(tools_called(s) != tools_called(v) for s, v in pairs),
        cells=dict(sorted(cells.items())),
        **describe(rows),
    )


@dataclass(frozen=True)
class Decision:
    verdict: str  # proceed / reconsider / no_decision
    p0_different: int
    moved: tuple[str, ...]
    fields: tuple[str, ...]


def enough_fields(fields: Sequence[str]) -> bool:
    return len(set(fields)) >= MOVED_MIN_FIELDS


def judge(different: Mapping[str, int]) -> Decision:
    """변형 이름 -> 처분이 S0와 다른 에피소드 수에 고정 규칙(DECISION_RULE)을 적용한다."""
    p0 = different["P0"]
    if p0 < P0_MIN_EPISODES:
        return Decision("no_decision", p0, (), ())
    moved = [v for v in VARIANTS if v.judged and different[v.name] >= MOVED_MIN_EPISODES]
    fields = tuple(dict.fromkeys(v.field for v in moved))
    ok = len(moved) >= MOVED_MIN_VARIANTS and enough_fields(fields)
    return Decision("proceed" if ok else "reconsider", p0, tuple(v.name for v in moved), fields)


def format_anomaly(shares: Sequence[Fraction]) -> bool:
    """연속 `FORMAT_ANOMALY_BLOCKS`개 블록에서 형식 오류 에피소드 비율이 1/2을 넘는가."""
    run = 0
    for share in shares:
        run = run + 1 if share > FORMAT_ANOMALY_RATE else 0
        if run >= FORMAT_ANOMALY_BLOCKS:
            return True
    return False


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeRun:
    blocks_run: int
    stopped_by: str | None  # max_blocks / stop_file / stop_at, 끝까지 돌았으면 None


def run_probe(
    policies: Mapping[str, Policy],
    *,
    root: Path,
    runs_dir: Path,
    out: TextIO,
    s0: Mapping[PairKey, EpisodeRecord],
    max_blocks: int | None = None,
    check_power: Callable[[], None] | None = None,
    stop_at: datetime | None = None,
    stop_file: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    session_fields: Mapping[str, Any] | None = None,
) -> ProbeRun:
    """변형 순서대로 블록 하나씩 v1 `run_blocks`로 돈다. 멈춤 규칙에 걸리면 BenchStopped."""
    log = SessionLog(runs_dir, SESSION_LOG)
    variants = {
        v.name: {"field": v.field, "config_id": config_id(policies[v.name])} for v in VARIANTS
    }
    log.write("probe_start", variants=variants, **dict(session_fields or {}))
    shares: list[Fraction] = []
    blocks_run = 0
    try:
        for variant, block in zip(VARIANTS, probe_blocks(), strict=True):
            policy = policies[variant.name]
            path = block_store(runs_dir, policy, block).path
            records = read_records(path)
            if len(records) < len(block.cases):
                if max_blocks is not None and blocks_run >= max_blocks:
                    log.write("probe_end", reason="max_blocks", blocks_run=blocks_run)
                    print(f"max blocks reached ({max_blocks}); stopping", file=out, flush=True)
                    return ProbeRun(blocks_run, "max_blocks")
                print(f"variant {variant.name} ({variant.field})", file=out, flush=True)
                summary = run_blocks(
                    [block],
                    policy,
                    root=root,
                    runs_dir=runs_dir,
                    out=out,
                    rules=CANDIDATE_RULES,
                    session_log=SESSION_LOG,
                    check_power=check_power,
                    stop_at=stop_at,
                    stop_file=stop_file,
                    clock=clock,
                    log_interrupt=True,
                    session_fields={"variant": variant.name, **dict(session_fields or {})},
                )
                if summary.stopped_by is not None:
                    log.write("probe_end", reason=summary.stopped_by, blocks_run=blocks_run)
                    return ProbeRun(blocks_run, summary.stopped_by)
                blocks_run += summary.blocks_run
                records = read_records(path)
            if len(records) < len(block.cases):
                raise BenchStopped(
                    f"{variant.name}: {len(records)} of {len(block.cases)} episodes stored "
                    "(episodes failed); run the same command again",
                    None,
                )
            share = Fraction(sum(r.trace.format_errors > 0 for r in records), len(records))
            shares.append(share)
            log.write("block_format_errors", variant=variant.name, share=str(share))
            if format_anomaly(shares):
                raise BenchStopped(
                    f"format error episodes exceed {FORMAT_ANOMALY_RATE} in "
                    f"{FORMAT_ANOMALY_BLOCKS} consecutive blocks (last {variant.name}); "
                    "a person should look",
                    None,
                )
            if variant.name == "P0":
                different = summarize(variant, pair(s0, records)).different_disposition
                log.write("p0_check", different_disposition=different, minimum=P0_MIN_EPISODES)
                if different < P0_MIN_EPISODES:
                    raise BenchStopped(
                        f"P0 check failed: {different} of {len(records)} dispositions differ "
                        f"from S0 (need {P0_MIN_EPISODES}); suspect the harness wiring; "
                        "no probe decision",
                        None,
                    )
    except BenchStopped as stop:
        log.write("probe_end", reason=stop.reason, blocks_run=blocks_run)
        raise
    except KeyboardInterrupt:
        log.write("probe_end", reason="interrupted", blocks_run=blocks_run)
        raise
    log.write("probe_end", reason="plan_done", blocks_run=blocks_run)
    return ProbeRun(blocks_run, None)


# ---------------------------------------------------------------------------
# 실행 전 점검
# ---------------------------------------------------------------------------


def check_variants(base_config: Mapping[str, Any]) -> Check:
    base_id = content_id(dict(base_config))
    ids: dict[str, str] = {}
    for v in VARIANTS:
        config = offline_policy(v, base_config).config
        changed = changed_paths(base_config, config)
        expected = _EXPECTED_CHANGE.get(v.field, {"harness"})
        if changed != expected:
            return Check("variants", False, f"{v.name} changes {sorted(changed)}")
        ids[v.name] = content_id(config)
    if len(set(ids.values())) != len(ids) or base_id in ids.values():
        return Check(
            "variants", False, f"config_ids are not distinct from each other and S0: {ids}"
        )
    return Check("variants", True, ", ".join(f"{n} {i}" for n, i in ids.items()))


def check_probe_paths(runs_dir: Path, base_config: Mapping[str, Any]) -> Check:
    s0_path = (runs_dir / S0_RECORDS).resolve()
    probe_root = (runs_dir / PROBE_SPLIT).resolve()
    for v, block in zip(VARIANTS, probe_blocks(), strict=True):
        policy = offline_policy(v, base_config)
        path = block_store(runs_dir, policy, block).path.resolve()
        if path == s0_path or probe_root not in path.parents:
            return Check("probe_paths", False, f"{v.name} would write to {path}")
    return Check("probe_paths", True, f"records under {runs_dir / PROBE_SPLIT}")


def probe_preflight(
    *,
    root: Path,
    runs_dir: Path,
    base_config: Mapping[str, Any],
    machine: Machine,
    s0_sha256: str = S0_RECORDS_SHA256,
    power_run: Run = system_run,
    disk_usage: Callable[[Path], Any] | None = None,
) -> list[Check]:
    try:
        engine = check_engine(base_config, reference_policy_config(runs_dir))
    except ValueError as error:
        engine = Check("engine", False, f"cannot read the S0 Dev-A reference: {error}")
    disk_path = runs_dir if runs_dir.exists() else root
    return [
        engine,
        check_config_id(base_config),
        check_machine(machine),
        check_parallel(base_config),
        check_ollama_version(base_config),
        check_power_now(power_run),
        check_disk(disk_path, usage=disk_usage) if disk_usage else check_disk(disk_path),
        check_s0_records(runs_dir, base_config, s0_sha256),
        check_variants(base_config),
        check_probe_paths(runs_dir, base_config),
    ]


# ---------------------------------------------------------------------------
# 진행 확인과 보고
# ---------------------------------------------------------------------------


def _variant_records(runs_dir: Path, base_config: Mapping[str, Any]) -> dict[str, list[Any]]:
    out = {}
    for v, block in zip(VARIANTS, probe_blocks(), strict=True):
        policy = offline_policy(v, base_config)
        out[v.name] = read_records(block_store(runs_dir, policy, block).path)
    return out


def status_lines(records: Mapping[str, Sequence[EpisodeRecord]]) -> list[str]:
    """CD·ES 없이 변형별 진행과 형식 오류 에피소드 수."""
    lines = []
    for v in VARIANTS:
        rows = records[v.name]
        errors = sum(r.trace.format_errors > 0 for r in rows)
        lines.append(
            f"{v.name} {v.field}: {len(rows)}/{EPISODES} episodes, format-error episodes {errors}"
        )
    return lines


def build_result(
    s0_records: Sequence[EpisodeRecord],
    records: Mapping[str, Sequence[EpisodeRecord]],
    s0_sha256: str,
) -> dict[str, Any]:
    s0 = {pair_key(r): r for r in s0_records}
    base = s0_records[0].policy_config
    summaries = {v.name: summarize(v, pair(s0, records[v.name])) for v in VARIANTS}
    decision = judge({n: s.different_disposition for n, s in summaries.items()})
    variants = []
    for v in VARIANTS:
        config = records[v.name][0].policy_config
        variants.append(
            {
                **asdict(summaries[v.name]),
                "config_id": content_id(config),
                "harness": config.get("harness"),
                "max_tool_calls": config["stop"]["max_tool_calls"],
                "moved": v.name in decision.moved,
            }
        )
    return {
        "ticket": "010",
        "stage": "dev_a_knob_probe",
        "rule": {
            "text": DECISION_RULE,
            "episodes": EPISODES,
            "k": MOVED_MIN_EPISODES,
            "m": MOVED_MIN_VARIANTS,
            "min_fields": MOVED_MIN_FIELDS,
            "p0_min": P0_MIN_EPISODES,
        },
        "s0_records": str(S0_RECORDS),
        "s0_records_sha256": s0_sha256,
        "s0_config_id": content_id(base),
        "engine": {k: base[k] for k in ("model", "model_digest", "quantization", "ollama_version")}
        | {"options": base["options"], "think": base["think"], "seed_rule": base["seed_rule"]},
        "probe_plan_sha256": plan_sha256(probe_blocks()),
        "s0": describe(s0_records),
        "variants": variants,
        "decision": asdict(decision),
    }


def result_bytes(result: Mapping[str, Any]) -> bytes:
    return (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def report_lines(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "variant field                diff_disp moved tool_set traffic fmt_err mean_calls stops",
    ]
    for r in result["variants"]:
        moved = "yes" if r["moved"] else ("-" if r["judged"] else "n/a")
        stops = ", ".join(f"{k} {n}" for k, n in r["stop_reasons"].items())
        lines.append(
            f"{r['name']:<7} {r['field']:<20} {r['different_disposition']:>9} {moved:>5} "
            f"{r['different_tool_set']:>8} {r['traffic_episodes']:>7} "
            f"{r['format_error_episodes']:>7} {r['mean_tool_calls']:>10.2f} {stops}"
        )
    s0 = result["s0"]
    lines.append(
        f"S0 (009 Dev-A): traffic {s0['traffic_episodes']}, format errors "
        f"{s0['format_error_episodes']}, mean calls {s0['mean_tool_calls']:.2f}"
    )
    lines.append("cell differences (variant - S0, count of 10; descriptive): CD / ES")
    for r in result["variants"]:
        cells = "  ".join(f"{c} {d['cd_diff']:+d}/{d['es_diff']:+d}" for c, d in r["cells"].items())
        lines.append(f"  {r['name']}: {cells}")
    d = result["decision"]
    lines.append(
        f"decision: {d['verdict']} (P0 {d['p0_different']}; moved {', '.join(d['moved']) or '-'};"
        f" fields {', '.join(d['fields']) or '-'})"
    )
    lines.append(f"rule: {result['rule']['text']}")
    return lines


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    client: httpx.Client | None = None,
    machine: Machine | None = None,
    preflight: Callable[[Policy], list[Check]] | None = None,
    probe: Callable[[], Any] | None = None,
    power_check: Callable[[], None] | None = None,
    ollama_version: Callable[[], str] | None = None,
    now: Callable[[], datetime] | None = None,
    s0_sha256: str = S0_RECORDS_SHA256,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t010-probe")
    parser.add_argument("command", choices=("preflight", "run", "status", "report"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--stop-at", default=None, help="local time HH:MM; stop at a block boundary"
    )
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--ollama-url", default=bench_runner.DEFAULT_OLLAMA_URL)
    args = parser.parse_args(argv)
    runs_dir: Path = args.runs_dir

    if args.command in ("status", "report"):
        return _status_or_report(args.command, runs_dir, args.root, s0_sha256, out, err)

    clock = now or (lambda: datetime.now().astimezone())
    try:
        stop_at = next_stop_at(args.stop_at, clock()) if args.stop_at else None
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    http = client or httpx.Client(base_url=args.ollama_url)
    the_machine = machine or detect_machine()
    try:
        base = OllamaPolicy.connect(variant_settings(None), http, machine=the_machine)
    except OllamaError as error:
        print(f"error: {error}", file=err)
        return 1
    checks = (
        preflight(base)
        if preflight
        else probe_preflight(
            root=args.root,
            runs_dir=runs_dir,
            base_config=base.config,
            machine=the_machine,
            s0_sha256=s0_sha256,
        )
    )
    for c in checks:
        print(f"preflight {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}", file=out)
    log = SessionLog(runs_dir, SESSION_LOG)
    if not preflight_ok(checks):
        print("preflight failed; nothing was run", file=err)
        log.write("preflight_failed", checks=[c.__dict__ for c in checks])
        return 1
    if args.command == "preflight":
        return 0
    try:
        s0_records = load_s0(runs_dir, s0_sha256)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1

    result = (
        probe()
        if probe
        else probe_parallelism(http, variant_settings(None), max_ratio=PARALLEL_PROBE_MAX_RATIO)
    )
    print(
        f"parallel probe: ratio {result.ratio:.2f} (limit {result.max_ratio:.2f}) -> "
        f"{'ok' if result.parallel else 'FAILED'}",
        file=out,
    )
    log.write("probe", ratio=result.ratio, ratios=list(result.ratios), parallel=result.parallel)
    if not result.parallel:
        print(f"error: restart Ollama with OLLAMA_NUM_PARALLEL={RUN_WORKERS}", file=err)
        return 1

    version = ollama_version or server_version(args.ollama_url)
    stop_file = args.stop_file or runs_dir / LOG_DIR / "STOP"
    policies = {v.name: variant_policy(v, http, base.config) for v in VARIANTS}
    try:
        run = run_probe(
            policies,
            root=args.root,
            runs_dir=runs_dir,
            out=out,
            s0={pair_key(r): r for r in s0_records},
            max_blocks=args.max_blocks,
            check_power=block_check(power_check or check_power, version),
            stop_at=stop_at,
            stop_file=stop_file,
            session_fields={
                "ollama_version": version(),
                "s0_records_sha256": s0_sha256,
                "probe_plan_sha256": plan_sha256(probe_blocks()),
                "stop_at": stop_at.isoformat() if stop_at else None,
                "stop_file": str(stop_file),
            },
        )
    except BenchStopped as stop:
        print(f"stopped: {stop.reason}", file=err)
        return 2
    print(f"done: {run.blocks_run} variant blocks run; stopped by {run.stopped_by}", file=out)
    for line in status_lines(_variant_records(runs_dir, base.config)):
        print(line, file=out)
    return 0


def _status_or_report(
    command: str, runs_dir: Path, root: Path, s0_sha256: str, out: TextIO, err: TextIO
) -> int:
    try:
        s0_records = load_s0(runs_dir, s0_sha256)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    records = _variant_records(runs_dir, s0_records[0].policy_config)
    lines = status_lines(records)
    if command == "status":
        for line in lines:
            print(line, file=out)
        return 0
    if any(len(records[v.name]) < EPISODES for v in VARIANTS):
        for line in lines:
            print(line, file=out)
        print("incomplete: not every variant has 60 episodes; no decision", file=err)
        return 1
    try:
        result = build_result(s0_records, records, s0_sha256)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 1
    data = result_bytes(result)
    path = root / RESULT_PATH
    if path.exists() and path.read_bytes() != data:
        print(f"error: {path} exists and differs from this report; not overwritten", file=err)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    for line in report_lines(result):
        print(line, file=out)
    print(f"wrote {path} (sha256 {hashlib.sha256(data).hexdigest()})", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
