"""proposer 진입점 (Ticket 004 §4). 실측과 풀 생성이 같은 명령을 쓴다.

    uv run python -m evidencegate.propose_cli models
    uv run python -m evidencegate.propose_cli propose --model <m> \\
        --input-usd-per-mtok <in> --output-usd-per-mtok <out> --out-dir <dir> [--calls N]

`propose`는 호출마다 `<out-dir>/call-NNN.json`을 쓴다. 폴더에 결과가 있으면 다음 call_index부터
이어서 부른다. 한 폴더는 한 proposer_config만 담는다(다르면 거부). 원장(기본
`runs/proposer/ledger.jsonl`)은 실측과 풀 생성이 함께 써서 누적 비용이 이어진다. 중단(예산, 할당량,
인증, 서버 오류, 형식 오류율)은 종료 코드 2다. API 키는 환경 변수 OPENAI_API_KEY에서만 읽는다.

풀(티켓 §5): 후보마다 §5 유효성 검사를 돌리고, 유효하면 이 폴더의 앞선 풀 후보와 중복인지
본다. proposer 중복이면 풀에서 빼고(`duplicate_of`), 통제 후보와 중복이면 남기고 표시한다
(`control_duplicates`). 이전 hypothesis 목록에는 풀에 남는 후보(`in_pool`)만 넣는다.
`--max-total-calls`는 폴더 전체 호출 수 상한, `--stop-when-pool`은 풀 크기가 목표에 닿으면 더
부르지 않는다. 실행 끝에 요약 줄을 출력한다.

선택 인자(Ticket 009b, 기본값이면 v1과 같다):
- `--previous valid`: 이전 hypothesis 목록에 유효 후보 전부(중복으로 빠진 후보 포함)를 넣는다.
  정규화한 skill_md가 앞선 유효 후보와 같으면 한 번만 넣는다. 기본 `pool`은 풀에 남는 후보만.
- `--dedupe word --dedupe-threshold t`: 중복 판정을 D1(`candidate_dedupe.is_duplicate_words`)로
  한다. 통제 후보 표시에도 같은 판정을 쓴다. 기본 `v1`(줄 토큰, 0.6)일 때는 임계값을 받지 않는다.
- `--stop-after-idle N`: 폴더의 최근 N회 연속 호출에서 풀에 새 후보가 없으면 더 부르지 않는다.
- `--quota-message 문구`(여러 번): 429 메시지에 이 문구가 있으면 잔액 부족으로 멈춘다.
- 기본값이 아닌 앞의 셋(`previous`, `dedupe`, `stop_after_idle`)은 `<out-dir>/run_options.json`에
  적는다. 폴더의 기록과 다른 값으로 이어 부르면 거부한다(기본값끼리는 파일을 만들지 않는다).
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate.candidate_checks import normalize_lines, validate_proposer_candidate
from evidencegate.candidate_dedupe import is_duplicate, is_duplicate_words
from evidencegate.proposer import (
    DEFAULT_BASE_URL,
    Proposer,
    ProposerConfig,
    ProposerResult,
    ProposerStopped,
    list_models,
    redact,
)
from evidencegate.proposer_inputs import load_proposer_cases

DEFAULT_INPUTS = Path(
    "runs/proposer_inputs/dev_a__ollama__gpt-oss__20b__7c66ee5bf13c__bf65e569ae37.json"
)
DEFAULT_SKILL = Path("skills/S0/SKILL.md")
DEFAULT_LEDGER = Path("runs/proposer/ledger.jsonl")
DEFAULT_CONTROLS = Path("skills/controls")
DEFAULT_MAX_COMPLETION_TOKENS = 8000
RUN_OPTIONS_FILE = "run_options.json"
# 기본값(v1 동작). 이것과 다를 때만 run_options.json을 쓴다.
DEFAULT_RUN_OPTIONS: dict[str, Any] = {
    "previous": "pool",
    "dedupe": "v1",
    "dedupe_threshold": None,
    "stop_after_idle": None,
}
# 한 출력 폴더 안에서 같아야 하는 proposer_config 항목
FIXED_CONFIG_FIELDS = (
    "model",
    "max_completion_tokens",
    "temperature",
    "reasoning_effort",
    "input_usd_per_mtok",
    "output_usd_per_mtok",
    "cases_per_call",
    "min_per_stratum",
    "prompt_sha256",
)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: httpx.Client | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "models":
            with client or httpx.Client(base_url=DEFAULT_BASE_URL, timeout=30) as c:
                for model_id in sorted(list_models(c)):
                    print(model_id, file=out)
            return 0
        return _propose(args, client, out, err)
    except ProposerStopped as error:
        print(f"stopped: {redact(str(error))}", file=err)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evidencegate-propose")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list model ids available to the API key")

    p = sub.add_parser("propose", help="call the proposer and save one result per call")
    p.add_argument("--model", required=True)
    p.add_argument("--input-usd-per-mtok", required=True, type=float)
    p.add_argument("--output-usd-per-mtok", required=True, type=float)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--budget-usd", type=float, default=1.5)
    p.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    p.add_argument("--skill", type=Path, default=DEFAULT_SKILL)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--controls-dir", type=Path, default=DEFAULT_CONTROLS)
    p.add_argument("--max-format-error-rate", type=float, default=0.05)
    p.add_argument(
        "--max-total-calls", type=int, default=None, help="cap on calls in this --out-dir"
    )
    p.add_argument(
        "--stop-when-pool", type=int, default=None, help="no more calls once the pool has N"
    )
    p.add_argument("--previous", choices=("pool", "valid"), default="pool")
    p.add_argument("--dedupe", choices=("v1", "word"), default="v1")
    p.add_argument("--dedupe-threshold", type=float, default=None)
    p.add_argument(
        "--stop-after-idle", type=int, default=None, help="no more calls after N idle calls"
    )
    p.add_argument("--quota-message", action="append", default=None, dest="quota_messages")
    return parser


def _run_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "previous": args.previous,
        "dedupe": args.dedupe,
        "dedupe_threshold": args.dedupe_threshold,
        "stop_after_idle": args.stop_after_idle,
    }


def _options_problem(options: dict[str, Any], out_dir: Path) -> str | None:
    threshold = options["dedupe_threshold"]
    if options["dedupe"] == "word" and threshold is None:
        return "--dedupe word needs --dedupe-threshold"
    if options["dedupe"] == "v1" and threshold is not None:
        return "--dedupe-threshold goes with --dedupe word only"
    if threshold is not None and not 0 < threshold <= 1:
        return f"--dedupe-threshold must be in (0, 1], got {threshold}"
    path = out_dir / RUN_OPTIONS_FILE
    recorded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    has_results = out_dir.exists() and any(out_dir.glob("call-*.json"))
    if recorded is None and has_results and options != DEFAULT_RUN_OPTIONS:
        return f"{out_dir} holds results made with the default run_options; use another --out-dir"
    if recorded is not None and recorded != options:
        return f"{out_dir} holds results with run_options {recorded}; use the same options"
    return None


def _is_valid(record: dict[str, Any]) -> bool:
    return record["candidate"] is not None and record.get("invalid_reasons") == []


def _idle_calls(results: Sequence[dict[str, Any]]) -> int:
    """마지막 풀 후보 뒤의 호출 수(풀 후보가 없으면 전체 호출 수)."""
    idle = 0
    for record in results:
        idle = 0 if record.get("in_pool") else idle + 1
    return idle


def _propose(
    args: argparse.Namespace, client: httpx.Client | None, out: TextIO, err: TextIO
) -> int:
    options = _run_options(args)
    problem = _options_problem(options, args.out_dir)
    if problem is not None:
        print(f"error: {problem}", file=err)
        return 1
    threshold = options["dedupe_threshold"]

    def duplicate(a: str, b: str, s0: str) -> bool:
        if threshold is None:
            return is_duplicate(a, b, s0)
        return is_duplicate_words(a, b, s0, threshold)

    config = ProposerConfig(
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        budget_usd=args.budget_usd,
        max_format_error_rate=args.max_format_error_rate,
    )
    s0_text = args.skill.read_text(encoding="utf-8")
    cases = load_proposer_cases(args.inputs)
    export_ids = frozenset(c.export_id for c in cases)
    proposer = Proposer(
        config,
        cases=cases,
        s0_text=s0_text,
        ledger_path=args.ledger,
        client=client,
        quota_messages=tuple(args.quota_messages or ()),
    )
    proposer_config = proposer.proposer_config()
    previous = _load_results(args.out_dir)
    for saved in previous:
        mismatched = [
            f for f in FIXED_CONFIG_FIELDS if saved["proposer_config"].get(f) != proposer_config[f]
        ]
        if mismatched:
            print(
                f"error: {args.out_dir} holds results with a different proposer_config "
                f"({', '.join(mismatched)}); use another --out-dir",
                file=err,
            )
            return 1
    controls = {
        path.parent.name: path.read_text(encoding="utf-8")
        for path in sorted(args.controls_dir.glob("*/SKILL.md"))
    }
    pool = [s for s in previous if s.get("in_pool")]
    # 이전 hypothesis 목록: 기본은 풀에 남는 후보만(티켓 §4·§5), --previous valid면 유효 후보 전부
    # (정규화한 skill_md가 같으면 한 번)
    listed = pool if options["previous"] == "pool" else [s for s in previous if _is_valid(s)]
    hypotheses: list[str] = []
    seen: list[list[str]] = []
    for s in listed:
        lines = normalize_lines(s["candidate"]["skill_md"])
        if options["previous"] == "valid" and lines in seen:
            continue
        seen.append(lines)
        hypotheses.append(str(s["candidate"]["evidence"]["hypothesis"]))
    if options != DEFAULT_RUN_OPTIONS:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / RUN_OPTIONS_FILE).write_text(json.dumps(options, indent=2) + "\n")
    results = list(previous)
    next_index = max((s["call_index"] for s in previous), default=-1) + 1

    for call_index in range(next_index, next_index + args.calls):
        if args.max_total_calls is not None and len(results) >= args.max_total_calls:
            print(f"max total calls {args.max_total_calls} reached; no more calls", file=out)
            break
        if args.stop_when_pool is not None and len(pool) >= args.stop_when_pool:
            print(f"pool reached {args.stop_when_pool}; no more calls", file=out)
            break
        idle_limit = options["stop_after_idle"]
        if idle_limit is not None and _idle_calls(results) >= idle_limit:
            print(f"no new pool candidate in the last {idle_limit} calls; no more calls", file=out)
            break
        max_cost = proposer.max_next_call_cost(call_index, hypotheses)
        print(f"call {call_index}: max cost ${max_cost:.4f}", file=out)
        try:
            result = proposer.propose(call_index, hypotheses)
        except ProposerStopped as error:
            if error.result is not None:
                _save(args.out_dir, _record(error.result, proposer_config))
            raise
        record = _record(result, proposer_config)
        candidate = result.candidate
        if candidate is None:
            status = f"format error ({result.format_error_kind})"
        else:
            reasons = validate_proposer_candidate(
                candidate.skill_md, candidate.evidence, s0_text, export_ids
            )
            record["invalid_reasons"] = list(reasons)
            if reasons:
                status = f"{candidate.candidate_id} invalid ({', '.join(reasons)})"
            else:
                status = _dedupe(record, candidate.skill_md, pool, controls, s0_text, duplicate)
                if record["in_pool"]:
                    pool.append(record)
                lines = normalize_lines(candidate.skill_md)
                if record["in_pool"] if options["previous"] == "pool" else lines not in seen:
                    seen.append(lines)
                    hypotheses.append(str(candidate.evidence["hypothesis"]))
        _save(args.out_dir, record)
        results.append(record)
        print(
            f"call {call_index}: {status}, cost ${result.cost_usd:.4f}, "
            f"cumulative ${proposer.cumulative_usd:.4f}",
            file=out,
        )
    print(_summary(results), file=out)
    return 0


def _dedupe(
    record: dict[str, Any],
    skill_md: str,
    pool: Sequence[dict[str, Any]],
    controls: dict[str, str],
    s0_text: str,
    duplicate: Callable[[str, str, str], bool] = is_duplicate,
) -> str:
    """유효한 후보의 중복 판정을 record에 적고 상태 문구를 돌려준다."""
    candidate_id = record["candidate"]["evidence"]["candidate_id"]
    for earlier in pool:  # 먼저 생성된 후보를 남긴다
        if duplicate(skill_md, earlier["candidate"]["skill_md"], s0_text):
            record["duplicate_of"] = earlier["candidate"]["evidence"]["candidate_id"]
            return f"{candidate_id} duplicate of {record['duplicate_of']}"
    marked = [name for name, text in controls.items() if duplicate(skill_md, text, s0_text)]
    record["control_duplicates"] = marked
    record["in_pool"] = True
    if marked:
        return f"{candidate_id} valid, marked as duplicate of control {', '.join(marked)}"
    return f"{candidate_id} valid"


def _summary(results: Sequence[dict[str, Any]]) -> str:
    format_kinds = Counter(
        str(r["format_error_kind"]) for r in results if r["format_error_kind"] is not None
    )
    invalid = Counter(reason for r in results for reason in (r.get("invalid_reasons") or []))
    invalid_count = sum(1 for r in results if r.get("invalid_reasons"))
    pool = [r for r in results if r.get("in_pool")]
    return (
        f"summary: calls {len(results)}, pool {len(pool)}, "
        f"proposer duplicates {sum(1 for r in results if r.get('duplicate_of'))}, "
        f"control-marked {sum(1 for r in pool if r.get('control_duplicates'))}, "
        f"format errors {sum(format_kinds.values())}{_counts(format_kinds)}, "
        f"invalid {invalid_count}{_counts(invalid)}"
    )


def _counts(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return " (" + ", ".join(f"{k} {v}" for k, v in sorted(counter.items())) + ")"


def _load_results(out_dir: Path) -> list[dict[str, Any]]:
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("call-*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _record(result: ProposerResult, proposer_config: dict[str, Any]) -> dict[str, Any]:
    candidate = result.candidate
    return {
        "call_index": result.call_index,
        "proposer_config": proposer_config,
        "cost_usd": result.cost_usd,
        "format_error": result.format_error,
        "format_error_kind": result.format_error_kind,
        # §5 유효성 검사에 걸린 이름. 형식 오류면 None
        "invalid_reasons": None,
        # 앞선 풀 후보와 중복이면 그 candidate_id(풀에서 뺌)
        "duplicate_of": None,
        # 중복인 통제 후보 이름(남기되 표시). 유효·비중복 후보에만 채움
        "control_duplicates": None,
        "in_pool": False,
        "candidate": None
        if candidate is None
        else {"skill_md": candidate.skill_md, "evidence": candidate.evidence},
    }


def _save(out_dir: Path, record: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"call-{record['call_index']:03d}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
