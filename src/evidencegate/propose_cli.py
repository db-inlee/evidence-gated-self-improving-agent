"""proposer 진입점 (Ticket 004 §4). 실측과 풀 생성이 같은 명령을 쓴다.

    uv run python -m evidencegate.propose_cli models
    uv run python -m evidencegate.propose_cli propose --model <m> \\
        --input-usd-per-mtok <in> --output-usd-per-mtok <out> --out-dir <dir> [--calls N]

`propose`는 호출마다 `<out-dir>/call-NNN.json`을 쓴다. 폴더에 결과가 있으면 다음 call_index부터
이어서 부르고, 앞선 결과 중 형식이 맞는 후보의 hypothesis를 프롬프트에 넣는다. 한 폴더는 한
proposer_config만 담는다(다르면 거부). 원장(기본 `runs/proposer/ledger.jsonl`)은 실측과 풀 생성이
함께 써서 누적 비용이 이어진다. 중단(예산, 할당량, 인증, 서버 오류, 형식 오류율)은 종료 코드 2다.
API 키는 환경 변수 OPENAI_API_KEY에서만 읽는다.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx

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
DEFAULT_MAX_COMPLETION_TOKENS = 8000
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
    return parser


def _propose(
    args: argparse.Namespace, client: httpx.Client | None, out: TextIO, err: TextIO
) -> int:
    config = ProposerConfig(
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        budget_usd=args.budget_usd,
    )
    proposer = Proposer(
        config,
        cases=load_proposer_cases(args.inputs),
        s0_text=args.skill.read_text(encoding="utf-8"),
        ledger_path=args.ledger,
        client=client,
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
    hypotheses = [s["candidate"]["evidence"]["hypothesis"] for s in previous if s["candidate"]]
    next_index = max((s["call_index"] for s in previous), default=-1) + 1

    for call_index in range(next_index, next_index + args.calls):
        max_cost = proposer.max_next_call_cost(call_index, hypotheses)
        print(f"call {call_index}: max cost ${max_cost:.4f}", file=out)
        try:
            result = proposer.propose(call_index, hypotheses)
        except ProposerStopped as error:
            if error.result is not None:
                _save(args.out_dir, error.result, proposer_config)
            raise
        _save(args.out_dir, result, proposer_config)
        if result.candidate is not None:
            hypotheses.append(str(result.candidate.evidence["hypothesis"]))
            status = result.candidate.candidate_id
        else:
            status = f"format error ({result.format_error_kind})"
        print(
            f"call {call_index}: {status}, cost ${result.cost_usd:.4f}, "
            f"cumulative ${proposer.cumulative_usd:.4f}",
            file=out,
        )
    return 0


def _load_results(out_dir: Path) -> list[dict[str, Any]]:
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("call-*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _save(out_dir: Path, result: ProposerResult, proposer_config: dict[str, Any]) -> None:
    candidate = result.candidate
    data = {
        "call_index": result.call_index,
        "proposer_config": proposer_config,
        "cost_usd": result.cost_usd,
        "format_error": result.format_error,
        "format_error_kind": result.format_error_kind,
        "candidate": None
        if candidate is None
        else {"skill_md": candidate.skill_md, "evidence": candidate.evidence},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"call-{result.call_index:03d}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
