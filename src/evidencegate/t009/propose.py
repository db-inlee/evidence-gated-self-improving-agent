"""Ticket 009 proposer 진입점(사전 등록 2.1절). v1 `propose_cli`에 009 고정 값을 넘기는 얇은 층이다.

    uv run python -m evidencegate.t009.propose models
    uv run python -m evidencegate.t009.propose propose --calls N

- 고정 값(명령줄로 바꿀 수 없음): 모델 gpt-5.4, reasoning low, max_completion_tokens 8000,
  단가 $2.5 / $15 per Mtok(v1 `POOL.json`과 같음), 비용 상한 $5, 호출 상한 80(출력 폴더 전체),
  풀 40에서 멈춤. 경로는 009 내보내기, 새 원장 `runs/proposer-009/ledger.jsonl`, 출력
  `runs/proposer-009/pool`, 통제 `skills/controls`, S0 `skills/S0/SKILL.md`다.
- 호출 전 점검(하나라도 어긋나면 종료 코드 1, API 요청 없음): 사전 등록 009 해시, 내보내기 파일과
  `.sha256`, 층과 호출 0~79 추출(`t009.proposer_inputs.check_sampling`), 프롬프트 sha256, 원장·출력
  경로가 v1 경로가 아님.
- 응답 모델 이름: "첫 응답의 모델 이름을 원장에 기록"(2.1절)을 원장 폴더의 옆 파일
  `runs/proposer-009/response_models.jsonl`로 한다(v1 원장 형식을 지키려는 것, 보고서 2026-11-05-04
  결정). httpx 응답 훅이 `chat/completions` 200 응답 JSON의 `model`만 읽는다. 헤더·요청 본문은
  쓰지 않는다. call_index는 그 시점에 출력 폴더에 저장된 호출 수다(v1은 응답 뒤에 결과를 저장한다).
- API 키는 v1과 같이 환경 변수 OPENAI_API_KEY에서만 읽는다(이 모듈은 키를 읽지 않는다).
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate import propose_cli
from evidencegate.analysis import REPO_ROOT
from evidencegate.proposer import DEFAULT_BASE_URL, prompt_sha256
from evidencegate.proposer_inputs import load_proposer_cases
from evidencegate.t009.preflight import check_preregistration_009
from evidencegate.t009.proposer_inputs import MAX_CALLS, OUT_PATH, check_sampling

MODEL = "gpt-5.4"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 8000
INPUT_USD_PER_MTOK = 2.5
OUTPUT_USD_PER_MTOK = 15.0
BUDGET_USD = 5.0
POOL_TARGET = 40
PROMPT_SHA256 = "58ffd9f0b92c4d063adb75ddaaa8477a2c381b772ec70a0a7dbdd54f2336e2c6"
RUN_DIR = Path("runs/proposer-009")
V1_PATHS = (Path("runs/proposer/ledger.jsonl"), Path("runs/proposer/pool"))
TIMEOUT_S = 180.0


@dataclass(frozen=True)
class Paths:
    inputs: Path
    ledger: Path
    out_dir: Path
    response_models: Path
    skill: Path
    controls: Path

    @classmethod
    def default(cls, root: Path) -> "Paths":
        return cls(
            inputs=root / OUT_PATH,
            ledger=root / RUN_DIR / "ledger.jsonl",
            out_dir=root / RUN_DIR / "pool",
            response_models=root / RUN_DIR / "response_models.jsonl",
            skill=root / "skills/S0/SKILL.md",
            controls=root / "skills/controls",
        )


def write_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def v1_argv(paths: Paths, calls: int) -> list[str]:
    fixed = {
        "--model": MODEL,
        "--reasoning-effort": REASONING_EFFORT,
        "--max-completion-tokens": str(MAX_COMPLETION_TOKENS),
        "--input-usd-per-mtok": str(INPUT_USD_PER_MTOK),
        "--output-usd-per-mtok": str(OUTPUT_USD_PER_MTOK),
        "--budget-usd": str(BUDGET_USD),
        "--max-total-calls": str(MAX_CALLS),
        "--stop-when-pool": str(POOL_TARGET),
        "--calls": str(calls),
        "--inputs": str(paths.inputs),
        "--ledger": str(paths.ledger),
        "--out-dir": str(paths.out_dir),
        "--skill": str(paths.skill),
        "--controls-dir": str(paths.controls),
    }
    return ["propose", *(x for pair in fixed.items() for x in pair)]


def preflight_problems(root: Path, paths: Paths) -> list[str]:
    problems = []
    prereg = check_preregistration_009(root)
    if not prereg.ok:
        problems.append(f"preregistration: {prereg.detail}")
    sha_file = paths.inputs.with_name(paths.inputs.name + ".sha256")
    if not paths.inputs.exists() or not sha_file.exists():
        problems.append(f"export: missing {paths.inputs} or its .sha256")
    else:
        actual = hashlib.sha256(paths.inputs.read_bytes()).hexdigest()
        recorded = sha_file.read_text().split()[0]
        if actual != recorded:
            problems.append(f"export: sha256 {actual} does not match {recorded}")
        sampling = check_sampling(load_proposer_cases(paths.inputs))
        if not sampling.ok:
            problems.append(f"strata: {sampling.detail}")
    if prompt_sha256() != PROMPT_SHA256:
        problems.append(f"prompt sha256 {prompt_sha256()} is not {PROMPT_SHA256}")
    for name, path in (("ledger", paths.ledger), ("out_dir", paths.out_dir)):
        if any(path.resolve() == (root / v1).resolve() for v1 in V1_PATHS):
            problems.append(f"{name}: {path} is a v1 path")
    return problems


def _saved_calls(out_dir: Path) -> int:
    return len(list(out_dir.glob("call-*.json"))) if out_dir.exists() else 0


def recording_client(paths: Paths, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """`chat/completions` 200 응답의 `model` 값만 옆 파일에 적는 클라이언트."""

    def hook(response: httpx.Response) -> None:
        if response.status_code != 200 or not response.request.url.path.endswith(
            "/chat/completions"
        ):
            return
        response.read()
        model = response.json().get("model")
        row: dict[str, Any] = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "call_index": _saved_calls(paths.out_dir),
            "requested_model": MODEL,
            "model": model,
        }
        paths.response_models.parent.mkdir(parents=True, exist_ok=True)
        with paths.response_models.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return httpx.Client(
        base_url=DEFAULT_BASE_URL,
        timeout=TIMEOUT_S,
        transport=transport,
        event_hooks={"response": [hook]},
    )


def _first_response_model(paths: Paths) -> str | None:
    if not paths.response_models.exists():
        return None
    lines = paths.response_models.read_text(encoding="utf-8").splitlines()
    return str(json.loads(lines[0])["model"]) if lines else None


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
    paths: Paths | None = None,
    transport: httpx.BaseTransport | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009-propose")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list model ids available to the API key")
    p = sub.add_parser("propose", help="call the proposer with the fixed Ticket 009 settings")
    p.add_argument("--calls", type=int, default=1)
    args = parser.parse_args(argv)
    paths = paths or Paths.default(root)
    if args.command == "models":
        client = httpx.Client(base_url=DEFAULT_BASE_URL, timeout=30, transport=transport)
        return propose_cli.main(["models"], client=client, out=out, err=err)
    problems = preflight_problems(root, paths)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=err)
        print("error: preflight failed; no request was sent", file=err)
        return 1
    with recording_client(paths, transport) as client:
        code = propose_cli.main(v1_argv(paths, args.calls), client=client, out=out, err=err)
    first = _first_response_model(paths)
    if first is not None:
        print(f"first response model: {first} (requested {MODEL})", file=out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
