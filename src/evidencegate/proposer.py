"""OpenAI API proposer (Ticket 004 §4).

proposer는 Dev-A 내보내기(`proposer_inputs`)만 본다. 호출 1회가 후보 1개다. 호출마다 실패 사례
k개를 층화해 뽑고(근거 부족 / 처분 오답 각각 최소 min_per_stratum개), 현재 S0 전문, 도구 명세,
지금까지 유효한 후보의 hypothesis 목록과 함께 보낸다. 응답은 structured outputs(JSON schema,
strict)로 받는다.

안전 장치:
- API 키는 환경 변수 OPENAI_API_KEY에서 요청마다 읽어 헤더에만 넣는다. 객체·설정·원장·예외
  메시지 어디에도 남기지 않으며, 서버가 돌려준 오류 문구의 키 조각(`sk-…`)도 지운다.
- 호출 전에 "누적 비용 + 다음 호출의 최대 비용 > 예산"이면 호출하지 않고 멈춘다. 다음 호출의
  최대 비용은 입력 토큰 추정(문자 수 / chars_per_token_estimate, 올림)과 출력 상한
  (max_completion_tokens, 추론 토큰 포함)으로 계산한다. API는 비용을 알려 주지 않으므로 비용은
  토큰 x 단가표다. 누적 비용은 원장 파일에서 이어서 읽는다(실행이 나뉘어도 예산이 이어짐).
- 서버 오류(5xx, 할당량이 아닌 429, 연결 오류)는 3회까지 재시도한다. 할당량·잔액 부족, 인증
  오류, 그 밖의 4xx는 재시도하지 않고 멈춘다.
- 형식 오류(파싱 실패, 스키마 불일치, 거절, 출력 상한 도달)는 기록하고 계속하되, 이 인스턴스에서
  형식 오류 수 / 호출 수가 max_format_error_rate를 넘으면 멈춘다.
"""

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evidencegate.proposer_inputs import ProposerCase
from opsworld import agent_env

API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
FAILURE_TYPES = ("PREMATURE_ACTION", "WRONG_INTERVENTION")
EFFECTS = ("up", "down", "none")
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*.]+")

SYSTEM_PROMPT = (
    "You improve the operating skill of an on-call SRE agent that handles incidents in a "
    "simulated environment. You will see the current skill, the tools the agent can call, and "
    "incidents where the agent failed. Propose one improved skill."
)
USER_TEMPLATE = """## Current skill (SKILL.md)
{skill}

## Tools the agent can call
{tools}

## Failed incidents
Each incident shows the alert, the tool calls with their outputs, the action taken, whether the \
disposition was correct, whether the evidence required for that action was gathered before \
acting (evidence_safe), the failure labels, and the missing evidence.
{cases}

## Changes already proposed
{previous}

## Instructions
- Choose ONE failure type shown above and change the skill in ONE way to address it.
- Pick a different failure type or a different change from the ones already proposed.
- Return the complete new SKILL.md text and an evidence card.
- The new skill must start with "# Incident Response Skill".
- Do not write numeric thresholds, service names, deployment IDs, versions, or times.
- Use only the tools listed above. Do not add steps that need other tools or actions \
(for example logs, restarts, failover, paging, or tickets).
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "skill_candidate",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "skill_md": {"type": "string"},
            "failure_type": {"type": "string", "enum": list(FAILURE_TYPES)},
            "hypothesis": {"type": "string"},
            "expected_effect": {
                "type": "object",
                "properties": {
                    "cd": {"type": "string", "enum": list(EFFECTS)},
                    "es": {"type": "string", "enum": list(EFFECTS)},
                },
                "required": ["cd", "es"],
                "additionalProperties": False,
            },
        },
        "required": ["skill_md", "failure_type", "hypothesis", "expected_effect"],
        "additionalProperties": False,
    },
}


class ProposerStopped(RuntimeError):
    """예산, 할당량, 인증, 반복된 서버 오류, 형식 오류율 때문에 proposer를 멈춘다."""


@dataclass(frozen=True)
class ProposerConfig:
    model: str
    max_completion_tokens: int  # 출력 상한(추론 토큰 포함)
    input_usd_per_mtok: float  # 단가표: 입력 100만 토큰당 달러
    output_usd_per_mtok: float  # 단가표: 출력 100만 토큰당 달러(추론 토큰 포함)
    temperature: float | None = None  # 모델이 지원할 때만 보냄
    reasoning_effort: str | None = None  # 추론형 모델일 때만 보냄
    budget_usd: float = 1.5
    cases_per_call: int = 6
    min_per_stratum: int = 2
    max_retries: int = 3
    max_format_error_rate: float = 0.05
    retry_wait_s: float = 2.0
    chars_per_token_estimate: float = 3.0  # 입력 토큰 추정(보수적으로 작게)
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 180.0


@dataclass(frozen=True)
class ProposerCandidate:
    candidate_id: str
    skill_md: str
    evidence: dict[str, Any]  # 증거 카드


@dataclass(frozen=True)
class ProposerResult:
    call_index: int
    candidate: ProposerCandidate | None
    format_error: str | None
    cost_usd: float


def prompt_sha256() -> str:
    fixed = json.dumps(
        {"system": SYSTEM_PROMPT, "user": USER_TEMPLATE, "schema": RESPONSE_SCHEMA},
        sort_keys=True,
    )
    return hashlib.sha256(fixed.encode("utf-8")).hexdigest()


def redact(text: str) -> str:
    """키 원문과 `sk-…` 조각을 지운다."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        text = text.replace(key, "[redacted]")
    return _KEY_PATTERN.sub("[redacted]", text)


def _auth_headers() -> dict[str, str]:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise ProposerStopped(f"{API_KEY_ENV} is not set")
    return {"Authorization": f"Bearer {key}"}


def list_models(client: httpx.Client) -> list[str]:
    response = client.get("/models", headers=_auth_headers())
    if response.status_code != 200:
        raise ProposerStopped(
            redact(f"model list failed ({response.status_code}): {response.text}")
        )
    return [m["id"] for m in response.json().get("data", [])]


def sample_cases(
    cases: Sequence[ProposerCase], call_index: int, k: int, min_per_stratum: int
) -> list[ProposerCase]:
    """호출마다 고정 seed로 k개를 뽑는다. 근거 부족(CD 참, ES 거짓)과 처분 오답(CD 거짓)을 각각
    최소 min_per_stratum개 넣는다."""
    ordered = sorted(cases, key=lambda c: c.export_id)
    evidence_only = [c for c in ordered if c.correct_disposition and not c.evidence_safe]
    wrong = [c for c in ordered if not c.correct_disposition]
    if len(evidence_only) < min_per_stratum or len(wrong) < min_per_stratum:
        raise ValueError(
            f"stratum too small: evidence-only {len(evidence_only)}, wrong {len(wrong)}, "
            f"need {min_per_stratum} each"
        )
    rng = random.Random(f"proposer-sample:{call_index}")
    picked = rng.sample(evidence_only, min_per_stratum) + rng.sample(wrong, min_per_stratum)
    rest = [c for c in ordered if c not in picked]
    picked += rng.sample(rest, k - len(picked))
    rng.shuffle(picked)
    return picked


class Proposer:
    def __init__(
        self,
        config: ProposerConfig,
        *,
        cases: Sequence[ProposerCase],
        s0_text: str,
        ledger_path: Path,
        client: httpx.Client | None = None,
    ) -> None:
        _auth_headers()  # 키가 없으면 여기서 멈춘다(키 값은 보관하지 않음)
        self.config = config
        self._cases = list(cases)
        self._s0_text = s0_text
        self._ledger_path = ledger_path
        self._client = client or httpx.Client(base_url=config.base_url, timeout=config.timeout_s)
        self.cumulative_usd = self._load_cumulative()
        self.calls = 0
        self.format_errors = 0

    def proposer_config(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "prompt_sha256": prompt_sha256(),
            "sample_rule": "random.Random('proposer-sample:<call_index>'), stratified",
            "endpoint": "chat/completions",
        }

    def messages(
        self, call_index: int, previous_hypotheses: Sequence[str] = ()
    ) -> tuple[list[dict[str, str]], list[ProposerCase]]:
        c = self.config
        shown = sample_cases(self._cases, call_index, c.cases_per_call, c.min_per_stratum)
        tools = "\n".join(f"- {s.name} ({s.kind}): {s.description}" for s in agent_env.TOOL_SPECS)
        cases = "\n".join(json.dumps(case.to_json(), sort_keys=True) for case in shown)
        previous = "\n".join(f"- {h}" for h in previous_hypotheses) or "(none yet)"
        user = USER_TEMPLATE.format(
            skill=self._s0_text, tools=tools, cases=cases, previous=previous
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ], shown

    def max_next_call_cost(self, call_index: int, previous_hypotheses: Sequence[str] = ()) -> float:
        messages, _ = self.messages(call_index, previous_hypotheses)
        chars = len(json.dumps(messages)) + len(json.dumps(RESPONSE_SCHEMA))
        input_tokens = math.ceil(chars / self.config.chars_per_token_estimate)
        return self._cost(input_tokens, self.config.max_completion_tokens)

    def propose(self, call_index: int, previous_hypotheses: Sequence[str] = ()) -> ProposerResult:
        c = self.config
        messages, shown = self.messages(call_index, previous_hypotheses)
        body: dict[str, Any] = {
            "model": c.model,
            "messages": messages,
            "max_completion_tokens": c.max_completion_tokens,
            "response_format": {"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        }
        if c.temperature is not None:
            body["temperature"] = c.temperature
        if c.reasoning_effort is not None:
            body["reasoning_effort"] = c.reasoning_effort

        for attempt in range(c.max_retries + 1):
            max_cost = self.max_next_call_cost(call_index, previous_hypotheses)
            if self.cumulative_usd + max_cost > c.budget_usd:
                raise ProposerStopped(
                    f"budget: cumulative ${self.cumulative_usd:.4f} + next call up to "
                    f"${max_cost:.4f} exceeds ${c.budget_usd:.2f}"
                )
            start = time.perf_counter()
            try:
                response = self._client.post(
                    "/chat/completions", json=body, headers=_auth_headers(), timeout=c.timeout_s
                )
            except httpx.TransportError as error:
                self._log(call_index, attempt, "server_error", error=f"connection: {error}")
                self._wait_or_stop(attempt, f"connection error: {error}")
                continue
            elapsed = time.perf_counter() - start
            status = response.status_code
            if status == 200:
                return self._handle_success(call_index, attempt, response.json(), shown, elapsed)
            message = _error_message(response)
            code = _error_code(response)
            if status == 429 and code == "insufficient_quota":
                self._log(call_index, attempt, "quota", error=message)
                raise ProposerStopped(redact(f"insufficient quota or balance: {message}"))
            if status in (401, 403):
                self._log(call_index, attempt, "auth_error", error=message)
                raise ProposerStopped(redact(f"authentication failed ({status}): {message}"))
            if status >= 500 or status == 429:
                self._log(call_index, attempt, "server_error", error=f"{status}: {message}")
                self._wait_or_stop(attempt, f"{status}: {message}")
                continue
            self._log(call_index, attempt, "client_error", error=f"{status}: {message}")
            raise ProposerStopped(redact(f"request rejected ({status}): {message}"))
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------

    def _handle_success(
        self,
        call_index: int,
        attempt: int,
        data: dict[str, Any],
        shown: list[ProposerCase],
        elapsed: float,
    ) -> ProposerResult:
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        cost = self._cost(input_tokens, output_tokens)
        self.cumulative_usd += cost
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        candidate: ProposerCandidate | None = None
        format_error: str | None = None
        try:
            candidate = _parse_candidate(message, choice.get("finish_reason"), call_index, shown)
        except ValueError as error:
            format_error = str(error)
        self._log(
            call_index,
            attempt,
            "ok" if format_error is None else "format_error",
            input_tokens=input_tokens,
            cached_input_tokens=int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            ),
            output_tokens=output_tokens,
            reasoning_tokens=int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            ),
            cost_usd=cost,
            duration_s=elapsed,
            error=format_error,
            content=message.get("content"),
        )
        self.calls += 1
        if format_error is not None:
            self.format_errors += 1
            rate = self.format_errors / self.calls
            if rate > self.config.max_format_error_rate:
                raise ProposerStopped(
                    f"format error rate {self.format_errors}/{self.calls} = {rate:.1%} exceeds "
                    f"{self.config.max_format_error_rate:.0%}: {format_error}"
                )
        return ProposerResult(call_index, candidate, format_error, cost)

    def _wait_or_stop(self, attempt: int, message: str) -> None:
        if attempt >= self.config.max_retries:
            raise ProposerStopped(redact(f"server error after {attempt + 1} attempts: {message}"))
        time.sleep(self.config.retry_wait_s)

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        c = self.config
        return (
            input_tokens / 1e6 * c.input_usd_per_mtok + output_tokens / 1e6 * c.output_usd_per_mtok
        )

    def _load_cumulative(self) -> float:
        if not self._ledger_path.exists():
            return 0.0
        total = 0.0
        for line in self._ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line).get("cost_usd") or 0.0)
        return total

    def _log(self, call_index: int, attempt: int, status: str, **fields: Any) -> None:
        entry: dict[str, Any] = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "call_index": call_index,
            "attempt": attempt,
            "model": self.config.model,
            "status": status,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            **fields,
            "cumulative_usd": self.cumulative_usd,
        }
        if entry.get("error") is not None:
            entry["error"] = redact(str(entry["error"]))
        if entry.get("content") is not None:
            entry["content"] = redact(str(entry["content"]))
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_candidate(
    message: dict[str, Any],
    finish_reason: str | None,
    call_index: int,
    shown: list[ProposerCase],
) -> ProposerCandidate:
    if message.get("refusal"):
        raise ValueError(f"refusal: {message['refusal']}")
    if finish_reason == "length":
        raise ValueError("output hit max_completion_tokens")
    content = message.get("content")
    if not content:
        raise ValueError("empty content")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"content is not JSON: {error}") from None
    required = {"skill_md", "failure_type", "hypothesis", "expected_effect"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError(f"unexpected fields: {sorted(data) if isinstance(data, dict) else data!r}")
    effect = data["expected_effect"]
    if (
        not isinstance(data["skill_md"], str)
        or data["failure_type"] not in FAILURE_TYPES
        or not isinstance(data["hypothesis"], str)
        or not data["hypothesis"].strip()
        or not isinstance(effect, dict)
        or set(effect) != {"cd", "es"}
        or effect["cd"] not in EFFECTS
        or effect["es"] not in EFFECTS
    ):
        raise ValueError("field values do not match the schema")
    candidate_id = f"cand-{call_index:03d}"
    return ProposerCandidate(
        candidate_id=candidate_id,
        skill_md=data["skill_md"],
        evidence={
            "candidate_id": candidate_id,
            "failure_type": data["failure_type"],
            "source_cases": [c.export_id for c in shown],
            "hypothesis": data["hypothesis"],
            "expected_effect": {"cd": effect["cd"], "es": effect["es"]},
        },
    )


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        return str(error.get("message") or response.text)
    except ValueError:
        return response.text


def _error_code(response: httpx.Response) -> str | None:
    try:
        error = response.json().get("error") or {}
        code = error.get("code")
        return str(code) if code is not None else None
    except ValueError:
        return None
