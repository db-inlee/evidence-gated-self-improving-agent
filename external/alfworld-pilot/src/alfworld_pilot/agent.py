"""S0 에이전트: 최소 ReAct 루프 (Ticket 008 §4 P2).

관측 → (지침 + SKILL.md + 이력) → 행동 하나. 행동은 `act(command)` 도구 호출 하나다.

- 프롬프트에 넣는 것: 지침(명령 형식 템플릿 포함), SKILL.md, 첫 관측, 이후 행동과 관측.
- 넣지 않는 것(결정 0·2): 가능한 명령 목록, 전문가 계획, `facts`, 게임 파일 경로. 실제 환경
  어댑터는 `won`과 가능한 명령 목록만 요청하고, 목록은 E-b용 참·거짓으로만 밖에 낸다.
- 형식 오류(v1 개념): 도구 호출이 하나가 아님, 없는 도구, command가 빈 문자열·여러 줄·문자열 아님,
  서버의 도구 호출 파싱 실패. 오류 문장을 tool 결과로 돌려주고 환경은 움직이지 않는다. 정상 호출이
  나오면 연속 횟수를 0으로 되돌린다. 텍스트 응답은 재촉 문구를 받고, 연속 형식 오류 수는 바꾸지
  않는다.
- 종료: won / 걸음 상한(max_steps) / 연속 텍스트 응답(no_tool_call) / 연속 형식 오류(format_error)
  / LLM 호출 상한(max_llm_calls).
- 이력 자르기: system과 첫 관측은 항상 둔다. 나머지 턴은 오래된 것부터 빼서 문자 예산 안에 넣고,
  뺀 자리에 `[N earlier steps omitted]`를 넣는다. 예산 = (num_ctx - reserve) x chars_per_token.

진단 기록(2026-11-03 amendment): 텍스트 턴마다 assistant 내용(앞 TEXT_CONTENT_LIMIT자)과 전체 길이,
thinking 길이를 `text_turn_diagnostics`에 남긴다. 기록만 더할 뿐 프롬프트·판정·흐름은 그대로다
(tests/test_p2_diagnostics.py가 수정 전 golden과 비교한다).

LLM이 읽는 텍스트는 영어이고 정답 단서를 넣지 않는다. 바꾸면 `instructions_sha256`이 바뀐다.
"""

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from alfworld_pilot import constants as C
from alfworld_pilot.evidence import StepRecord

StepTrace = StepRecord
TEXT_CONTENT_LIMIT = C.TEXT_CONTENT_LIMIT
DIAGNOSTIC_FIELDS = ("text_turn_diagnostics",)  # 진단 전용, 결과 비교에서 뺀다

TEMPLATES = "\n".join(
    (
        "go to {recep}",
        "take {obj} from {recep}",
        "move {obj} to {recep}",
        "open {recep}",
        "close {recep}",
        "heat {obj} with {recep}",
        "cool {obj} with {recep}",
        "clean {obj} with {recep}",
        "use {obj}",
        "examine {x}",
        "look",
        "inventory",
    )
)
INSTRUCTIONS = (
    "You are an agent in a text-based household environment. "
    "Complete the task stated in the first observation.\n"
    "Each turn, call the act tool with exactly one command. "
    "The environment replies with an observation.\n"
    "Commands take these forms, where {recep}, {obj} and {x} are names written exactly as in "
    "the observations (for example, drawer 1):\n" + TEMPLATES + "\nFollow the skill below."
)
NUDGE = "No command was given. Call the act tool with exactly one command."
TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Send one command to the environment.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "One command."}},
            "required": ["command"],
        },
    },
}
SEED_RULE = "derive_llm_seed(game_rel, repeat): sha256, first 8 bytes mod 2^31"
HISTORY_RULE = (
    "keep system + first observation; drop oldest turns until the prompt fits the character "
    "budget; insert '[N earlier steps omitted]'"
)


def instructions_sha256() -> str:
    fixed = json.dumps({"instructions": INSTRUCTIONS, "nudge": NUDGE, "tool": TOOL}, sort_keys=True)
    return hashlib.sha256(fixed.encode()).hexdigest()


def derive_llm_seed(game_rel: str, repeat: int) -> int:
    digest = hashlib.sha256(f"{game_rel}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % C.SEED_MODULUS


@dataclass(frozen=True)
class PolicySettings:
    model: str = C.MODEL
    think: str = C.THINK
    temperature: float = C.TEMPERATURE
    num_ctx: int = C.NUM_CTX
    keep_alive: str = C.KEEP_ALIVE
    max_steps: int = C.MAX_STEPS
    max_consecutive_format_errors: int = C.MAX_CONSECUTIVE_FORMAT_ERRORS
    max_consecutive_text_turns: int = C.MAX_CONSECUTIVE_TEXT_TURNS
    max_llm_calls: int = C.MAX_LLM_CALLS
    reserve_tokens: int = C.RESERVE_TOKENS
    chars_per_token: int = C.CHARS_PER_TOKEN
    history_budget_override: int | None = None  # 테스트 전용

    @property
    def history_budget_chars(self) -> int:
        if self.history_budget_override is not None:
            return self.history_budget_override
        return (self.num_ctx - self.reserve_tokens) * self.chars_per_token

    def config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "think": self.think,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "keep_alive": self.keep_alive,
            "seed_rule": SEED_RULE,
            "instructions_sha256": instructions_sha256(),
            "max_steps": self.max_steps,
            "max_llm_calls": self.max_llm_calls,
            "stop": {"max_consecutive_text_turns": self.max_consecutive_text_turns},
            "format_error_policy": {
                "feedback": "error_as_tool_result",
                "max_consecutive_format_errors": self.max_consecutive_format_errors,
            },
            "history": {
                "rule": HISTORY_RULE,
                "budget_chars": self.history_budget_chars,
                "reserve_tokens": self.reserve_tokens,
                "chars_per_token": self.chars_per_token,
            },
            "command_list_in_prompt": False,
            "expert_plan_in_prompt": False,
        }


@dataclass(frozen=True)
class Outcome:
    observation: str
    won: bool
    done: bool
    admissible: bool  # 행동이 직전 가능한 명령 목록에 있었는가(E-b)


class Env(Protocol):
    def reset(self) -> str: ...

    def step(self, command: str) -> Outcome: ...


class ChatClient(Protocol):
    def chat(self, body: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EpisodeTrace:
    steps: tuple[StepTrace, ...]
    won: bool
    stop_reason: str
    first_observation: str
    format_errors: int
    text_turns: int
    llm_calls: int
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    truncated_calls: int  # 이력을 잘라 보낸 LLM 호출 수
    max_omitted: int  # 한 호출에서 뺀 턴 수의 최댓값
    wall_s: float
    text_turn_diagnostics: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Message = dict[str, Any]
Turn = tuple[Message, ...]


def step_turn(command: str, observation: str) -> Turn:
    return (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "act", "arguments": {"command": command}}}],
        },
        {"role": "tool", "tool_name": "act", "content": observation},
    )


def _size(messages: Sequence[Message]) -> int:
    return sum(len(json.dumps(m)) for m in messages)


def build_messages(
    system: str,
    first_observation: str,
    turns: Sequence[tuple[str, str] | Turn],
    *,
    budget_chars: int,
) -> tuple[list[Message], int]:
    """이력 자르기. 돌려주는 값은 (메시지, 뺀 턴 수)."""
    head: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": first_observation},
    ]
    blocks: list[Turn] = [step_turn(t[0], t[1]) if isinstance(t[0], str) else t for t in turns]
    for omitted in range(len(blocks) + 1):
        note = (
            [{"role": "user", "content": f"[{omitted} earlier steps omitted]"}] if omitted else []
        )
        messages = head + note + [m for b in blocks[omitted:] for m in b]
        if _size(messages) <= budget_chars or omitted == len(blocks):
            return messages, omitted
    raise AssertionError("unreachable")


def _command(calls: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """(명령, 오류 문장) 중 하나."""
    if len(calls) != 1:
        return None, "Format error: call the act tool exactly once per turn."
    fn = calls[0].get("function") or {}
    if fn.get("name") != "act":
        return None, f"Format error: unknown tool {fn.get('name')!r}. Use the act tool."
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    command = args.get("command") if isinstance(args, dict) else None
    if not isinstance(command, str) or not command.strip() or "\n" in command.strip():
        return None, "Format error: act needs one command as a single-line string."
    return command.strip(), None


def run_episode(
    env: Env, skill_text: str, chat: ChatClient, *, seed: int, settings: PolicySettings
) -> EpisodeTrace:
    s = settings
    start = time.perf_counter()
    system = INSTRUCTIONS + "\n\n" + skill_text
    first = env.reset()
    turns: list[Turn] = []
    steps: list[StepTrace] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    format_errors = text_turns = consecutive_errors = consecutive_text = 0
    diagnostics: list[dict[str, Any]] = []
    truncated_calls = max_omitted = 0
    won = False
    stop_reason = ""
    command: str | None
    error: str | None
    while not stop_reason:
        if len(prompt_tokens) >= s.max_llm_calls:
            stop_reason = "max_llm_calls"
            break
        messages, omitted = build_messages(
            system, first, turns, budget_chars=s.history_budget_chars
        )
        if omitted:
            truncated_calls += 1
            max_omitted = max(max_omitted, omitted)
        reply = chat.chat(
            {
                "model": s.model,
                "messages": messages,
                "tools": [TOOL],
                "stream": False,
                "think": s.think,
                "keep_alive": s.keep_alive,
                "options": {"temperature": s.temperature, "seed": seed, "num_ctx": s.num_ctx},
            }
        )
        prompt_tokens.append(int(reply.get("prompt_eval_count", 0)))
        completion_tokens.append(int(reply.get("eval_count", 0)))

        if "parse_error" in reply:
            command, error = None, "Format error: the tool call could not be parsed."
            assistant: Message | None = None
        else:
            message = reply["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                content = str(message.get("content") or "")
                diagnostics.append(
                    {
                        "llm_call": len(prompt_tokens) - 1,
                        "content": content[:TEXT_CONTENT_LIMIT],
                        "content_chars": len(content),
                        "thinking_chars": len(str(message.get("thinking") or "")),
                    }
                )
                text_turns += 1
                consecutive_text += 1
                turns.append(
                    (
                        {"role": "assistant", "content": message.get("content", "")},
                        {"role": "user", "content": NUDGE},
                    )
                )
                if consecutive_text >= s.max_consecutive_text_turns:
                    stop_reason = "no_tool_call"
                continue
            consecutive_text = 0
            command, error = _command(calls)
            assistant = {
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": calls,
            }

        if error is not None:
            format_errors += 1
            consecutive_errors += 1
            feedback = {"role": "tool", "tool_name": "act", "content": error}
            turns.append((assistant, feedback) if assistant is not None else (feedback,))
            if consecutive_errors >= s.max_consecutive_format_errors:
                stop_reason = "format_error"
            continue

        assert command is not None
        consecutive_errors = 0
        outcome = env.step(command)
        steps.append(StepTrace(command, outcome.observation, outcome.admissible))
        turns.append(step_turn(command, outcome.observation))
        if outcome.won:
            won, stop_reason = True, "won"
        elif outcome.done or len(steps) >= s.max_steps:
            stop_reason = "max_steps"

    return EpisodeTrace(
        steps=tuple(steps),
        won=won,
        stop_reason=stop_reason,
        first_observation=first,
        format_errors=format_errors,
        text_turns=text_turns,
        llm_calls=len(prompt_tokens),
        prompt_tokens=tuple(prompt_tokens),
        completion_tokens=tuple(completion_tokens),
        truncated_calls=truncated_calls,
        max_omitted=max_omitted,
        wall_s=round(time.perf_counter() - start, 3),
        text_turn_diagnostics=tuple(diagnostics),
    )


class AlfworldEnv:
    """실제 ALFWorld 텍스트 게임 어댑터. 전문가 래퍼 없음, `won`과 가능한 명령 목록만 요청한다."""

    def __init__(self, game_file: Path, max_steps: int) -> None:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        self._infos = textworld.EnvInfos(won=True, admissible_commands=True)
        env_id = textworld.gym.register_game(
            str(game_file), self._infos, max_episode_steps=max_steps, wrappers=[AlfredDemangler()]
        )
        self._env = textworld.gym.make(env_id)
        self._admissible: list[str] = []

    def reset(self) -> str:
        obs, infos = self._env.reset()
        self._admissible = list(infos["admissible_commands"])
        return str(obs)

    def step(self, command: str) -> Outcome:
        admissible = command in self._admissible
        obs, _score, done, infos = self._env.step(command)
        self._admissible = list(infos["admissible_commands"])
        return Outcome(str(obs), bool(infos["won"]), bool(done), admissible)

    def close(self) -> None:
        self._env.close()

    def last_admissible_for_tests(self) -> list[str]:
        return list(self._admissible)

    def request_infos_for_tests(self) -> dict[str, Any]:
        return {
            "admissible_commands": self._infos.admissible_commands,
            "won": self._infos.won,
            "facts": self._infos.facts,
            "extras": list(self._infos.extras),
        }


class OllamaError(RuntimeError):
    """정책 행동이 아닌 서버·연결 오류. 실행을 멈춘다(그 에피소드는 기록하지 않는다)."""


class OllamaChat:
    """Ollama `/api/chat` 클라이언트(표준 라이브러리). P2에서는 호출하지 않는다."""

    def __init__(self, base_url: str = C.OLLAMA_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=C.REQUEST_TIMEOUT_S) as response:
                result: dict[str, Any] = json.loads(response.read())
                return result
        except urllib.error.HTTPError as error:
            text = error.read().decode(errors="replace")
            if path == "/api/chat" and "pars" in text.lower() and "tool" in text.lower():
                return {"parse_error": text}
            raise OllamaError(f"Ollama {path} returned {error.code}: {text}") from error
        except (urllib.error.URLError, OSError) as error:
            raise OllamaError(f"request to Ollama failed: {error}") from error

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("/api/chat", body)

    def server_record(self, model: str) -> dict[str, Any]:
        """Ollama 버전, 모델 digest, 양자화."""
        version = self._request("/api/version")["version"]
        models = self._request("/api/tags").get("models", [])
        entry = next((m for m in models if model in (m.get("name"), m.get("model"))), None)
        if entry is None:
            raise OllamaError(f"model {model!r} not found on the Ollama server")
        return {
            "ollama_version": version,
            "model_digest": entry["digest"],
            "quantization": entry.get("details", {}).get("quantization_level"),
        }
