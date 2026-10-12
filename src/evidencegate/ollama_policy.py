"""로컬 LLM(Ollama) 정책.

LangGraph `StateGraph`로 도구 호출 루프를 만들고, LLM 호출은 `httpx`로 Ollama `/api/chat`을 직접
부른다(seed, options, think를 그대로 통제하기 위해서). 정책은 `AgentEnv`만 받는다.

- 메시지: system = 고정 지침 + SKILL.md, user = 초기 알림. 도구 정의는 `AgentEnv.tool_specs()`.
- 종료: 조치 도구가 받아들여짐(action) / 도구 호출 `max_tool_calls`회(max_steps) / 도구 호출 없는
  텍스트 응답 `max_consecutive_text_turns`회 연속(no_tool_call) / 형식 오류만
  `max_consecutive_format_errors`회 연속(error).
- 형식 오류(인자 파싱 실패, 없는 도구, 잘못된 인자, 서버의 도구 호출 파싱 실패)는 오류 메시지를
  tool 결과로 돌려주고 계속한다. 정상 도구 호출이 나오면 연속 횟수를 0으로 되돌린다. 텍스트 응답은
  연속 형식 오류 횟수를 바꾸지 않는다. 이 정책은 실험 내내 바꾸지 않는다.
- 그 밖의 서버 오류(모델 크래시, 연결 실패)는 정책 행동이 아니므로 `OllamaError`로 실행을 멈춘다.
  결과 저장소는 그 에피소드를 기록하지 않으므로 같은 명령으로 이어서 돌 수 있다.

LLM이 읽는 텍스트(지침, 재촉 문구)는 영어로 쓰고 정답 단서를 넣지 않는다. 바꾸면
`instructions_sha256`이 바뀌어 config_id가 달라진다.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from evidencegate.policy import (
    PolicyTrace,
    StopReason,
    TurnRecord,
    call_tool,
    make_trace,
    summarize_result,
)
from evidencegate.skills import Skill
from opsworld.agent_env import AgentEnv, ToolSpec

INSTRUCTIONS = (
    "You are the on-call SRE agent for a production incident in a simulated environment.\n"
    "Investigate with the provided tools, following the skill below.\n"
    "When you have decided, call exactly one disposition tool: rollback_deployment, "
    "scale_service, or observe_only. The disposition ends the incident.\n"
    "Act only through tool calls."
)
NUDGE = (
    "No tool was called. Continue by calling a tool. "
    "Call exactly one disposition tool when you have decided."
)
UNPARSED_TOOL = "<unparsed>"
SEED_RULE = "derive_llm_seed(family, incident_seed, repeat): sha256, first 8 bytes mod 2^31"
_RECURSION_LIMIT = 200


class OllamaError(RuntimeError):
    """정책 행동이 아닌 Ollama 서버·연결 오류. 실행을 멈춘다."""


@dataclass(frozen=True)
class OllamaSettings:
    model: str
    think: bool | str
    temperature: float = 0.7
    num_ctx: int = 8192
    keep_alive: str = "60m"
    max_tool_calls: int = 8
    max_consecutive_text_turns: int = 2
    max_consecutive_format_errors: int = 3
    request_timeout_s: float = 600.0

    @classmethod
    def for_model(cls, model: str, **overrides: Any) -> "OllamaSettings":
        """모델별 think 기본값: gemma 계열은 False, gpt-oss는 "low"(끌 수 없음)."""
        if "think" not in overrides:
            if model.startswith("gemma"):
                overrides["think"] = False
            elif model.startswith("gpt-oss"):
                overrides["think"] = "low"
            else:
                raise ValueError(f"no default think setting for {model!r}; pass think=")
        return cls(model=model, **overrides)


def instructions_sha256() -> str:
    fixed = json.dumps({"instructions": INSTRUCTIONS, "nudge": NUDGE}, sort_keys=True)
    return hashlib.sha256(fixed.encode("utf-8")).hexdigest()


class _LoopState(TypedDict):
    messages: list[dict[str, Any]]
    turns: list[TurnRecord]
    usage: list[dict[str, Any]]
    pending_calls: list[dict[str, Any]]
    consecutive_text: int
    consecutive_format_errors: int
    stop_reason: StopReason | None


class OllamaPolicy:
    def __init__(
        self,
        settings: OllamaSettings,
        client: httpx.Client,
        *,
        model_digest: str,
        quantization: str | None,
        ollama_version: str,
        machine: str,
        chip: str,
    ) -> None:
        self.settings = settings
        self.policy_id = f"ollama:{settings.model}"
        self._client = client
        self._server = {
            "model_digest": model_digest,
            "quantization": quantization,
            "ollama_version": ollama_version,
        }
        self._machine = {"machine": machine, "chip": chip}
        s = settings
        self.config: dict[str, Any] = {
            "type": "ollama",
            "model": s.model,
            **self._server,
            "instructions_sha256": instructions_sha256(),
            "options": {"temperature": s.temperature, "num_ctx": s.num_ctx},
            "think": s.think,
            "stop": {
                "max_tool_calls": s.max_tool_calls,
                "max_consecutive_text_turns": s.max_consecutive_text_turns,
            },
            "format_error_policy": {
                "feedback": "error_as_tool_result",
                "max_consecutive_format_errors": s.max_consecutive_format_errors,
            },
            "seed_rule": SEED_RULE,
        }

    @classmethod
    def connect(
        cls, settings: OllamaSettings, client: httpx.Client, *, machine: str, chip: str
    ) -> "OllamaPolicy":
        """서버에서 Ollama 버전과 모델 digest·양자화를 읽어 설정에 고정한다."""
        version = _get_json(client, "/api/version")["version"]
        models = _get_json(client, "/api/tags").get("models", [])
        entry = next(
            (m for m in models if settings.model in (m.get("name"), m.get("model"))),
            None,
        )
        if entry is None:
            raise OllamaError(f"model {settings.model!r} not found on the Ollama server")
        return cls(
            settings,
            client,
            model_digest=entry["digest"],
            quantization=entry.get("details", {}).get("quantization_level"),
            ollama_version=version,
            machine=machine,
            chip=chip,
        )

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        s = self.settings
        tools = [_ollama_tool(spec) for spec in env.tool_specs()]

        def llm_node(state: _LoopState) -> dict[str, Any]:
            start = time.perf_counter()
            reply = self._chat(state["messages"], tools, llm_seed)
            usage = [*state["usage"], _usage(reply, time.perf_counter() - start)]
            if reply.get("parse_error") is not None:
                error = {"error": f"could not parse tool call: {reply['parse_error']}"}
                return {"usage": usage, "pending_calls": [{"_server_parse_error": error}]}

            message = reply["message"]
            calls = message.get("tool_calls") or []
            if calls:
                assistant = {"role": "assistant", "content": message.get("content", "")}
                assistant["tool_calls"] = calls
                return {
                    "usage": usage,
                    "messages": [*state["messages"], assistant],
                    "pending_calls": list(calls),
                    "consecutive_text": 0,
                }

            consecutive_text = state["consecutive_text"] + 1
            update: dict[str, Any] = {
                "usage": usage,
                "messages": [
                    *state["messages"],
                    {"role": "assistant", "content": message.get("content", "")},
                    {"role": "user", "content": NUDGE},
                ],
                "pending_calls": [],
                "consecutive_text": consecutive_text,
            }
            if consecutive_text >= s.max_consecutive_text_turns:
                update["stop_reason"] = StopReason.NO_TOOL_CALL
            return update

        def tools_node(state: _LoopState) -> dict[str, Any]:
            messages = list(state["messages"])
            turns = list(state["turns"])
            consecutive_errors = state["consecutive_format_errors"]
            stop_reason: StopReason | None = None
            for raw in state["pending_calls"]:
                name, result, turn = _execute(env, raw)
                turns.append(turn)
                messages.append(_tool_message(name, result))
                consecutive_errors = consecutive_errors + 1 if turn.format_error else 0
                if env.closed:
                    stop_reason = StopReason.ACTION
                elif len(turns) >= s.max_tool_calls:
                    stop_reason = StopReason.MAX_STEPS
                elif consecutive_errors >= s.max_consecutive_format_errors:
                    stop_reason = StopReason.ERROR
                if stop_reason is not None:
                    break
            return {
                "messages": messages,
                "turns": turns,
                "pending_calls": [],
                "consecutive_format_errors": consecutive_errors,
                "stop_reason": stop_reason,
            }

        def after_llm(state: _LoopState) -> str:
            if state["stop_reason"] is not None:
                return END
            return "tools" if state["pending_calls"] else "llm"

        def after_tools(state: _LoopState) -> str:
            return END if state["stop_reason"] is not None else "llm"

        graph = StateGraph(_LoopState)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", after_llm)
        graph.add_conditional_edges("tools", after_tools)
        app = graph.compile()

        initial: _LoopState = {
            "messages": [
                {"role": "system", "content": INSTRUCTIONS + "\n\n" + skill.text},
                {"role": "user", "content": alert},
            ],
            "turns": [],
            "usage": [],
            "pending_calls": [],
            "consecutive_text": 0,
            "consecutive_format_errors": 0,
            "stop_reason": None,
        }
        final = app.invoke(initial, config={"recursion_limit": _RECURSION_LIMIT})

        stop_reason = final["stop_reason"]
        assert stop_reason is not None
        return make_trace(
            final["turns"],
            stop_reason,
            llm_config={
                "model": s.model,
                **self._server,
                "temperature": s.temperature,
                "seed": llm_seed,
                "num_ctx": s.num_ctx,
                "think": s.think,
                "keep_alive": s.keep_alive,
                **self._machine,
            },
            usage={
                "calls": final["usage"],
                "llm_calls": len(final["usage"]),
                "wall_s": sum(u["wall_s"] for u in final["usage"]),
            },
        )

    def _chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], seed: int
    ) -> dict[str, Any]:
        s = self.settings
        body = {
            "model": s.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": s.think,
            "keep_alive": s.keep_alive,
            "options": {"temperature": s.temperature, "seed": seed, "num_ctx": s.num_ctx},
        }
        try:
            response = self._client.post("/api/chat", json=body, timeout=s.request_timeout_s)
        except httpx.HTTPError as error:
            raise OllamaError(f"request to Ollama failed: {error}") from error
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            return data
        message = _error_text(response)
        if "pars" in message.lower() and "tool" in message.lower():
            return {"parse_error": message}
        raise OllamaError(f"Ollama /api/chat returned {response.status_code}: {message}")


def _execute(env: AgentEnv, raw: dict[str, Any]) -> tuple[str, dict[str, Any], TurnRecord]:
    """LLM이 낸 도구 호출 하나를 실행한다. 파싱 실패는 환경을 부르지 않고 형식 오류로 남긴다."""
    if "_server_parse_error" in raw:
        result = raw["_server_parse_error"]
        turn = TurnRecord(UNPARSED_TOOL, {}, summarize_result(result), format_error=True)
        return UNPARSED_TOOL, result, turn

    function = raw.get("function") or {}
    name = function.get("name")
    arguments = function.get("arguments", {})
    if not isinstance(name, str) or not name:
        result = {"error": "tool call without a function name"}
        turn = TurnRecord(UNPARSED_TOOL, {"_raw": repr(raw)}, summarize_result(result), True)
        return UNPARSED_TOOL, result, turn
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            result = {"error": "arguments are not valid JSON"}
            turn = TurnRecord(name, {"_raw": arguments}, summarize_result(result), True)
            return name, result, turn
    result, turn = call_tool(env, name, arguments)
    return name, result, turn


def _tool_message(name: str, result: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "tool", "content": json.dumps(result, sort_keys=True)}
    if name != UNPARSED_TOOL:
        message["tool_name"] = name
    return message


def _ollama_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _usage(reply: dict[str, Any], wall_s: float) -> dict[str, Any]:
    return {
        "prompt_eval_count": reply.get("prompt_eval_count"),
        "prompt_eval_cached_count": reply.get("prompt_eval_cached_count"),
        "eval_count": reply.get("eval_count"),
        "total_duration_ns": reply.get("total_duration"),
        "load_duration_ns": reply.get("load_duration"),
        "prompt_eval_duration_ns": reply.get("prompt_eval_duration"),
        "eval_duration_ns": reply.get("eval_duration"),
        "wall_s": wall_s,
    }


def _get_json(client: httpx.Client, path: str) -> dict[str, Any]:
    try:
        response = client.get(path)
    except httpx.HTTPError as error:
        raise OllamaError(f"cannot reach Ollama ({path}): {error}") from error
    if response.status_code != 200:
        raise OllamaError(f"Ollama {path} returned {response.status_code}: {_error_text(response)}")
    data: dict[str, Any] = response.json()
    return data


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text
