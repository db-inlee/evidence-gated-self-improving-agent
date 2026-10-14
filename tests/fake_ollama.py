"""테스트용 가짜 Ollama 서버(httpx MockTransport). 실제 Ollama를 부르지 않는다."""

import json
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

MODEL = "gemma4:12b"
DIGEST = "sha256:0123456789abcdef"

ChatResponder = Callable[[dict[str, Any]], dict[str, Any] | httpx.Response]


def tool_call(name: str, args: dict[str, Any] | str | None = None) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": {} if args is None else args}}


def reply_tools(*calls: dict[str, Any], cached: int | None = 50) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "message": {"role": "assistant", "content": "", "tool_calls": list(calls)},
        "done": True,
        "prompt_eval_count": 120,
        "eval_count": 15,
        "total_duration": 2_000_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_duration": 500_000_000,
        "eval_duration": 1_000_000_000,
    }
    if cached is not None:
        body["prompt_eval_cached_count"] = cached
    return body


def reply_text(content: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 100,
        "eval_count": 20,
    }


class FakeOllama:
    """정해진 chat 응답을 순서대로 돌려주고 받은 요청을 기록한다."""

    def __init__(
        self,
        chat: Sequence[dict[str, Any] | httpx.Response] | ChatResponder,
        models: list[dict[str, Any]] | None = None,
        version: str = "0.33.3",
        delay_s: float = 0.0,
        serialize: bool = False,
    ) -> None:
        """delay_s: chat 요청마다 지연. serialize: 요청을 한 번에 하나씩만 처리(병렬 꺼진 서버)."""
        self.delay_s = delay_s
        self.serialize = serialize
        self._lock = threading.Lock()
        self.responder = chat if callable(chat) else None
        self.chat_replies = [] if callable(chat) else list(chat)
        self.chat_requests: list[dict[str, Any]] = []
        self.models = models if models is not None else [model_entry(MODEL, DIGEST)]
        self.version = version

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": self.models})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            if self.serialize:
                with self._lock:
                    time.sleep(self.delay_s)
            elif self.delay_s:
                time.sleep(self.delay_s)
            self.chat_requests.append(body)
            if "tools" not in body:  # 병렬 탐침 같은 도구 없는 요청
                return httpx.Response(200, json=reply_text("ok"))
            if self.responder is not None:
                reply = self.responder(body)
                return (
                    reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)
                )
            if not self.chat_replies:
                return httpx.Response(200, json=reply_text("(no scripted reply)"))
            reply = self.chat_replies.pop(0)
            return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)
        return httpx.Response(404, json={"error": "not found"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url="http://ollama")


def model_entry(name: str, digest: str, quant: str = "Q4_K_M") -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "digest": digest,
        "details": {"quantization_level": quant, "parameter_size": "12B"},
    }
