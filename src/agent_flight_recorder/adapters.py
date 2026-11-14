from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from agent_flight_recorder.models import Completion, ToolCall


class LLM(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


class MockLLM:
    """Scripted or callback model. Tests and the demo use this; it never hits the network."""

    def __init__(
        self,
        handler: Callable[[list[dict[str, Any]], str], Completion] | None = None,
        *,
        script: list[Completion] | None = None,
    ) -> None:
        if handler is None and script is None:
            raise ValueError("mock llm needs a handler or a script")
        self._handler = handler
        self._script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "tools": tools,
            }
        )
        if self._handler is not None:
            return self._handler(messages, model)
        if not self._script:
            raise RuntimeError("mock llm script exhausted")
        return self._script.pop(0)


class OpenAICompatLLM:
    """Chat completions against any server that speaks the OpenAI request shape.

    No streaming. Tool arguments are parsed as JSON. The recorder, not this
    class, decides what gets written to the tape.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        if not self.api_key:
            raise RuntimeError("no API key; set OPENAI_API_KEY or pass api_key")
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
        if tools is not None:
            body["tools"] = tools
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"llm http {exc.code}: {detail}") from exc
        return _parse_completion(payload, fallback_model=model)


def _parse_completion(payload: dict[str, Any], *, fallback_model: str) -> Completion:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("llm response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("llm response is missing the message")
    calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"tool arguments are not json: {raw_args[:120]}") from exc
        elif isinstance(raw_args, dict):
            parsed = raw_args
        else:
            raise RuntimeError("tool arguments are not an object")
        if not isinstance(parsed, dict):
            raise RuntimeError("tool arguments are not an object")
        calls.append(ToolCall(str(function.get("name") or ""), parsed))
    content = message.get("content")
    if content is not None:
        content = str(content)
    raw_usage = payload.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    tokens_in = usage.get("prompt_tokens")
    tokens_out = usage.get("completion_tokens")
    model = payload.get("model") or fallback_model
    return Completion(
        content=content,
        tool_calls=tuple(calls),
        model=str(model),
        tokens_in=None if tokens_in is None else int(tokens_in),
        tokens_out=None if tokens_out is None else int(tokens_out),
    )
