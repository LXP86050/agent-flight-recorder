from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class Completion:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass(frozen=True, slots=True)
class Step:
    step_id: str
    run_id: str
    seq: int
    kind: str
    name: str
    actor: str
    parents: tuple[str, ...]
    lamport: int
    vc: tuple[tuple[str, int], ...]
    input: Any
    output: Any
    error: str | None
    tokens_in: int | None
    tokens_out: int | None
    started_ns: int
    duration_ns: int
    prev: str
    line_hash: str


@dataclass(frozen=True, slots=True)
class Run:
    run_id: str
    name: str
    entrypoint: str | None
    created_ns: int
    replay_of: str | None
    replay_until: int | None
    status: str
    steps: tuple[Step, ...]
    head: str
    intact: bool
    failure: str | None


def completion_to_json(completion: Completion) -> dict[str, Any]:
    return {
        "content": completion.content,
        "tool_calls": [
            {"name": call.name, "arguments": dict(call.arguments)} for call in completion.tool_calls
        ],
        "model": completion.model,
    }


def completion_from_json(
    data: dict[str, Any],
    *,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> Completion:
    calls = tuple(
        ToolCall(str(item["name"]), dict(item["arguments"]))
        for item in data.get("tool_calls") or []
    )
    return Completion(
        content=data.get("content"),
        tool_calls=calls,
        model=str(data.get("model") or ""),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
