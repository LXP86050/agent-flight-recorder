"""A small refund agent with one bad policy, used by the demo and by `afr replay`.

The model only decides which tool to call. The agent fills in the amount. The
bug is that it refunds the whole order instead of the line the customer named.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent_flight_recorder import Completion, MockLLM, Recorder, Session, ToolCall, record_tool
from agent_flight_recorder.hashing import canonical

ORDER_ID = "A184"
LINE_CENTS = 1500
ORDER_CENTS = 4200
MODEL = "support-mock"
ENTRYPOINT = "examples/refund_agent.py:main"
SYSTEM = "Look up the order, then request a refund for the line the customer asked about."
USER = "Please refund the keyboard on order A184. Do not refund the rest of the order."

# Tool bodies that actually ran. Replay should leave this alone.
executed: list[str] = []


@record_tool
def search_order(order_id: str) -> dict[str, Any]:
    executed.append("search_order")
    if order_id != ORDER_ID:
        raise KeyError(order_id)
    return {
        "order_id": order_id,
        "sku": "KBD-01",
        "line_total_cents": LINE_CENTS,
        "order_total_cents": ORDER_CENTS,
    }


@record_tool
def refund(order_id: str, amount_cents: int, idempotency_key: str) -> dict[str, Any]:
    executed.append("refund")
    return {
        "status": "refunded",
        "order_id": order_id,
        "amount_cents": amount_cents,
        "idempotency_key": idempotency_key,
    }


def refund_cents(order: dict[str, Any], *, bug: bool) -> int:
    # The model named the line. This returns the whole order anyway.
    if bug:
        return int(order["order_total_cents"])
    return int(order["line_total_cents"])


def make_llm() -> MockLLM:
    def handle(messages: list[dict[str, Any]], model: str) -> Completion:
        tool_msgs = [message for message in messages if message.get("role") == "tool"]
        if not tool_msgs:
            calls = (ToolCall("search_order", {"order_id": ORDER_ID}),)
            content = None
            rendered = "search_order"
        elif len(tool_msgs) == 1:
            calls = (ToolCall("refund", {"order_id": ORDER_ID}),)
            content = None
            rendered = "refund"
        else:
            result = json.loads(str(tool_msgs[-1]["content"]))
            calls = ()
            content = f"Refunded {result['amount_cents']} cents on {ORDER_ID}."
            rendered = content
        tokens_in = max(1, len(canonical(messages)) // 4)
        tokens_out = max(1, len(rendered) // 4)
        return Completion(
            content=content,
            tool_calls=calls,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    return MockLLM(handle)


def run_agent(session: Session, llm: MockLLM, *, bug: bool) -> str:
    rng = session.rng(17)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    for _ in range(6):
        completion = session.complete(
            llm,
            messages=messages,
            model=MODEL,
            temperature=0,
            seed=17,
        )
        messages.append(
            {
                "role": "assistant",
                "content": completion.content,
                "tool_calls": [
                    {"name": call.name, "arguments": dict(call.arguments)}
                    for call in completion.tool_calls
                ],
            }
        )
        if not completion.tool_calls:
            return completion.content or ""
        for call in completion.tool_calls:
            args = dict(call.arguments)
            if call.name == "refund":
                args["amount_cents"] = refund_cents(_last_order(messages), bug=bug)
                args["idempotency_key"] = f"rf-{rng.randrange(1_000_000):06d}"
            if call.name == "search_order":
                result = search_order(**args)
            elif call.name == "refund":
                result = refund(**args)
            else:
                raise RuntimeError(f"unknown tool {call.name}")
            messages.append({"role": "tool", "name": call.name, "content": canonical(result)})
    raise RuntimeError("agent exceeded its step budget")


def main(recorder: Recorder) -> None:
    """Entrypoint stored on the tape. AFR_MODE=fixed selects the corrected amount."""

    bug = os.environ.get("AFR_MODE", "buggy") != "fixed"
    with recorder.session("refund-agent", entrypoint=ENTRYPOINT) as session:
        run_agent(session, make_llm(), bug=bug)


def _last_order(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == "search_order":
            loaded = json.loads(str(message["content"]))
            if isinstance(loaded, dict):
                return loaded
    raise RuntimeError("refund requested before the order was loaded")
