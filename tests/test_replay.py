from pathlib import Path
from typing import Any

import pytest

from agent_flight_recorder import RecordedFailure, Recorder, ReplayDivergence, record_tool
from agent_flight_recorder.adapters import MockLLM
from agent_flight_recorder.hashing import canonical
from agent_flight_recorder.models import Completion, Run, ToolCall

seen: list[int] = []


@record_tool
def add(n: int) -> dict[str, int]:
    seen.append(n)
    return {"n": n}


@record_tool
def fail(n: int) -> dict[str, int]:
    seen.append(n)
    raise RuntimeError("bad tool")


def _handle(messages: list[dict[str, Any]], model: str) -> Completion:
    done = [message for message in messages if message.get("role") == "tool"]
    if len(done) >= 2:
        return Completion("done", (), model, 3, 1)
    n = 1 if not done else 2
    return Completion(None, (ToolCall("add", {"n": n}),), model, 3, 1)


def _live(messages: list[dict[str, Any]], model: str) -> Completion:
    done = [message for message in messages if message.get("role") == "tool"]
    if len(done) >= 2:
        return Completion("fixed", (), model, 3, 1)
    return Completion(None, (ToolCall("add", {"n": 9}),), model, 3, 1)


def _agent(session: Any, llm: MockLLM) -> str:
    session.rng(4)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for _ in range(4):
        completion = session.complete(llm, messages=messages, model="m", temperature=0)
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
            result = add(**call.arguments)
            messages.append({"role": "tool", "name": call.name, "content": canonical(result)})
    raise RuntimeError("budget")


def _logical(run: Run) -> list[tuple[str, str, str, str, str | None]]:
    return [
        (step.kind, step.name, canonical(step.input), canonical(step.output), step.error)
        for step in run.steps
        if step.kind != "seal"
    ]


def test_replay_matches_and_does_not_call_out(tmp_path: Path) -> None:
    seen.clear()
    with Recorder(tmp_path) as recorder:
        with recorder.session("add") as session:
            assert _agent(session, MockLLM(_handle)) == "done"
            source_id = session.run_id
        source = recorder.store.load(source_id)
        assert seen == [1, 2]

        seen.clear()
        live = MockLLM(_handle)
        with recorder.replay(source_id) as session:
            assert _agent(session, live) == "done"
            first = session.run_id
        assert live.calls == []
        assert seen == []
        assert _logical(recorder.store.load(first)) == _logical(source)

        seen.clear()
        with recorder.replay(source_id) as session:
            _agent(session, MockLLM(_handle))
            second = session.run_id
        assert _logical(recorder.store.load(second)) == _logical(recorder.store.load(first))
        assert seen == []


def test_replay_until_then_live(tmp_path: Path) -> None:
    seen.clear()
    with Recorder(tmp_path) as recorder:
        with recorder.session("add") as session:
            _agent(session, MockLLM(_handle))
            source_id = session.run_id
        source = recorder.store.load(source_id)
        until = next(
            step.seq
            for step in source.steps
            if step.kind == "tool" and step.input["args"]["n"] == 1
        )
        seen.clear()
        with recorder.replay(source_id, until=until) as session:
            text = _agent(session, MockLLM(_live))
            pinned = session.run_id
        assert text == "fixed"
        assert seen == [9]
        pinned_run = recorder.store.load(pinned)
        assert pinned_run.replay_until == until
        prefix = [step for step in source.steps if step.seq <= until and step.kind != "seal"]
        pinned_prefix = pinned_run.steps[: len(prefix)]
        assert _logical_steps(pinned_prefix) == _logical_steps(prefix)
        assert any(
            step.kind == "tool" and step.input["args"]["n"] == 9 for step in pinned_run.steps
        )


def test_replay_divergence_names_the_step(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder:
        with recorder.session("add") as session:
            _agent(session, MockLLM(_handle))
            source_id = session.run_id

        def drifted(session: Any, llm: MockLLM) -> None:
            session.rng(4)
            session.complete(
                llm, messages=[{"role": "user", "content": "stop"}], model="m", temperature=0
            )

        with pytest.raises(ReplayDivergence, match="inputs differ") as caught:
            with recorder.replay(source_id) as session:
                drifted(session, MockLLM(_handle))
        assert caught.value.seq == 1


def test_replay_reports_a_different_kind_and_a_tape_that_has_ended(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder:
        with recorder.session("add") as session:
            session.rng(1)
            source_id = session.run_id

        def extra(session: Any) -> None:
            session.rng(1)
            session.rng(2)

        with pytest.raises(ReplayDivergence, match="tape ended"):
            with recorder.replay(source_id) as session:
                extra(session)

        with recorder.replay(source_id, until=0) as session:
            session.rng(1)
            session.rng(2)
            session.rng(3)
            assert session.live

        def wrong_kind(session: Any) -> None:
            session.rng(4)
            add(n=1)

        with recorder.session("add") as session:
            _agent(session, MockLLM(_handle))
            longer = session.run_id
        with pytest.raises(ReplayDivergence, match="tape has llm"):
            with recorder.replay(longer) as session:
                wrong_kind(session)


def test_replay_serves_a_recorded_failure_without_running_the_tool(tmp_path: Path) -> None:
    seen.clear()

    def handle(messages: list[dict[str, Any]], model: str) -> Completion:
        return Completion(None, (ToolCall("fail", {"n": 4}),), model, 1, 1)

    def agent(session: Any, llm: MockLLM) -> None:
        session.rng(1)
        completion = session.complete(llm, messages=[{"role": "user", "content": "go"}], model="m")
        call = completion.tool_calls[0]
        fail(**call.arguments)

    with Recorder(tmp_path) as recorder:
        with pytest.raises(RuntimeError, match="bad tool"):
            with recorder.session("fail") as session:
                agent(session, MockLLM(handle))
        source_id = session.run_id
        assert seen == [4]
        seen.clear()
        with pytest.raises(RecordedFailure, match="bad tool"):
            with recorder.replay(source_id) as session:
                agent(session, MockLLM(handle))
        assert seen == []


def _logical_steps(steps: list[Any]) -> list[tuple[str, str, str, str]]:
    return [
        (step.kind, step.name, canonical(step.input), canonical(step.output))
        for step in steps
        if step.kind != "seal"
    ]
