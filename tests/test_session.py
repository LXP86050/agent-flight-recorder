import random
import threading
import time
from pathlib import Path

import pytest

from agent_flight_recorder import Recorder, record_tool
from agent_flight_recorder.adapters import MockLLM
from agent_flight_recorder.clock import Clock, compare
from agent_flight_recorder.models import Completion, ToolCall

calls: list[str] = []


@record_tool
def upper(text: str) -> dict[str, str]:
    calls.append(text)
    return {"text": text.upper()}


@record_tool
def boom(text: str) -> dict[str, str]:
    calls.append("boom")
    raise RuntimeError(f"nope {text}")


@record_tool
def left_branch() -> dict[str, str]:
    barrier.wait(timeout=2)
    return {"side": "left"}


@record_tool
def right_branch() -> dict[str, str]:
    barrier.wait(timeout=2)
    return {"side": "right"}


barrier = threading.Barrier(2)


def test_session_records_model_tool_seed_and_plain_calls(tmp_path: Path) -> None:
    calls.clear()
    assert upper(text="off") == {"text": "OFF"}
    assert calls == ["off"]

    llm = MockLLM(
        script=[
            Completion(None, (ToolCall("upper", {"text": "ab"}),), "mock", 4, 2),
            Completion("done", (), "mock", 6, 1),
        ]
    )
    with Recorder(tmp_path) as recorder:
        with recorder.session("plain", entrypoint="tests:go") as session:
            assert session.rng(3).randrange(10) == random.Random(3).randrange(10)
            completion = session.complete(
                llm,
                messages=[{"role": "user", "content": "hi"}],
                model="mock",
                temperature=0,
                seed=3,
            )
            assert completion.tool_calls[0].name == "upper"
            result = session.call_tool("upper", text="ab")
            assert result == {"text": "AB"}
            final = session.complete(
                llm, messages=[{"role": "user", "content": "hi"}], model="mock"
            )
            assert final.content == "done"
            run_id = session.run_id
        run = recorder.store.load(run_id)
    assert run.status == "ok"
    assert run.entrypoint == "tests:go"
    kinds = [step.kind for step in run.steps]
    assert kinds == ["seed", "llm", "tool", "llm", "seal"]
    assert run.steps[0].input == {"seed": 3}
    assert run.steps[2].input["args"] == {"text": "ab"}
    assert run.steps[2].output == {"text": "AB"}
    assert run.steps[1].tokens_in == 4
    assert run.intact
    assert calls == ["off", "ab"]


def test_live_step_is_timestamped_when_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"wall": 0, "mono": 0}

    def wall() -> int:
        clock["wall"] += 1_000
        return clock["wall"]

    def mono() -> int:
        clock["mono"] += 1_000
        return clock["mono"]

    monkeypatch.setattr(time, "time_ns", wall)
    monkeypatch.setattr(time, "monotonic_ns", mono)
    seen: dict[str, int] = {}

    def probe() -> dict[str, bool]:
        seen["inside"] = time.time_ns()
        return {"ok": True}

    def explode() -> dict[str, bool]:
        seen["inside"] = time.time_ns()
        raise RuntimeError("nope")

    with Recorder(tmp_path) as recorder:
        with recorder.session("time") as session:
            session.invoke_tool("probe", probe, (), {})
        step = recorder.store.load(session.run_id).steps[0]
        assert step.kind == "tool"
        assert step.started_ns < seen["inside"]
        assert step.duration_ns > 0

        with pytest.raises(RuntimeError, match="nope"):
            with recorder.session("time") as session:
                session.invoke_tool("explode", explode, (), {})
        failed = next(
            step for step in recorder.store.load(session.run_id).steps if step.kind == "tool"
        )
        assert failed.started_ns < seen["inside"]


def test_unknown_tool_is_an_error(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder, pytest.raises(KeyError, match="no tool"):
        with recorder.session("missing") as session:
            session.call_tool("not_registered")


def test_tool_exception_is_on_the_tape_and_reraised(tmp_path: Path) -> None:
    calls.clear()
    with Recorder(tmp_path) as recorder:
        with pytest.raises(RuntimeError, match="nope hi"):
            with recorder.session("bad") as session:
                session.rng(1)
                boom(text="hi")
        run = recorder.store.load(session.run_id)
    assert run.status == "error"
    tool = next(step for step in run.steps if step.kind == "tool")
    assert tool.error is not None and tool.error.startswith("RuntimeError")
    assert run.steps[-1].kind == "seal"
    assert run.steps[-1].input == {"status": "error"}


def test_parallel_branches_are_concurrent_and_the_next_step_joins_them(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder:
        with recorder.session("parallel") as session:
            session.rng(1)

            def run_left() -> None:
                block.do("left", left_branch)

            def run_right() -> None:
                block.do("right", right_branch)

            with session.parallel() as block:
                threads = [
                    threading.Thread(target=run_left),
                    threading.Thread(target=run_right),
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2)
            session.rng(2)
            run_id = session.run_id
        run = recorder.store.load(run_id)

    seed, left, right, after = run.steps[0], run.steps[1], run.steps[2], run.steps[3]
    by_name = {step.name: step for step in (left, right)}
    left = by_name["left_branch"]
    right = by_name["right_branch"]
    assert seed.kind == "seed" and after.kind == "seed"
    assert compare(Clock(seed.lamport, seed.vc), Clock(left.lamport, left.vc)) == "before"
    assert compare(Clock(left.lamport, left.vc), Clock(right.lamport, right.vc)) == "concurrent"
    assert compare(Clock(left.lamport, left.vc), Clock(after.lamport, after.vc)) == "before"
    assert compare(Clock(right.lamport, right.vc), Clock(after.lamport, after.vc)) == "before"
    assert left.parents == (seed.step_id,)
    assert right.parents == (seed.step_id,)
    assert set(after.parents) == {left.step_id, right.step_id}
    assert left.actor == "left"
    assert right.actor == "right"


def test_parallel_refuses_a_branch_still_running(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def wait() -> dict[str, bool]:
        started.set()
        assert release.wait(timeout=2)
        return {"ok": True}

    thread: threading.Thread | None = None
    with Recorder(tmp_path) as recorder:
        with pytest.raises(RuntimeError, match="still running"):
            with recorder.session("race") as session:
                with session.parallel() as block:

                    def run() -> None:
                        block.do("worker", wait)

                    thread = threading.Thread(target=run)
                    thread.start()
                    assert started.wait(timeout=2)
        release.set()
        assert thread is not None
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_parallel_rejects_the_session_actor_and_nesting(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder:
        with recorder.session("nest") as session:
            with session.parallel() as block:
                with pytest.raises(ValueError, match="session actor"):
                    block.do("agent", lambda: None)

                def nested() -> None:
                    with session.parallel() as inner:
                        inner.do("child", lambda: None)

                with pytest.raises(RuntimeError, match="nest"):
                    block.do("outer", nested)
