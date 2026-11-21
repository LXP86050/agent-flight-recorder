from pathlib import Path
from typing import Any

from agent_flight_recorder import Recorder, record_tool
from agent_flight_recorder.adapters import MockLLM
from agent_flight_recorder.diff import diff_runs, summary
from agent_flight_recorder.models import Completion, Run, Step, ToolCall
from agent_flight_recorder.report import diff_html, run_html

flag = {"value": "a"}
reads: list[str] = []


@record_tool
def read_flag() -> dict[str, str]:
    reads.append(flag["value"])
    return {"value": flag["value"], "marker": "<script>"}


def _ask(messages: list[dict[str, Any]], model: str) -> Completion:
    if any(message.get("role") == "tool" for message in messages):
        return Completion("saw it", (), model, 2, 1)
    return Completion(None, (ToolCall("read_flag", {}),), model, 2, 1)


def _run(recorder: Recorder, *, text: str = "go") -> str:
    llm = MockLLM(_ask)
    with recorder.session("read") as session:
        session.rng(1)
        messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
        completion = session.complete(llm, messages=messages, model="m")
        if completion.tool_calls:
            result = read_flag()
            messages.append({"role": "tool", "name": "read_flag", "content": result["value"]})
            completion = session.complete(llm, messages=messages, model="m")
        return session.run_id


def test_diff_reports_output_and_argument_changes(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder:
        flag["value"] = "a"
        first = recorder.store.load(_run(recorder))
        flag["value"] = "b"
        second = recorder.store.load(_run(recorder))
        same = recorder.store.load(_run(recorder))

    identical = diff_runs(second, same)
    assert not identical.diverged
    assert summary(identical) == "no divergence"

    changed = diff_runs(first, second)
    assert changed.first is not None
    assert changed.first.kind == "output"
    assert changed.first.left is not None and changed.first.left.name == "read_flag"
    assert "output differs" in summary(changed)

    page = diff_html(changed)
    assert "&lt;script&gt;" in page
    assert "<script>" not in page
    transcript = run_html(first)
    assert first.run_id in transcript
    assert "&lt;script&gt;" in transcript
    assert "<script>" not in transcript


def _step(seq: int, payload: dict[str, Any]) -> Step:
    return Step(
        step_id=f"{seq:064d}",
        run_id="run",
        seq=seq,
        kind="tool",
        name="refund",
        actor="agent",
        parents=(),
        lamport=seq + 1,
        vc=(("agent", seq + 1),),
        input=payload,
        output={"status": "refunded"},
        error=None,
        tokens_in=None,
        tokens_out=None,
        started_ns=0,
        duration_ns=0,
        prev="0" * 64,
        line_hash="1" * 64,
    )


def _run_of(run_id: str, steps: list[Step]) -> Run:
    return Run(
        run_id=run_id,
        name="refund",
        entrypoint=None,
        created_ns=1,
        replay_of=None,
        replay_until=None,
        status="ok",
        steps=tuple(steps),
        head="h",
        intact=True,
        failure=None,
    )


def test_diff_names_an_argument_missing_on_one_side() -> None:
    left = _run_of(
        "a",
        [
            _step(
                0,
                {"args": {"amount_cents": 4200, "note": None, "order_id": "A184"}},
            )
        ],
    )
    right = _run_of(
        "b",
        [_step(0, {"args": {"amount_cents": 1500, "order_id": "A184"}})],
    )
    diff = diff_runs(left, right)
    assert diff.first is not None
    assert diff.first.kind == "args"
    assert diff.first.detail == (
        "arguments differ (amount_cents: 4200 -> 1500, note: None -> <missing>)"
    )

    swapped = diff_runs(right, left)
    assert swapped.first is not None
    assert swapped.first.detail == (
        "arguments differ (amount_cents: 1500 -> 4200, note: <missing> -> None)"
    )


def test_diff_reports_input_changes_and_steps_present_on_one_side(tmp_path: Path) -> None:
    with Recorder(tmp_path) as recorder:
        with recorder.session("prompt") as session:
            session.complete(
                MockLLM(script=[Completion("one", (), "m", 1, 1)]),
                messages=[{"role": "user", "content": "one"}],
                model="m",
            )
            left_id = session.run_id
        with recorder.session("prompt") as session:
            session.complete(
                MockLLM(script=[Completion("one", (), "m", 1, 1)]),
                messages=[{"role": "user", "content": "two"}],
                model="m",
            )
            right_id = session.run_id
        with recorder.session("short") as session:
            session.rng(1)
            short_id = session.run_id
        with recorder.session("long") as session:
            session.rng(1)
            session.rng(2)
            long_id = session.run_id

        diff = diff_runs(recorder.store.load(left_id), recorder.store.load(right_id))
        extra = diff_runs(recorder.store.load(short_id), recorder.store.load(long_id))
    assert diff.first is not None
    assert diff.first.kind == "args"
    assert "inputs differ" in diff.first.detail
    assert any(row.kind == "right" for row in extra.rows)
