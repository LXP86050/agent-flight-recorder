"""Record a bad refund, replay it, apply the fix from the lookup onward, and diff.

Run from the repo root, after `pip install -e .`:

    python examples/demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_flight_recorder import Recorder
from agent_flight_recorder.diff import diff_runs
from agent_flight_recorder.hashing import canonical
from agent_flight_recorder.models import Run
from refund_agent import (
    ENTRYPOINT,
    LINE_CENTS,
    ORDER_CENTS,
    executed,
    make_llm,
    run_agent,
)


@dataclass
class DemoResult:
    directory: Path
    buggy: str
    replay: str
    fixed: str
    pinned: str
    until: int
    buggy_amount: int
    fixed_amount: int
    pinned_amount: int


def run_demo(directory: str | Path) -> DemoResult:
    directory = Path(directory)
    executed.clear()
    with Recorder(directory) as recorder:
        with recorder.session("refund-agent", entrypoint=ENTRYPOINT) as session:
            run_agent(session, make_llm(), bug=True)
            buggy = session.run_id
        buggy_run = recorder.store.load(buggy)
        buggy_amount = _amount(buggy_run)
        if buggy_amount != ORDER_CENTS:
            raise SystemExit(f"buggy run refunded {buggy_amount}, expected {ORDER_CENTS}")
        if executed != ["search_order", "refund"]:
            raise SystemExit(f"buggy run tool calls: {executed}")

        executed.clear()
        replay_llm = make_llm()
        with recorder.replay(buggy, entrypoint=ENTRYPOINT) as session:
            run_agent(session, replay_llm, bug=True)
            replay_id = session.run_id
        replay_run = recorder.store.load(replay_id)
        if replay_llm.calls:
            raise SystemExit("full replay called the model")
        if executed:
            raise SystemExit(f"full replay executed tools: {executed}")
        if _logical(replay_run) != _logical(buggy_run):
            raise SystemExit("full replay does not match the buggy tape")
        if replay_run.replay_of != buggy:
            raise SystemExit("replay run is not linked to its source")

        executed.clear()
        with recorder.session("refund-agent", entrypoint=ENTRYPOINT) as session:
            run_agent(session, make_llm(), bug=False)
            fixed = session.run_id
        fixed_run = recorder.store.load(fixed)
        fixed_amount = _amount(fixed_run)
        if fixed_amount != LINE_CENTS:
            raise SystemExit(f"fixed run refunded {fixed_amount}, expected {LINE_CENTS}")

        until = _seq(buggy_run, "tool", "search_order")
        executed.clear()
        live_llm = make_llm()
        with recorder.replay(buggy, until=until, entrypoint=ENTRYPOINT) as session:
            run_agent(session, live_llm, bug=False)
            pinned = session.run_id
        pinned_run = recorder.store.load(pinned)
        pinned_amount = _amount(pinned_run)
        if pinned_amount != LINE_CENTS:
            raise SystemExit(f"pinned replay refunded {pinned_amount}, expected {LINE_CENTS}")
        if executed != ["refund"]:
            raise SystemExit(f"pinned replay should rerun only the refund, saw {executed}")
        if len(live_llm.calls) != 2:
            raise SystemExit(
                f"pinned replay made {len(live_llm.calls)} live model calls, expected 2"
            )

        buggy_fixed = diff_runs(buggy_run, fixed_run)
        first = buggy_fixed.first
        if (
            first is None
            or first.kind != "args"
            or first.left is None
            or first.left.name != "refund"
        ):
            raise SystemExit(f"expected the refund arguments to diverge first, saw {first}")
        pinned_diff = diff_runs(buggy_run, pinned_run)
        pinned_first = pinned_diff.first
        if (
            pinned_first is None
            or pinned_first.kind != "args"
            or pinned_first.left is None
            or pinned_first.left.name != "refund"
        ):
            raise SystemExit(f"pinned replay diverged in the wrong place: {pinned_first}")

    return DemoResult(
        directory=directory,
        buggy=buggy,
        replay=replay_id,
        fixed=fixed,
        pinned=pinned,
        until=until,
        buggy_amount=buggy_amount,
        fixed_amount=fixed_amount,
        pinned_amount=pinned_amount,
    )


def _logical(run: Run) -> list[tuple[str, str, str, str, str | None]]:
    return [
        (step.kind, step.name, canonical(step.input), canonical(step.output), step.error)
        for step in run.steps
        if step.kind != "seal"
    ]


def _amount(run: Run) -> int:
    for step in run.steps:
        if step.kind == "tool" and step.name == "refund" and isinstance(step.input, dict):
            return int(step.input["args"]["amount_cents"])
    raise RuntimeError("no refund step")


def _seq(run: Run, kind: str, name: str) -> int:
    for step in run.steps:
        if step.kind == kind and step.name == name:
            return step.seq
    raise RuntimeError(f"no {kind} {name}")


def main() -> None:
    result = run_demo(Path(".afr"))
    from agent_flight_recorder.cli import main as afr

    print(f"buggy   {result.buggy}  refund {result.buggy_amount}")
    print(f"replay  {result.replay}  logical match, model and tools not called")
    print(f"fixed   {result.fixed}  refund {result.fixed_amount}")
    print(f"pinned  {result.pinned}  replay through seq {result.until}, then the fix")
    print()
    afr(["--dir", ".afr", "diff", result.buggy, result.fixed])


if __name__ == "__main__":
    main()
