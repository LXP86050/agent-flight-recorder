from __future__ import annotations

from agent_flight_recorder.errors import ReplayDivergence
from agent_flight_recorder.hashing import canonical
from agent_flight_recorder.models import Run, Step


class Tape:
    """Serves a recorded run in order.

    The next agent call has to match the next taped step. A mismatch raises
    instead of searching the rest of the tape: silent realignment hides the
    bug you are trying to reproduce. `until` is inclusive. The following call
    goes live and stays live.
    """

    def __init__(self, run: Run, until: int | None) -> None:
        self.source_run_id = run.run_id
        self.until = until
        self.steps = [step for step in run.steps if step.kind != "seal"]
        self.index = 0
        self.live = False

    def next_for(self, kind: str, name: str, payload: object) -> Step | None:
        if self.live:
            return None
        if self.index >= len(self.steps):
            if self.until is not None:
                self.live = True
                return None
            raise ReplayDivergence("replay tape ended but the agent issued another call")
        step = self.steps[self.index]
        if self.until is not None and step.seq > self.until:
            self.live = True
            return None
        if step.kind != kind or step.name != name:
            raise ReplayDivergence(
                f"replay diverged at seq {step.seq}: tape has {step.kind} {step.name}, "
                f"agent issued {kind} {name}",
                seq=step.seq,
            )
        if canonical(step.input) != canonical(payload):
            raise ReplayDivergence(
                f"replay diverged at seq {step.seq}: {kind} {name} inputs differ",
                seq=step.seq,
            )
        self.index += 1
        return step
