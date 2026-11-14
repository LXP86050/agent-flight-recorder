from __future__ import annotations


class SealConflict(Exception):
    """A step id is already on the tape with a different body."""

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        super().__init__(f"step {step_id[:12]} is already on the tape with a different body")


class RunSealed(Exception):
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run {run_id} is sealed")


class ChainBroken(Exception):
    pass


class ReplayDivergence(Exception):
    def __init__(self, message: str, *, seq: int | None = None) -> None:
        self.seq = seq
        super().__init__(message)


class RecordedFailure(Exception):
    """Raised when replay serves a step that failed during the original run."""


class StoreBusy(Exception):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"another recorder holds {path}")
