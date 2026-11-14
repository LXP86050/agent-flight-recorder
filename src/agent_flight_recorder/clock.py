from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Clock:
    lamport: int
    vector: tuple[tuple[str, int], ...]

    @staticmethod
    def zero() -> Clock:
        return Clock(0, ())

    def tick(self, actor: str) -> Clock:
        counts = dict(self.vector)
        counts[actor] = counts.get(actor, 0) + 1
        return Clock(self.lamport + 1, tuple(sorted(counts.items())))

    def merge(self, other: Clock) -> Clock:
        counts = dict(self.vector)
        for actor, value in other.vector:
            counts[actor] = max(counts.get(actor, 0), value)
        return Clock(max(self.lamport, other.lamport), tuple(sorted(counts.items())))


def compare(left: Clock, right: Clock) -> str:
    """Return before, after, equal, or concurrent from left's point of view."""

    ahead = False
    behind = False
    counts_left = dict(left.vector)
    counts_right = dict(right.vector)
    for actor in set(counts_left) | set(counts_right):
        a = counts_left.get(actor, 0)
        b = counts_right.get(actor, 0)
        if a > b:
            ahead = True
        elif a < b:
            behind = True
    if ahead and behind:
        return "concurrent"
    if ahead:
        return "after"
    if behind:
        return "before"
    return "equal"
