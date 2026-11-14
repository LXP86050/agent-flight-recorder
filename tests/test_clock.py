from agent_flight_recorder.clock import Clock, compare


def test_tick_and_merge() -> None:
    base = Clock.zero().tick("agent")
    left = base.tick("left")
    right = base.tick("right")
    assert compare(base, left) == "before"
    assert compare(left, base) == "after"
    assert compare(left, right) == "concurrent"
    assert compare(left, left) == "equal"
    joined = left.merge(right).tick("agent")
    assert compare(left, joined) == "before"
    assert compare(right, joined) == "before"
    assert joined.lamport == left.lamport + 1
