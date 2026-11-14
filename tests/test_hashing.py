import pytest

from agent_flight_recorder.hashing import canonical, freeze, make_step_id


def test_canonical_sorts_keys_and_keeps_unicode() -> None:
    assert canonical({"b": 1, "a": {"d": True, "c": None}}) == '{"a":{"c":null,"d":true},"b":1}'
    assert canonical({"n": "café"}) == '{"n":"café"}'


def test_freeze_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        freeze(float("nan"))
    with pytest.raises(TypeError):
        freeze({1, 2})


def test_step_id_depends_on_the_call_not_parent_order() -> None:
    left = make_step_id("run", "tool", "add", "agent", ("b", "a"), {"args": {"n": 1}})
    right = make_step_id("run", "tool", "add", "agent", ("a", "b"), {"args": {"n": 1}})
    changed = make_step_id("run", "tool", "add", "agent", ("a", "b"), {"args": {"n": 2}})
    assert left == right
    assert left != changed
