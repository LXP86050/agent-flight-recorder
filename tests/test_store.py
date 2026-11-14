import json
from pathlib import Path
from typing import Any

import pytest

from agent_flight_recorder.errors import ChainBroken, RunSealed, SealConflict, StoreBusy
from agent_flight_recorder.hashing import digest, make_step_id
from agent_flight_recorder.store import Store


def _record(
    run_id: str,
    *,
    kind: str = "tool",
    name: str = "echo",
    actor: str = "agent",
    parents: tuple[str, ...] = (),
    payload: dict[str, Any] | None = None,
    output: Any = None,
    lamport: int = 1,
    started_ns: int = 10,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {} if payload is None else payload
    return {
        "step_id": make_step_id(run_id, kind, name, actor, parents, payload),
        "run_id": run_id,
        "kind": kind,
        "name": name,
        "actor": actor,
        "parents": list(parents),
        "lamport": lamport,
        "vc": {actor: lamport},
        "input": payload,
        "output": output,
        "error": error,
        "tokens_in": None,
        "tokens_out": None,
        "started_ns": started_ns,
        "duration_ns": 5,
    }


def test_append_is_idempotent_across_reopen_and_conflict_does_not_grow(tmp_path: Path) -> None:
    with Store(tmp_path) as store:
        store.create_run(run_id="ab12", name="echo", entrypoint=None, created_ns=1)
        first = _record("ab12", payload={"args": {"n": "café"}}, output={"ok": True})
        stored = store.append_step(first)
        assert stored.seq == 0
        assert stored.output == {"ok": True}

    tape = tmp_path / "runs" / "ab12.jsonl"
    size = tape.stat().st_size
    with Store(tmp_path) as store:
        again = dict(first)
        again["started_ns"] = 999
        again["duration_ns"] = 999
        replayed = store.append_step(again)
        assert replayed.step_id == stored.step_id
        assert replayed.started_ns == 10
        assert tape.stat().st_size == size

        conflict = dict(first)
        conflict["output"] = {"ok": False}
        with pytest.raises(SealConflict):
            store.append_step(conflict)
        assert tape.stat().st_size == size

        bogus = dict(first)
        bogus["step_id"] = "0" * 64
        with pytest.raises(ValueError, match="step_id"):
            store.append_step(bogus)


def test_seal_closes_the_run_and_a_missing_seal_is_torn(tmp_path: Path) -> None:
    with Store(tmp_path) as store:
        store.create_run(run_id="cd34", name="echo", entrypoint="demo.py:main", created_ns=5)
        step = store.append_step(_record("cd34"))
        assert store.load("cd34").status == "torn"
        seal = _record(
            "cd34",
            kind="seal",
            name="seal",
            parents=(step.step_id,),
            payload={"status": "ok"},
            lamport=2,
        )
        store.append_step(seal)
        store.append_step(seal)
        with pytest.raises(RunSealed):
            store.append_step(
                _record(
                    "cd34", name="later", parents=(step.step_id,), payload={"args": {}}, lamport=3
                )
            )

    with Store(tmp_path) as store:
        run = store.load("cd")
        assert run.run_id == "cd34"
        assert run.status == "ok"
        assert run.intact
        assert run.entrypoint == "demo.py:main"
        raw = (tmp_path / "runs" / "cd34.jsonl").read_bytes().split(b"\n")
        assert digest(raw[0]) == run.steps[0].prev
        assert run.steps[1].prev == run.steps[0].line_hash


def test_torn_tail_is_truncated_and_a_tampered_line_breaks_the_chain(tmp_path: Path) -> None:
    with Store(tmp_path) as store:
        store.create_run(run_id="ef56", name="echo", entrypoint=None, created_ns=1)
        store.append_step(_record("ef56", payload={"text": "alpha"}))
        store.append_step(
            _record("ef56", name="second", payload={"text": "beta"}, lamport=2, started_ns=20)
        )

    tape = tmp_path / "runs" / "ef56.jsonl"
    intact = tape.read_bytes()
    tape.write_bytes(intact + b'{"not":"finished"')
    with Store(tmp_path) as store:
        run = store.load("ef56")
        assert run.intact
        assert len(run.steps) == 2
        assert tape.read_bytes() == intact

    text = intact.decode()
    tape.write_bytes(text.replace("alpha", "ALPHA", 1).encode())
    with Store(tmp_path) as store:
        run = store.load("ef56")
        assert run.failure is not None and "chain break" in run.failure

    tape.write_bytes(text.replace("beta", "BETA", 1).encode())
    with Store(tmp_path) as store:
        run = store.load("ef56")
        assert not run.intact
        assert run.status == "corrupt"
        assert run.failure is not None and "tip hash" in run.failure
        with pytest.raises(ChainBroken):
            store.append_step(_record("ef56", name="nope", payload={"text": "gamma"}, lamport=3))
    with Store(tmp_path) as store:
        again = store.load("ef56")
        assert again.failure is not None and "tip hash" in again.failure


def test_index_is_rebuilt_from_the_tape(tmp_path: Path) -> None:
    with Store(tmp_path) as store:
        store.create_run(run_id="aa11", name="one", entrypoint=None, created_ns=2)
        store.create_run(run_id="aa22", name="two", entrypoint=None, created_ns=1)
        store.append_step(_record("aa11", output={"n": 1}))
    (tmp_path / "index.sqlite").unlink()
    with Store(tmp_path) as store:
        listed = store.list_runs()
        assert [run.run_id for run in listed] == ["aa11", "aa22"]
        assert store.load("aa11").steps[0].output == {"n": 1}
        with pytest.raises(KeyError, match="matches"):
            store.resolve("aa")


def test_rewriting_the_tip_without_updating_the_chain_is_caught(tmp_path: Path) -> None:
    with Store(tmp_path) as store:
        store.create_run(run_id="tip1", name="echo", entrypoint=None, created_ns=1)
        step = store.append_step(_record("tip1", payload={"text": "keep"}))
        store.append_step(
            _record(
                "tip1",
                kind="seal",
                name="seal",
                parents=(step.step_id,),
                payload={"status": "ok"},
                lamport=2,
            )
        )
    tape = tmp_path / "runs" / "tip1.jsonl"
    text = tape.read_text(encoding="utf-8")
    tape.write_text(text.replace('"status":"ok"', '"status":"zz"', 1), encoding="utf-8")
    with Store(tmp_path) as store:
        run = store.load("tip1")
        assert not run.intact
        assert run.failure is not None and "tip hash" in run.failure


def test_second_store_is_rejected_until_the_lock_is_released(tmp_path: Path) -> None:
    first = Store(tmp_path)
    try:
        with pytest.raises(StoreBusy):
            Store(tmp_path)
    finally:
        first.close()
    with Store(tmp_path) as second:
        assert second.list_runs() == []


def test_header_only_tape_roundtrips_json(tmp_path: Path) -> None:
    with Store(tmp_path) as store:
        store.create_run(
            run_id="0011",
            name="empty",
            entrypoint=None,
            created_ns=1,
            replay_of="abcd",
            replay_until=3,
        )
    line = (tmp_path / "runs" / "0011.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["replay_until"] == 3
    with Store(tmp_path) as store:
        run = store.load("0011")
        assert run.status == "torn"
        assert run.replay_of == "abcd"
        assert run.replay_until == 3
        assert run.steps == ()
