from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_flight_recorder.errors import ChainBroken, RunSealed, SealConflict, StoreBusy
from agent_flight_recorder.hashing import (
    GENESIS,
    body_hash,
    canonical,
    digest,
    freeze,
    make_step_id,
)
from agent_flight_recorder.models import Run, Step

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entrypoint TEXT,
    created_ns INTEGER NOT NULL,
    replay_of TEXT,
    replay_until INTEGER,
    status TEXT NOT NULL,
    step_count INTEGER NOT NULL,
    head TEXT NOT NULL,
    nbytes INTEGER NOT NULL,
    intact INTEGER NOT NULL,
    failure TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    lamport INTEGER NOT NULL,
    actor TEXT NOT NULL,
    body_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS steps_by_run ON steps(run_id, seq);
"""


@dataclass
class _Held:
    body_hash: str
    step: Step


@dataclass
class _MemRun:
    run_id: str
    name: str
    entrypoint: str | None
    created_ns: int
    replay_of: str | None
    replay_until: int | None
    status: str
    head: str
    nbytes: int
    intact: bool
    failure: str | None
    sealed: bool
    steps: list[Step] = field(default_factory=list)
    by_id: dict[str, _Held] = field(default_factory=dict)

    def to_run(self) -> Run:
        return Run(
            run_id=self.run_id,
            name=self.name,
            entrypoint=self.entrypoint,
            created_ns=self.created_ns,
            replay_of=self.replay_of,
            replay_until=self.replay_until,
            status=self.status,
            steps=tuple(self.steps),
            head=self.head,
            intact=self.intact,
            failure=self.failure,
        )


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def repair_tape(path: Path) -> None:
    """Drop a torn trailing line left by a crash mid-append.

    A line counts as durable only when it ends in a newline and parses.
    Anything after the last good line is truncated and fsynced.
    """

    if not path.exists():
        return
    data = path.read_bytes()
    if not data:
        return
    parts = data.split(b"\n")[:-1]
    good: list[bytes] = []
    for part in parts:
        if not part:
            break
        try:
            json.loads(part)
        except json.JSONDecodeError:
            break
        good.append(part)
    rebuilt = b"".join(part + b"\n" for part in good)
    if rebuilt == data:
        return
    path.write_bytes(rebuilt)
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _step_from(obj: dict[str, Any], line_hash: str) -> Step:
    vector = obj["vc"]
    pairs = tuple(sorted((str(key), int(value)) for key, value in vector.items()))
    parents = tuple(str(parent) for parent in obj["parents"])
    return Step(
        step_id=str(obj["step_id"]),
        run_id=str(obj["run_id"]),
        seq=int(obj["seq"]),
        kind=str(obj["kind"]),
        name=str(obj["name"]),
        actor=str(obj["actor"]),
        parents=parents,
        lamport=int(obj["lamport"]),
        vc=pairs,
        input=obj["input"],
        output=obj["output"],
        error=None if obj["error"] is None else str(obj["error"]),
        tokens_in=None if obj["tokens_in"] is None else int(obj["tokens_in"]),
        tokens_out=None if obj["tokens_out"] is None else int(obj["tokens_out"]),
        started_ns=int(obj["started_ns"]),
        duration_ns=int(obj["duration_ns"]),
        prev=str(obj["prev"]),
        line_hash=line_hash,
    )


def parse_tape(
    data: bytes,
) -> tuple[dict[str, Any] | None, list[tuple[dict[str, Any], str]], str | None]:
    """Walk a tape. A chain break keeps the prefix and reports where it broke."""

    if not data:
        return None, [], "empty tape"
    prev = GENESIS
    header: dict[str, Any] | None = None
    steps: list[tuple[dict[str, Any], str]] = []
    failure: str | None = None
    lines = data.split(b"\n")[:-1]
    for index, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            failure = f"line {index} is not json"
            break
        if not isinstance(obj, dict):
            failure = f"line {index} is not an object"
            break
        if obj.get("prev") != prev:
            failure = f"chain break at line {index}"
            break
        if obj.get("v") != 1:
            failure = f"unsupported tape version {obj.get('v')!r} at line {index}"
            break
        line_hash = digest(line)
        kind = obj.get("type")
        if kind == "header" and header is None:
            header = obj
        elif kind == "step":
            if obj.get("seq") != len(steps):
                failure = f"seq gap at line {index}"
                break
            steps.append((obj, line_hash))
        else:
            failure = f"line {index} has an unknown type"
            break
        prev = line_hash
    if header is None and failure is None:
        failure = "tape has no header"
    return header, steps, failure


class Store:
    """One writer for a directory of tapes.

    The JSONL files are the record. SQLite is rebuilt from them on open, so a
    crash between the fsync and the index update cannot lose a step.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock_fd = -1
        self._conn: sqlite3.Connection | None = None
        self._mu = threading.RLock()
        self._runs: dict[str, _MemRun] = {}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "runs").mkdir(exist_ok=True)
            self._lock_fd = os.open(self.directory / "lock", os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StoreBusy(str(self.directory)) from None
            self._conn = sqlite3.connect(self.directory / "index.sqlite", check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._load()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            conn.close()
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _path(self, run_id: str) -> Path:
        return self.directory / "runs" / f"{run_id}.jsonl"

    def _load(self) -> None:
        assert self._conn is not None
        # The index remembers the tip we last fsynced. A later edit of the last
        # line still chains (nothing follows it), so the remembered tip is the
        # witness. A crash that only appended a valid line keeps the old tip in
        # the chain; a missing index has no witness and the tape is trusted.
        old_heads = {
            str(row[0]): str(row[1]) for row in self._conn.execute("SELECT run_id, head FROM runs")
        }
        self._runs.clear()
        self._conn.execute("DELETE FROM steps")
        self._conn.execute("DELETE FROM runs")
        for path in sorted((self.directory / "runs").glob("*.jsonl")):
            repair_tape(path)
            data = path.read_bytes()
            if not data:
                continue
            header, raw_steps, failure = parse_tape(data)
            if header is None:
                continue
            run_id = str(header["run_id"])
            steps: list[Step] = []
            by_id: dict[str, _Held] = {}
            header_line = data.split(b"\n", 1)[0]
            head = digest(header_line)
            seen = {head}
            for obj, line_hash in raw_steps:
                step = _step_from(obj, line_hash)
                steps.append(step)
                by_id[step.step_id] = _Held(body_hash(obj), step)
                head = line_hash
                seen.add(head)
            witnessed = old_heads.get(run_id)
            if witnessed is not None and witnessed not in seen:
                # Keep the previous tip in the index so the next open still
                # disagrees with the rewritten line.
                failure = failure or "tip hash does not match the index"
                head = witnessed
            if path.stem != run_id:
                failure = failure or "run_id does not match the filename"
            intact = failure is None
            sealed = bool(steps) and steps[-1].kind == "seal" and intact
            if not intact:
                status = "corrupt"
            elif sealed:
                status_value = (
                    steps[-1].input.get("status") if isinstance(steps[-1].input, dict) else "ok"
                )
                status = str(status_value)
            else:
                status = "torn"
            mem = _MemRun(
                run_id=run_id,
                name=str(header["name"]),
                entrypoint=None if header.get("entrypoint") is None else str(header["entrypoint"]),
                created_ns=int(header["created_ns"]),
                replay_of=None if header.get("replay_of") is None else str(header["replay_of"]),
                replay_until=None
                if header.get("replay_until") is None
                else int(header["replay_until"]),
                status=status,
                head=head,
                nbytes=path.stat().st_size,
                intact=intact,
                failure=failure,
                sealed=sealed,
                steps=steps,
                by_id=by_id,
            )
            self._runs[run_id] = mem
            self._index_run(mem)
        self._conn.commit()

    def _index_run(self, mem: _MemRun) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, name, entrypoint, created_ns, replay_of, replay_until,
                status, step_count, head, nbytes, intact, failure
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem.run_id,
                mem.name,
                mem.entrypoint,
                mem.created_ns,
                mem.replay_of,
                mem.replay_until,
                mem.status,
                len(mem.steps),
                mem.head,
                mem.nbytes,
                1 if mem.intact else 0,
                mem.failure,
            ),
        )
        self._conn.executemany(
            """
            INSERT INTO steps (step_id, run_id, seq, kind, name, lamport, actor, body_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    held.step.step_id,
                    mem.run_id,
                    held.step.seq,
                    held.step.kind,
                    held.step.name,
                    held.step.lamport,
                    held.step.actor,
                    held.body_hash,
                )
                for held in mem.by_id.values()
            ],
        )

    def create_run(
        self,
        *,
        run_id: str,
        name: str,
        entrypoint: str | None,
        created_ns: int,
        replay_of: str | None = None,
        replay_until: int | None = None,
    ) -> None:
        with self._mu:
            path = self._path(run_id)
            if path.exists() or run_id in self._runs:
                raise FileExistsError(run_id)
            header = {
                "v": 1,
                "type": "header",
                "run_id": run_id,
                "name": name,
                "entrypoint": entrypoint,
                "created_ns": created_ns,
                "replay_of": replay_of,
                "replay_until": replay_until,
                "prev": GENESIS,
            }
            raw = canonical(header).encode()
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, raw + b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(self.directory / "runs")
            mem = _MemRun(
                run_id=run_id,
                name=name,
                entrypoint=entrypoint,
                created_ns=created_ns,
                replay_of=replay_of,
                replay_until=replay_until,
                status="torn",
                head=digest(raw),
                nbytes=path.stat().st_size,
                intact=True,
                failure=None,
                sealed=False,
            )
            self._runs[run_id] = mem
            assert self._conn is not None
            self._index_run(mem)
            self._conn.commit()

    def append_step(self, record: dict[str, Any]) -> Step:
        with self._mu:
            record = dict(record)
            record["input"] = freeze(record["input"])
            record["output"] = None if record["output"] is None else freeze(record["output"])
            record["parents"] = sorted(str(parent) for parent in record["parents"])
            expected = make_step_id(
                str(record["run_id"]),
                str(record["kind"]),
                str(record["name"]),
                str(record["actor"]),
                record["parents"],
                record["input"],
            )
            if expected != record["step_id"]:
                raise ValueError("step_id does not match the logical body")
            run_id = str(record["run_id"])
            try:
                mem = self._runs[run_id]
            except KeyError:
                raise KeyError(f"no run {run_id}") from None
            if not mem.intact:
                raise ChainBroken(mem.failure or "tape is not intact")
            hashed = body_hash(record)
            existing = mem.by_id.get(str(record["step_id"]))
            if existing is not None:
                if existing.body_hash != hashed:
                    raise SealConflict(str(record["step_id"]))
                return existing.step
            if mem.sealed:
                raise RunSealed(run_id)
            full = {
                "v": 1,
                "type": "step",
                "step_id": record["step_id"],
                "run_id": run_id,
                "seq": len(mem.steps),
                "kind": record["kind"],
                "name": record["name"],
                "actor": record["actor"],
                "parents": record["parents"],
                "lamport": record["lamport"],
                "vc": record["vc"],
                "input": record["input"],
                "output": record["output"],
                "error": record["error"],
                "tokens_in": record["tokens_in"],
                "tokens_out": record["tokens_out"],
                "started_ns": record["started_ns"],
                "duration_ns": record["duration_ns"],
                "prev": mem.head,
            }
            raw = canonical(full).encode()
            path = self._path(run_id)
            fd = os.open(path, os.O_APPEND | os.O_WRONLY)
            try:
                view = raw + b"\n"
                written = 0
                while written < len(view):
                    written += os.write(fd, view[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
            step = _step_from(full, digest(raw))
            mem.steps.append(step)
            mem.by_id[step.step_id] = _Held(hashed, step)
            mem.head = step.line_hash
            mem.nbytes = path.stat().st_size
            if step.kind == "seal":
                mem.sealed = True
                status = step.input.get("status") if isinstance(step.input, dict) else "ok"
                mem.status = str(status)
            assert self._conn is not None
            self._conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._index_run(mem)
            self._conn.commit()
            return step

    def resolve(self, prefix: str) -> str:
        with self._mu:
            hits = [run_id for run_id in self._runs if run_id.startswith(prefix)]
        if not hits:
            raise KeyError(f"no run matches {prefix!r}")
        if len(hits) > 1:
            listed = ", ".join(sorted(hits))
            raise KeyError(f"prefix {prefix!r} matches {listed}")
        return hits[0]

    def load(self, run_id: str) -> Run:
        with self._mu:
            if run_id not in self._runs:
                run_id = self.resolve(run_id)
            return self._runs[run_id].to_run()

    def list_runs(self) -> list[Run]:
        with self._mu:
            runs = [mem.to_run() for mem in self._runs.values()]
        runs.sort(key=lambda run: run.created_ns, reverse=True)
        return runs
