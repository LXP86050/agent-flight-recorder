from __future__ import annotations

import inspect
import random
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from agent_flight_recorder.adapters import LLM
from agent_flight_recorder.clock import Clock
from agent_flight_recorder.context import reset_session, set_session
from agent_flight_recorder.errors import ChainBroken, RecordedFailure
from agent_flight_recorder.hashing import freeze, make_step_id
from agent_flight_recorder.models import (
    Completion,
    Run,
    Step,
    completion_from_json,
    completion_to_json,
)
from agent_flight_recorder.replay import Tape
from agent_flight_recorder.store import Store
from agent_flight_recorder.tools import lookup

T = TypeVar("T")


@dataclass
class _Cursor:
    actor: str
    clock: Clock
    tips: tuple[str, ...]


# Set on the thread that runs a parallel branch. The session actor is the default.
_branch: ContextVar[_Cursor | None] = ContextVar("afr_branch", default=None)


class _Parallel:
    def __init__(self, session: Session, clock: Clock, tips: tuple[str, ...]) -> None:
        self.session = session
        self.base_clock = clock
        self.base_tips = tips
        self.branches: list[_Cursor] = []
        self._actors: set[str] = set()
        self._inflight = 0
        self._closed = False
        self._mu = threading.Lock()

    def do(self, actor: str, fn: Callable[[], T]) -> T:
        if actor == self.session.actor:
            raise ValueError(f"actor {actor!r} is the session actor; pick another name")
        if _branch.get() is not None:
            raise RuntimeError("parallel branches cannot nest")
        with self._mu:
            if self._closed:
                raise RuntimeError("parallel block is closed")
            if actor in self._actors:
                raise ValueError(f"actor {actor!r} is already used in this parallel block")
            self._actors.add(actor)
            self._inflight += 1
            cursor = _Cursor(actor, self.base_clock, self.base_tips)
        branch_token = _branch.set(cursor)
        session_token = set_session(self.session)
        try:
            return fn()
        finally:
            reset_session(session_token)
            _branch.reset(branch_token)
            with self._mu:
                self._inflight -= 1
                self.branches.append(cursor)

    def finish(self) -> None:
        with self._mu:
            if self._closed:
                return
            self._closed = True
            if self._inflight:
                raise RuntimeError("a parallel branch is still running")
            branches = list(self.branches)
        with self.session._mu:
            clock = self.base_clock
            tips: list[str] = []
            for branch in branches:
                clock = clock.merge(branch.clock)
                if branch.tips != self.base_tips:
                    tips.extend(branch.tips)
            self.session._root.clock = clock
            self.session._root.tips = tuple(tips) if tips else self.base_tips


class Session:
    def __init__(
        self,
        store: Store,
        *,
        name: str,
        entrypoint: str | None,
        tape: Tape | None,
    ) -> None:
        self.store = store
        self.name = name
        self.entrypoint = entrypoint
        self.actor = "agent"
        self.run_id = secrets.token_hex(8)
        self._tape = tape
        self._root = _Cursor(self.actor, Clock.zero(), ())
        self._mu = threading.Lock()
        self.opened = False
        self._sealed = False

    @property
    def live(self) -> bool:
        return self._tape is None or self._tape.live

    def open(self) -> None:
        tape = self._tape
        self.store.create_run(
            run_id=self.run_id,
            name=self.name,
            entrypoint=self.entrypoint,
            created_ns=time.time_ns(),
            replay_of=None if tape is None else tape.source_run_id,
            replay_until=None if tape is None else tape.until,
        )
        self.opened = True

    def close(self, exc: BaseException | None) -> None:
        if not self.opened or self._sealed:
            return
        if exc is None:
            self._seal("ok", None)
        else:
            self._seal("error", f"{type(exc).__name__}: {exc}")

    def rng(self, seed: int) -> random.Random:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an int")
        self._serve(
            kind="seed",
            name="random",
            payload={"seed": seed},
            call=lambda: ({"seed": seed}, None, None, None),
        )
        return random.Random(seed)

    def complete(
        self,
        llm: LLM,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        payload = freeze(
            {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": messages,
                "tools": tools,
            }
        )

        def call() -> tuple[Any, str | None, int | None, int | None]:
            completion = llm.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                seed=seed,
                tools=tools,
            )
            return completion_to_json(completion), None, completion.tokens_in, completion.tokens_out

        step = self._serve(kind="llm", name=model, payload=payload, call=call)
        output = step.output if isinstance(step.output, dict) else {}
        return completion_from_json(output, tokens_in=step.tokens_in, tokens_out=step.tokens_out)

    def call_tool(self, name: str, /, **kwargs: Any) -> Any:
        return lookup(name)(**kwargs)

    def invoke_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        payload = {"args": {key: freeze(value) for key, value in bound.arguments.items()}}

        def call() -> tuple[Any, str | None, int | None, int | None]:
            return freeze(fn(*args, **kwargs)), None, None, None

        step = self._serve(kind="tool", name=name, payload=payload, call=call)
        return freeze(step.output)

    @contextmanager
    def parallel(self) -> Iterator[_Parallel]:
        with self._mu:
            base_clock = self._root.clock
            base_tips = self._root.tips
        group = _Parallel(self, base_clock, base_tips)
        try:
            yield group
        finally:
            group.finish()

    def _serve(
        self,
        *,
        kind: str,
        name: str,
        payload: Any,
        call: Callable[[], tuple[Any, str | None, int | None, int | None]],
    ) -> Step:
        payload = freeze(payload)
        tape = self._tape
        if tape is not None:
            taped = tape.next_for(kind, name, payload)
            if taped is not None:
                started = time.monotonic_ns()
                step = self._emit(
                    kind=kind,
                    name=name,
                    payload=payload,
                    output=taped.output,
                    error=taped.error,
                    tokens_in=taped.tokens_in,
                    tokens_out=taped.tokens_out,
                    started_ns=time.time_ns(),
                    duration_ns=time.monotonic_ns() - started,
                )
                if taped.error:
                    raise RecordedFailure(taped.error)
                return step
        started_ns = time.time_ns()
        started = time.monotonic_ns()
        try:
            output, error, tokens_in, tokens_out = call()
        except Exception as exc:
            self._emit(
                kind=kind,
                name=name,
                payload=payload,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
                tokens_in=None,
                tokens_out=None,
                started_ns=started_ns,
                duration_ns=time.monotonic_ns() - started,
            )
            raise
        return self._emit(
            kind=kind,
            name=name,
            payload=payload,
            output=output,
            error=error,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            started_ns=started_ns,
            duration_ns=time.monotonic_ns() - started,
        )

    def _seal(self, status: str, error: str | None) -> None:
        self._sealed = True
        self._emit(
            kind="seal",
            name="seal",
            payload={"status": status},
            output=None,
            error=error,
            tokens_in=None,
            tokens_out=None,
            started_ns=time.time_ns(),
            duration_ns=0,
        )

    def _active(self) -> _Cursor:
        return _branch.get() or self._root

    def _emit(
        self,
        *,
        kind: str,
        name: str,
        payload: Any,
        output: Any,
        error: str | None,
        tokens_in: int | None,
        tokens_out: int | None,
        started_ns: int,
        duration_ns: int,
    ) -> Step:
        payload = freeze(payload)
        output = None if output is None else freeze(output)
        with self._mu:
            cursor = self._active()
            clock = cursor.clock.tick(cursor.actor)
            parents = tuple(sorted(cursor.tips))
            step_id = make_step_id(self.run_id, kind, name, cursor.actor, parents, payload)
            record = {
                "step_id": step_id,
                "run_id": self.run_id,
                "kind": kind,
                "name": name,
                "actor": cursor.actor,
                "parents": list(parents),
                "lamport": clock.lamport,
                "vc": dict(clock.vector),
                "input": payload,
                "output": output,
                "error": error,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "started_ns": started_ns,
                "duration_ns": duration_ns,
            }
            step = self.store.append_step(record)
            cursor.clock = clock
            cursor.tips = (step.step_id,)
            return step


class Recorder:
    def __init__(self, directory: str | Path = ".afr") -> None:
        self.store = Store(directory)
        self._plan: tuple[Run, int | None] | None = None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def arm_replay(self, run_id: str, *, until: int | None = None) -> None:
        if until is not None and until < 0:
            raise ValueError("until must be >= 0")
        run = self.store.load(run_id)
        if not run.intact:
            raise ChainBroken(run.failure or f"run {run.run_id} is not intact")
        self._plan = (run, until)

    @contextmanager
    def session(self, name: str, *, entrypoint: str | None = None) -> Iterator[Session]:
        plan = self._plan
        self._plan = None
        tape = None if plan is None else Tape(plan[0], plan[1])
        session = Session(self.store, name=name, entrypoint=entrypoint, tape=tape)
        token = set_session(session)
        try:
            session.open()
            yield session
        except BaseException as exc:
            session.close(exc)
            raise
        else:
            session.close(None)
        finally:
            reset_session(token)

    @contextmanager
    def replay(
        self,
        run_id: str,
        *,
        until: int | None = None,
        name: str | None = None,
        entrypoint: str | None = None,
    ) -> Iterator[Session]:
        source = self.store.load(run_id)
        self.arm_replay(source.run_id, until=until)
        chosen = source.name if name is None else name
        chosen_entry = source.entrypoint if entrypoint is None else entrypoint
        with self.session(chosen, entrypoint=chosen_entry) as session:
            yield session
