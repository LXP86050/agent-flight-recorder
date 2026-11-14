from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from agent_flight_recorder.context import get_session

F = TypeVar("F", bound=Callable[..., Any])

_REGISTRY: dict[str, Callable[..., Any]] = {}


def record_tool(fn: F) -> F:
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session = get_session()
        if session is None:
            return fn(*args, **kwargs)
        return session.invoke_tool(name, fn, args, kwargs)

    _REGISTRY[name] = wrapper
    return wrapper  # type: ignore[return-value]


def lookup(name: str) -> Callable[..., Any]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no tool registered as {name!r}") from None
