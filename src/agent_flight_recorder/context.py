from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_flight_recorder.session import Session

_current: ContextVar[Session | None] = ContextVar("afr_session", default=None)


def get_session() -> Session | None:
    return _current.get()


def set_session(session: Session | None) -> Token[Session | None]:
    return _current.set(session)


def reset_session(token: Token[Session | None]) -> None:
    _current.reset(token)
