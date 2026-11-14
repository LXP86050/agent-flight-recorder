from __future__ import annotations

import hashlib
import json
import math
from typing import Any

Json = dict[str, Any] | list[Any] | str | int | float | bool | None

GENESIS = "0" * 64


def freeze(obj: Any) -> Json:
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite float is not json")
        return obj
    if isinstance(obj, dict):
        return {str(key): freeze(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [freeze(value) for value in obj]
    raise TypeError(f"{type(obj).__name__} is not jsonable")


def canonical(obj: Any) -> str:
    return json.dumps(
        freeze(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_step_id(
    run_id: str,
    kind: str,
    name: str,
    actor: str,
    parents: tuple[str, ...] | list[str],
    payload: Any,
) -> str:
    material = {
        "run_id": run_id,
        "kind": kind,
        "name": name,
        "actor": actor,
        "parents": sorted(parents),
        "input": payload,
    }
    return digest(canonical(material).encode())


def body_hash(record: dict[str, Any]) -> str:
    """Hash the sealed observation, not the wall-clock fields around it.

    A retry after a crash can carry new timestamps. The body is the same
    observation, so the append is a no-op. A different output is a conflict.
    """

    body = {
        "step_id": record["step_id"],
        "run_id": record["run_id"],
        "kind": record["kind"],
        "name": record["name"],
        "actor": record["actor"],
        "parents": list(record["parents"]),
        "lamport": record["lamport"],
        "vc": record["vc"],
        "input": record["input"],
        "output": record["output"],
        "error": record["error"],
        "tokens_in": record["tokens_in"],
        "tokens_out": record["tokens_out"],
    }
    return digest(canonical(body).encode())
