from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from agent_flight_recorder.hashing import canonical, digest
from agent_flight_recorder.models import Run, Step


@dataclass(frozen=True, slots=True)
class DiffRow:
    kind: str
    left: Step | None
    right: Step | None
    detail: str


@dataclass(frozen=True, slots=True)
class TraceDiff:
    left: Run
    right: Run
    rows: tuple[DiffRow, ...]

    @property
    def diverged(self) -> bool:
        return any(row.kind != "same" for row in self.rows)

    @property
    def first(self) -> DiffRow | None:
        for row in self.rows:
            if row.kind != "same":
                return row
        return None


def summary(diff: TraceDiff) -> str:
    first = diff.first
    if first is None:
        return "no divergence"
    step = first.left or first.right
    if step is None:
        return first.detail
    return f"first divergence at seq {step.seq} {step.kind} {step.name}: {first.detail}"


def diff_runs(left: Run, right: Run) -> TraceDiff:
    matcher = SequenceMatcher(
        a=[_signature(step) for step in left.steps],
        b=[_signature(step) for step in right.steps],
        autojunk=False,
    )
    rows: list[DiffRow] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                rows.append(_classify(left.steps[i1 + offset], right.steps[j1 + offset]))
            continue
        left_block = list(left.steps[i1:i2])
        right_block = list(right.steps[j1:j2])
        for offset in range(max(len(left_block), len(right_block))):
            lstep = left_block[offset] if offset < len(left_block) else None
            rstep = right_block[offset] if offset < len(right_block) else None
            rows.append(_classify(lstep, rstep))
    return TraceDiff(left=left, right=right, rows=tuple(rows))


def _signature(step: Step) -> tuple[str, str, str]:
    return (step.kind, step.name, digest(canonical(step.input).encode()))


def _classify(left: Step | None, right: Step | None) -> DiffRow:
    if left is None and right is not None:
        return DiffRow("right", left, right, "only in b")
    if right is None and left is not None:
        return DiffRow("left", left, right, "only in a")
    if left is None or right is None:
        return DiffRow("left", left, right, "missing")
    if (
        left.kind == right.kind
        and left.name == right.name
        and canonical(left.input) == canonical(right.input)
    ):
        if canonical(left.output) == canonical(right.output) and left.error == right.error:
            return DiffRow("same", left, right, "")
        return DiffRow("output", left, right, _output_detail(left, right))
    if left.kind == right.kind and left.name == right.name:
        return DiffRow("args", left, right, _input_detail(left, right))
    if left.kind == right.kind:
        return DiffRow("name", left, right, f"{left.name} -> {right.name}")
    return DiffRow("name", left, right, f"{left.kind} {left.name} -> {right.kind} {right.name}")


def _input_detail(left: Step, right: Step) -> str:
    if left.kind == "tool":
        changes = _arg_changes(left.input, right.input)
        if changes:
            return "arguments differ (" + ", ".join(changes) + ")"
        return "arguments differ"
    return "inputs differ"


def _output_detail(left: Step, right: Step) -> str:
    brief = f"{_brief(left)} -> {_brief(right)}"
    if len(brief) > 160:
        brief = brief[:157] + "..."
    return f"output differs ({brief})"


def _arg_changes(left: Any, right: Any) -> list[str]:
    left_args = left.get("args") if isinstance(left, dict) else None
    right_args = right.get("args") if isinstance(right, dict) else None
    if not isinstance(left_args, dict) or not isinstance(right_args, dict):
        return []
    changes: list[str] = []
    for key in sorted(set(left_args) | set(right_args)):
        if key not in left_args:
            changes.append(f"{key}: <missing> -> {right_args[key]!r}")
        elif key not in right_args:
            changes.append(f"{key}: {left_args[key]!r} -> <missing>")
        elif left_args[key] != right_args[key]:
            changes.append(f"{key}: {left_args[key]!r} -> {right_args[key]!r}")
    return changes


def _brief(step: Step) -> str:
    if step.error and step.kind != "seal":
        return step.error
    if step.kind == "llm" and isinstance(step.output, dict):
        calls = step.output.get("tool_calls") or []
        if calls:
            rendered = []
            for call in calls:
                rendered.append(f"{call['name']}({canonical(call['arguments'])})")
            return "call " + ", ".join(rendered)
        return repr(step.output.get("content"))
    if step.output is None:
        return "null"
    text = canonical(step.output)
    if len(text) > 80:
        return text[:77] + "..."
    return text
