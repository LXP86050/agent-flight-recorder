from __future__ import annotations

import html
import os
import time
from typing import IO, Any

from rich import box
from rich.console import Console
from rich.table import Table

from agent_flight_recorder.diff import TraceDiff, summary
from agent_flight_recorder.hashing import canonical
from agent_flight_recorder.models import Run, Step


def make_console(file: IO[str] | None = None) -> Console:
    # Rich ignores a width on a non-tty unless height is set too.
    raw = os.environ.get("COLUMNS")
    width = int(raw) if raw and raw.isdigit() else None
    height = 24 if width is not None else None
    return Console(
        file=file,
        markup=False,
        emoji=False,
        highlight=False,
        width=width,
        height=height,
    )


def format_duration(ns: int) -> str:
    if ns < 1_000:
        return f"{ns}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f}ms"
    return f"{ns / 1_000_000_000:.2f}s"


def render_list(runs: list[Run], console: Console | None = None) -> None:
    console = console or make_console()
    if not runs:
        console.print("no runs")
        return
    table = Table(box=box.SIMPLE, header_style="bold", show_edge=False)
    for title in ("run", "created", "name", "steps", "status"):
        table.add_column(title)
    for run in runs:
        table.add_row(
            run.run_id,
            _created(run),
            run.name,
            str(len(run.steps)),
            run.status,
        )
    console.print(table)


def render_run(run: Run, console: Console | None = None, *, causal: bool = False) -> None:
    console = console or make_console()
    parts = [
        f"run {run.run_id}",
        run.name,
        f"{len(run.steps)} steps",
        run.status,
        _created(run) + "Z",
    ]
    if run.replay_of:
        until = "" if run.replay_until is None else f" until {run.replay_until}"
        parts.append(f"replay of {run.replay_of}{until}")
    console.print("  ".join(parts))
    if run.entrypoint:
        console.print(f"entrypoint {run.entrypoint}")
    if run.intact:
        console.print("chain intact")
    else:
        console.print(f"chain broken: {run.failure}")
    table = Table(box=box.SIMPLE, header_style="bold", show_edge=False)
    for title in ("seq", "lamport", "kind", "name", "dt", "tokens", "note"):
        table.add_column(title)
    if causal:
        table.add_column("parents", overflow="fold")
    for step in run.steps:
        row = [
            str(step.seq),
            str(step.lamport),
            step.kind,
            step.name,
            format_duration(step.duration_ns),
            _tokens(step),
            _note(step),
        ]
        if causal:
            row.append(" ".join(parent[:8] for parent in step.parents) or "-")
        table.add_row(*row)
    console.print(table)


def render_diff(diff: TraceDiff, console: Console | None = None) -> None:
    console = console or make_console()
    console.print(f"a  {diff.left.run_id}  {diff.left.name}  {diff.left.status}")
    console.print(f"b  {diff.right.run_id}  {diff.right.name}  {diff.right.status}")
    console.print(summary(diff))
    table = Table(box=box.SIMPLE, header_style="bold", show_edge=False)
    for title in ("a", "b", "a step", "b step", "note"):
        table.add_column(title)
    for row in diff.rows:
        style = "bold" if row.kind != "same" else None
        table.add_row(
            "" if row.left is None else str(row.left.seq),
            "" if row.right is None else str(row.right.seq),
            _label(row.left),
            _label(row.right),
            row.detail,
            style=style,
        )
    console.print(table)


def run_html(run: Run) -> str:
    rows = []
    for step in run.steps:
        rows.append(
            "<article class='step'>"
            f"<h2>seq {step.seq} {html.escape(step.kind)} {html.escape(step.name)}</h2>"
            f"<p class='meta'>{html.escape(_step_meta(step))}</p>"
            f"<h3>input</h3><pre>{_pre(step.input)}</pre>"
            f"<h3>output</h3><pre>{_pre(step.output)}</pre>"
            + (f"<p class='err'>{html.escape(step.error)}</p>" if step.error else "")
            + "</article>"
        )
    chain = "chain intact" if run.intact else f"chain broken: {run.failure}"
    replay = ""
    if run.replay_of:
        until = "" if run.replay_until is None else f" until {run.replay_until}"
        replay = f"<p class='meta'>replay of {html.escape(run.replay_of)}{html.escape(until)}</p>"
    body = "\n".join(rows)
    title = f"{run.name} {run.run_id}"
    return _page(
        title,
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='meta'>{html.escape(run.status)} · {len(run.steps)} steps · "
        f"{html.escape(_created(run))}Z · {html.escape(chain)}</p>"
        f"{replay}{body}",
    )


def diff_html(diff: TraceDiff) -> str:
    body_rows = []
    for row in diff.rows:
        body_rows.append(
            f"<tr class='{html.escape(row.kind)}'>"
            f"<td>{_seq(row.left)}</td><td>{_seq(row.right)}</td>"
            f"<td>{html.escape(_label(row.left))}</td>"
            f"<td>{html.escape(_label(row.right))}</td>"
            f"<td>{html.escape(row.detail)}</td></tr>"
        )
        if row.kind != "same":
            body_rows.append(
                "<tr class='detail'><td colspan='5'>"
                f"<pre>a input  {_pre(None if row.left is None else row.left.input)}\n"
                f"b input  {_pre(None if row.right is None else row.right.input)}\n"
                f"a output {_pre(None if row.left is None else row.left.output)}\n"
                f"b output {_pre(None if row.right is None else row.right.output)}</pre>"
                "</td></tr>"
            )
    title = f"diff {diff.left.run_id} {diff.right.run_id}"
    table = (
        "<table><thead><tr>"
        "<th>a</th><th>b</th><th>a step</th><th>b step</th><th>note</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )
    return _page(
        title,
        f"<h1>trace diff</h1>"
        f"<p class='meta'>a {html.escape(diff.left.run_id)} {html.escape(diff.left.name)} "
        f"{html.escape(diff.left.status)}<br>"
        f"b {html.escape(diff.right.run_id)} {html.escape(diff.right.name)} "
        f"{html.escape(diff.right.status)}</p>"
        f"<p>{html.escape(summary(diff))}</p>{table}",
    )


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 2rem auto; max-width: 68rem; padding: 0 1rem;
  font: 16px/1.45 Palatino, Georgia, serif; color: #1c1915; background: #f7f4ee; }}
h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; }}
h2 {{ font: 600 0.95rem/1.3 ui-monospace, Menlo, Consolas, monospace; margin: 1.25rem 0 0.2rem; }}
h3 {{ font: 600 0.75rem/1.2 ui-monospace, Menlo, Consolas, monospace; margin: 0.6rem 0 0.15rem;
  letter-spacing: 0.04em; text-transform: uppercase; color: #5c564c; }}
.meta {{ color: #5c564c; margin-top: 0; }}
.err {{ color: #8a2b1f; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1rem;
  font: 12.5px/1.4 ui-monospace, Menlo, Consolas, monospace; }}
th, td {{ text-align: left; vertical-align: top; padding: 0.35rem 0.45rem;
  border-bottom: 1px solid #e4ddd0; }}
tr.args td, tr.output td, tr.name td {{ background: #fff4d6; }}
tr.left td, tr.right td {{ background: #fde8e4; }}
pre {{ white-space: pre-wrap; word-break: break-word; margin: 0.2rem 0 0.6rem;
  font: 12px/1.4 ui-monospace, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _created(run: Run) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(run.created_ns // 1_000_000_000))


def _tokens(step: Step) -> str:
    if step.tokens_in is None and step.tokens_out is None:
        return ""
    return f"{step.tokens_in or 0}/{step.tokens_out or 0}"


def _note(step: Step) -> str:
    text = ""
    if step.error and step.kind != "seal":
        text = step.error
    elif step.kind == "llm":
        text = _llm_note(step)
    elif step.kind == "tool" and isinstance(step.input, dict):
        args = step.input.get("args")
        if isinstance(args, dict):
            text = " ".join(f"{key}={args[key]}" for key in sorted(args))
        else:
            text = canonical(step.input)
    elif step.kind == "seed" and isinstance(step.input, dict):
        text = f"seed={step.input.get('seed')}"
    elif step.kind == "seal" and isinstance(step.input, dict):
        text = str(step.input.get("status") or "")
    if len(text) > 72:
        return text[:69] + "..."
    return text


def _llm_note(step: Step) -> str:
    output = step.output if isinstance(step.output, dict) else {}
    calls = output.get("tool_calls") or []
    if calls:
        return "call " + ", ".join(str(call.get("name")) for call in calls)
    content = output.get("content")
    if content is None:
        return ""
    return str(content).replace("\n", " ")


def _label(step: Step | None) -> str:
    if step is None:
        return ""
    return f"{step.kind} {step.name}"


def _seq(step: Step | None) -> str:
    if step is None:
        return ""
    return str(step.seq)


def _step_meta(step: Step) -> str:
    parents = " ".join(parent[:8] for parent in step.parents) or "-"
    return (
        f"actor {step.actor} · lamport {step.lamport} · {format_duration(step.duration_ns)}"
        f" · tokens {_tokens(step) or '-'} · parents {parents}"
    )


def _pre(value: Any) -> str:
    if value is None:
        return "null"
    return html.escape(canonical(value))
