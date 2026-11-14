from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from agent_flight_recorder import __version__
from agent_flight_recorder.diff import diff_runs
from agent_flight_recorder.errors import (
    ChainBroken,
    ReplayDivergence,
    RunSealed,
    SealConflict,
    StoreBusy,
)
from agent_flight_recorder.report import diff_html, render_diff, render_list, render_run, run_html
from agent_flight_recorder.session import Recorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="afr")
    parser.add_argument("--dir", default=None, help="tape directory (default: $AFR_DIR or .afr)")
    parser.add_argument("--version", action="version", version=f"afr {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list recorded runs")

    show = sub.add_parser("show", help="print one run")
    show.add_argument("run")
    show.add_argument("--causal", action="store_true", help="include parent step ids")

    diff = sub.add_parser("diff", help="compare two runs")
    diff.add_argument("run_a")
    diff.add_argument("run_b")
    diff.add_argument("--html", metavar="PATH", help="also write an HTML report")

    replay = sub.add_parser("replay", help="re-run an entrypoint against a tape")
    replay.add_argument("run")
    replay.add_argument(
        "--until", type=int, default=None, help="serve steps with seq <= N, then go live"
    )
    replay.add_argument("--entry", default=None, help="module:function or path.py:function")

    export = sub.add_parser("export", help="write one run as HTML")
    export.add_argument("run")
    export.add_argument("--html", required=True, metavar="PATH")

    args = parser.parse_args(argv)
    directory = args.dir or os.environ.get("AFR_DIR", ".afr")
    try:
        with Recorder(directory) as recorder:
            return _dispatch(args, recorder)
    except ReplayDivergence as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (
        KeyError,
        ChainBroken,
        StoreBusy,
        RunSealed,
        SealConflict,
        FileNotFoundError,
        ImportError,
        AttributeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, recorder: Recorder) -> int:
    if args.cmd == "list":
        render_list(recorder.store.list_runs())
        return 0
    if args.cmd == "show":
        render_run(recorder.store.load(args.run), causal=args.causal)
        return 0
    if args.cmd == "diff":
        diff = diff_runs(recorder.store.load(args.run_a), recorder.store.load(args.run_b))
        render_diff(diff)
        if args.html:
            Path(args.html).write_text(diff_html(diff), encoding="utf-8")
            print(f"wrote {args.html}")
        return 1 if diff.diverged else 0
    if args.cmd == "export":
        text = run_html(recorder.store.load(args.run))
        Path(args.html).write_text(text, encoding="utf-8")
        print(f"wrote {args.html}")
        return 0
    if args.cmd == "replay":
        return _replay(args, recorder)
    raise AssertionError(args.cmd)


def _replay(args: argparse.Namespace, recorder: Recorder) -> int:
    if args.until is not None and args.until < 0:
        print("until must be >= 0", file=sys.stderr)
        return 2
    run = recorder.store.load(args.run)
    entry = args.entry or run.entrypoint
    if not entry:
        print(
            "run has no entrypoint; pass --entry module:function or path.py:function",
            file=sys.stderr,
        )
        return 2
    recorder.arm_replay(run.run_id, until=args.until)
    load_entry(entry)(recorder)
    return 0


def load_entry(spec: str) -> Callable[[Recorder], object]:
    if ":" not in spec:
        raise ValueError(
            f"entrypoint must be 'module:function' or 'path.py:function', not {spec!r}"
        )
    target, func_name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"entrypoint not found: {path}")
        loaded = importlib.util.spec_from_file_location(f"afr_entry_{path.stem}", path)
        if loaded is None or loaded.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"{spec} has no callable {func_name}")
    return cast(Callable[[Recorder], object], fn)
