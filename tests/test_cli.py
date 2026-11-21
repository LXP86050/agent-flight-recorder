from pathlib import Path

import pytest

from agent_flight_recorder import Recorder
from agent_flight_recorder.adapters import MockLLM
from agent_flight_recorder.cli import main
from agent_flight_recorder.models import Completion

ENTRY = """
from pathlib import Path

from agent_flight_recorder import Recorder
from agent_flight_recorder.adapters import MockLLM
from agent_flight_recorder.models import Completion

CALLS = Path(__file__).with_suffix(".calls")


def main(recorder: Recorder) -> None:
    def handle(messages, model):
        CALLS.write_text((CALLS.read_text(encoding="utf-8") if CALLS.exists() else "") + "x")
        return Completion("pong", (), model, tokens_in=2, tokens_out=1)

    entry = str(Path(__file__)) + ":main"
    with recorder.session("ping", entrypoint=entry) as session:
        session.rng(1)
        session.complete(
            MockLLM(handle),
            messages=[{"role": "user", "content": "ping"}],
            model="m",
        )
"""


def test_cli_lists_shows_diffs_exports_and_replays(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = tmp_path / "entry.py"
    entry.write_text(ENTRY, encoding="utf-8")
    tapes = tmp_path / "tapes"
    from agent_flight_recorder.cli import load_entry

    with Recorder(tapes) as recorder:
        load_entry(f"{entry}:main")(recorder)
        run_id = recorder.store.list_runs()[0].run_id
        with recorder.session("other") as session:
            session.complete(
                MockLLM(script=[Completion("other", (), "m", 1, 1)]),
                messages=[{"role": "user", "content": "else"}],
                model="m",
            )
            other = session.run_id

    assert (tmp_path / "entry.calls").read_text(encoding="utf-8") == "x"
    assert main(["--dir", str(tapes), "list"]) == 0
    listed = capsys.readouterr().out
    assert run_id in listed

    assert main(["--dir", str(tapes), "show", run_id[:8], "--causal"]) == 0
    shown = capsys.readouterr().out
    assert "pong" in shown
    assert "chain intact" in shown

    assert main(["--dir", str(tapes), "replay", run_id]) == 0
    assert (tmp_path / "entry.calls").read_text(encoding="utf-8") == "x"

    assert main(["--dir", str(tapes), "diff", run_id, other]) == 1
    diffed = capsys.readouterr().out
    assert "first divergence" in diffed

    html_path = tmp_path / "one.html"
    diff_path = tmp_path / "diff.html"
    assert main(["--dir", str(tapes), "export", run_id, "--html", str(html_path)]) == 0
    assert main(["--dir", str(tapes), "diff", run_id, run_id, "--html", str(diff_path)]) == 0
    assert run_id in html_path.read_text(encoding="utf-8")
    assert "no divergence" in diff_path.read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out


def test_show_prints_the_exception_that_sealed_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    with Recorder(tmp_path) as recorder:
        with pytest.raises(RuntimeError, match="between steps"):
            with recorder.session("bare") as session:
                session.rng(1)
                raise RuntimeError("between steps")
        run_id = session.run_id
    assert main(["--dir", str(tmp_path), "show", run_id]) == 0
    assert "error: RuntimeError: between steps" in capsys.readouterr().out


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0

    assert main(["--dir", str(tmp_path), "list"]) == 0
    assert "no runs" in capsys.readouterr().out

    with Recorder(tmp_path) as recorder:
        with recorder.session("bare") as session:
            session.rng(1)
            bare = session.run_id

    assert main(["--dir", str(tmp_path), "show", "nope"]) == 2
    assert main(["--dir", str(tmp_path), "replay", bare]) == 2
    assert main(["--dir", str(tmp_path), "replay", bare, "--until", "-1", "--entry", "x:y"]) == 2
    err = capsys.readouterr().err
    assert "no run" in err
    assert "entrypoint" in err
    assert "until" in err
