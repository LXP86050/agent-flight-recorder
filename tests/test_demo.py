from pathlib import Path

from agent_flight_recorder.cli import main
from demo import run_demo


def test_demo_replay_fix_and_cli(tmp_path: Path, monkeypatch) -> None:
    result = run_demo(tmp_path / "tapes")
    assert result.buggy_amount == 4200
    assert result.fixed_amount == 1500
    assert result.pinned_amount == 1500
    assert result.until >= 0

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.delenv("AFR_MODE", raising=False)
    tapes = str(tmp_path / "tapes")
    assert main(["--dir", tapes, "replay", result.buggy]) == 0
    monkeypatch.setenv("AFR_MODE", "fixed")
    assert main(["--dir", tapes, "replay", result.buggy]) == 1
    assert main(["--dir", tapes, "show", result.buggy]) == 0
