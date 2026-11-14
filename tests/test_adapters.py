import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_flight_recorder import Recorder
from agent_flight_recorder.adapters import MockLLM, OpenAICompatLLM
from agent_flight_recorder.models import Completion


class _Handler(BaseHTTPRequestHandler):
    status = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.last_body = body  # type: ignore[attr-defined]
        self.server.last_auth = self.headers.get("Authorization")  # type: ignore[attr-defined]
        if _Handler.status != 200:
            raw = b"nope"
            self.send_response(_Handler.status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        payload = {
            "model": body["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_mock_script_and_missing_key() -> None:
    llm = MockLLM(script=[Completion("hi", (), "m", 1, 1)])
    assert llm.complete(messages=[], model="m").content == "hi"
    with pytest.raises(RuntimeError, match="exhausted"):
        llm.complete(messages=[], model="m")
    with pytest.raises(RuntimeError, match="API key"):
        OpenAICompatLLM(api_key="").complete(messages=[], model="m")


def test_openai_compatible_adapter_against_a_local_server(tmp_path: Path) -> None:
    _Handler.status = 200
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm = OpenAICompatLLM(base_url=f"http://127.0.0.1:{port}/v1", api_key="test-key", timeout=5)
        with Recorder(tmp_path) as recorder:
            with recorder.session("remote") as session:
                completion = session.complete(
                    llm,
                    messages=[{"role": "user", "content": "look up x"}],
                    model="demo",
                    temperature=0,
                    seed=7,
                )
                run_id = session.run_id
            run = recorder.store.load(run_id)
        assert completion.tool_calls[0].name == "lookup"
        assert completion.tool_calls[0].arguments == {"q": "x"}
        assert completion.tokens_in == 3
        assert completion.tokens_out == 4
        assert server.last_auth == "Bearer test-key"  # type: ignore[attr-defined]
        assert server.last_body["seed"] == 7  # type: ignore[attr-defined]
        assert run.steps[0].kind == "llm"
        assert run.steps[0].output["tool_calls"][0]["name"] == "lookup"

        _Handler.status = 500
        with pytest.raises(RuntimeError, match="llm http 500"):
            llm.complete(messages=[{"role": "user", "content": "x"}], model="demo")
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
