import json
import pathlib
import sys
import threading
import types

import pytest

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _StubTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, *_, **__):  # pragma: no cover - simple stub
            return {"input_ids": [[0]]}

        def apply_chat_template(self, *_, **__):  # pragma: no cover - simple stub
            return ""

    class _StubModel:
        def eval(self):  # pragma: no cover - simple stub
            return self

        def generate(self, *_, **__):  # pragma: no cover - simple stub
            return [[0]]

    class AutoTokenizer(_StubTokenizer):
        @classmethod
        def from_pretrained(cls, *_, **__):  # pragma: no cover - simple stub
            return cls()

    class AutoModelForCausalLM(_StubModel):
        @classmethod
        def from_pretrained(cls, *_, **__):  # pragma: no cover - simple stub
            return cls()

    class TextIteratorStreamer:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    transformers_stub.AutoTokenizer = AutoTokenizer
    transformers_stub.AutoModelForCausalLM = AutoModelForCausalLM
    transformers_stub.TextIteratorStreamer = TextIteratorStreamer
    sys.modules["transformers"] = transformers_stub

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from agent import (
    MiniTfidf,
    LocalAgent,
    Tool,
    build_kb,
    build_tools,
    save_state,
    tool_read_kb_file,
    tool_save_note,
    tool_web_search,
)


class DummyTokenizer:
    eos_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):  # pragma: no cover - trivial
        return "decoded"


class DummyModel:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def generate(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        if "streamer" in kwargs:
            return None
        return [[1, 2, 3]]


class DummyStreamer:
    def __init__(self, *_, **__):
        self._chunks = []

    def __iter__(self):
        return iter(self._chunks)


def test_streaming_immediate_eos(monkeypatch):
    agent = object.__new__(LocalAgent)
    agent.stream_output = True
    agent.tok = DummyTokenizer()
    dummy_model = DummyModel()
    agent.model = dummy_model

    monkeypatch.setattr("transformers.TextIteratorStreamer", DummyStreamer)

    model_inputs = {"input_ids": [[0]], "attention_mask": [[1]]}

    result = LocalAgent._generate_text(agent, model_inputs)

    assert result == "decoded"
    assert len(dummy_model.calls) == 2
    assert "streamer" in dummy_model.calls[0]
    assert "streamer" not in dummy_model.calls[1]


def _build_agent(monkeypatch, responses):
    agent = object.__new__(LocalAgent)
    agent.state = {"history": []}
    agent.tools = {}
    agent.tok = DummyTokenizer()
    agent.stream_output = False

    monkeypatch.setattr(
        "agent.use_chat_template",
        lambda _tok, _msgs: {"input_ids": [[0]], "attention_mask": [[1]]},
    )
    monkeypatch.setattr(
        LocalAgent,
        "_extract_last_json",
        staticmethod(lambda text: json.loads(text) if text.strip().startswith("{") else None),
    )

    def fake_messages(self, current_msg):
        return [{"role": "user", "content": current_msg}]

    agent._messages = types.MethodType(fake_messages, agent)

    queue = list(responses)

    def fake_generate(_):
        return queue.pop(0)

    agent._generate_text = fake_generate
    return agent, queue


def test_step_returns_final_answer_and_saves_state(monkeypatch, tmp_path):
    agent, queue = _build_agent(monkeypatch, ['{"final_answer": "done"}'])

    saved_states = []

    def fake_save(state):
        saved_states.append(json.loads(json.dumps(state)))

    monkeypatch.setattr("agent.save_state", fake_save)

    result = LocalAgent.step(agent, "hello")

    assert result == "done"
    assert agent.state["history"][-1]["content"] == "done"
    assert saved_states and saved_states[-1]["history"][-1]["content"] == "done"
    assert not queue


def test_step_handles_tool_call(monkeypatch):
    responses = [
        '{"tool_name": "echo", "arguments": {"msg": "hi"}}',
        '{"final_answer": "bye"}',
    ]
    agent, queue = _build_agent(monkeypatch, responses)

    called = []

    def handler(args):
        called.append(args)
        return {"echo": args.get("msg")}

    agent.tools = {"echo": Tool("echo", "", handler)}

    monkeypatch.setattr("agent.save_state", lambda state: None)

    result = LocalAgent.step(agent, "ping")

    assert result == "bye"
    assert called == [{"msg": "hi"}]
    assert any(entry["content"].startswith("TOOL_RESULT(echo)") for entry in agent.state["history"])  # tool log present
    assert not queue


def test_step_handles_unknown_tool_error(monkeypatch):
    responses = [
        '{"tool_name": "mystery", "arguments": {}}',
        '{"final_answer": "done"}',
    ]
    agent, queue = _build_agent(monkeypatch, responses)

    monkeypatch.setattr("agent.save_state", lambda state: None)

    result = LocalAgent.step(agent, "ask")

    assert result == "done"
    assert any("unknown tool: mystery" in entry["content"] for entry in agent.state["history"])
    assert not queue


def test_step_non_json_response(monkeypatch):
    agent, queue = _build_agent(monkeypatch, ["plain text response"])  # type: ignore[arg-type]

    saved = []

    monkeypatch.setattr("agent.save_state", lambda state: saved.append(json.loads(json.dumps(state))))

    result = LocalAgent.step(agent, "question")

    assert result == "plain text response"
    assert saved and saved[-1]["history"][-1]["content"] == "plain text response"
    assert not queue


def test_mini_tfidf_search_orders_by_similarity():
    docs = [
        "Python code and unit tests",
        "Gardening tips and tricks",
        "Advanced Python testing strategies",
    ]
    paths = ["a.txt", "b.txt", "c.txt"]

    tfidf = MiniTfidf(docs, paths)
    results = tfidf.search("python testing", k=2)

    assert len(results) == 2
    scores = [score for score, *_ in results]
    assert scores[0] >= scores[1]
    assert results[0][1] == "c.txt"


def test_tool_read_kb_file_path_validation(monkeypatch, tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    safe_file = kb_dir / "note.txt"
    safe_file.write_text("hello", encoding="utf-8")

    monkeypatch.setattr("agent.DATA_DIR", kb_dir)

    ok = tool_read_kb_file({"path": str(safe_file.name), "max_chars": 10})
    assert ok["content"] == "hello"

    denied = tool_read_kb_file({"path": str(tmp_path / "note.txt"), "max_chars": 10})
    assert denied["error"].startswith("access denied")


def test_tool_save_note_empty_error(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.NOTES_PATH", tmp_path / "notes.md")
    assert tool_save_note({"text": "   "}) == {"error": "empty note"}


def test_tool_add_kb_document_creates_file_and_refreshes_index(monkeypatch, tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    monkeypatch.setattr("agent.DATA_DIR", kb_dir)

    tools = build_tools(None)
    add_tool = tools["add_kb_document"]

    result = add_tool.handler({"title": "Test Doc", "content": "Python agents can store knowledge."})

    assert result["ok"] is True
    assert result["refresh_kb"] is True

    expected_path = kb_dir / "Test_Doc.txt"
    assert expected_path.exists()

    kb = build_kb(data_dir=kb_dir)
    assert kb is not None

    hits = kb.search("python agents", k=1)
    assert hits
    top_path = hits[0][1]
    assert top_path.endswith("Test_Doc.txt")


def test_save_state_writes_file(monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("agent.STATE_PATH", state_file)

    state = {"history": [{"role": "user", "content": "hi"}]}
    save_state(state)

    assert json.loads(state_file.read_text(encoding="utf-8")) == state


class _DummyResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text or json.dumps(self._data)
        self.headers = {}

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def test_tool_web_search_returns_trimmed_summaries(monkeypatch):
    payload = {
        "results": [
            {
                "title": "Example Title",
                "content": "Snippet from the result with a lot of extra    spaces.",
                "url": "https://example.com/article",
            },
            {
                "title": "Another",
                "content": "Second snippet",
                "url": "https://example.com/second",
            },
        ]
    }

    def fake_get(url, params, timeout, headers):
        assert url == "https://search.example.com/search"
        assert params["q"] == "python"
        assert params["format"] == "json"
        assert timeout == 15
        return _DummyResponse(200, payload)

    monkeypatch.setenv("AGENT_SEARXNG_URL", "https://search.example.com")
    monkeypatch.setattr("requests.get", fake_get)

    result = tool_web_search({"query": "python", "max_results": 1})

    assert "results" in result
    assert len(result["results"]) == 1
    summary = result["results"][0]
    assert "Example Title" in summary
    assert "https://example.com/article" in summary


def test_tool_web_search_handles_http_errors(monkeypatch):
    def fake_get(url, params, timeout, headers):
        return _DummyResponse(503, {}, text="Service unavailable")

    monkeypatch.setenv("AGENT_SEARXNG_URL", "https://search.example.com")
    monkeypatch.setattr("requests.get", fake_get)

    result = tool_web_search({"query": "fail"})

    assert result["error"].startswith("search failed")
    assert result["details"] == "Service unavailable"
