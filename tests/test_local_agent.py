import pathlib
import sys
import threading

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from agent import LocalAgent


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
