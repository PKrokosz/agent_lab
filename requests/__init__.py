"""Minimal requests stub for tests."""

class Response:  # pragma: no cover - placeholder only
    def __init__(self, status_code=200, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


def get(*args, **kwargs):  # pragma: no cover - replaced in tests
    raise RuntimeError("requests.get stubbed")
