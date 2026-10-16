from live_stt.torch_index import DEFAULT_TORCH_INDEX_URL, pip_index_args


def test_default_uses_cu126_index(monkeypatch):
    monkeypatch.delenv("LSTT_TORCH_INDEX_URL", raising=False)
    assert pip_index_args() == ["--extra-index-url", DEFAULT_TORCH_INDEX_URL]
    assert DEFAULT_TORCH_INDEX_URL.endswith("/cu126")


def test_env_override(monkeypatch):
    monkeypatch.setenv("LSTT_TORCH_INDEX_URL", "https://example.test/whl/cu121")
    assert pip_index_args() == ["--extra-index-url", "https://example.test/whl/cu121"]


def test_empty_env_disables(monkeypatch):
    monkeypatch.setenv("LSTT_TORCH_INDEX_URL", "")
    assert pip_index_args() == []
