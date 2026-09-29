"""Tests for LLMClient: caching, the no-key fallback, retries, and concurrency (SPEC §9.3)."""

import hashlib
import json
import logging

from amongus.llm.client import LLMClient


class _FakeResponse:
    """Mimics google.genai's GenerateContentResponse: only the `.text` attribute is used."""

    def __init__(self, text: str) -> None:
        """Store the text this fake response returns."""
        self.text = text


class _CountingModels:
    """Fake `client.models`: returns canned text, or raises, and counts calls made."""

    def __init__(self, texts: list[str] | None = None, raise_times: int = 0) -> None:
        """texts are returned in order (repeating the last); raise_times calls raise first."""
        self.calls = 0
        self._texts = texts or ["{}"]
        self._raise_times = raise_times

    def generate_content(self, *, model, contents, config):
        """Return the next canned text, raising for the first raise_times calls."""
        self.calls += 1
        if self.calls <= self._raise_times:
            raise RuntimeError("simulated transport error")
        idx = min(self.calls - self._raise_times - 1, len(self._texts) - 1)
        return _FakeResponse(self._texts[idx])


class _FakeGenAIClient:
    """Fake google.genai.Client: only exposes `.models`."""

    def __init__(self, models: _CountingModels) -> None:
        """Wrap the fake models object under the real SDK's attribute name."""
        self.models = models


class _RaisingModels:
    """A `client.models` that always raises — proves a cache hit never reaches it."""

    def generate_content(self, *, model, contents, config):
        """Fail loudly; any call here is a test failure."""
        raise AssertionError("network must not be called on a cache hit")


def _cache_key(model: str, prompt: str) -> str:
    """Reproduce LLMClient.cache_key's sha256(model + '\\n' + prompt) independently."""
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


def test_cache_hit_makes_no_network_call(tmp_path) -> None:
    """A pre-populated cache entry is returned without ever touching the fake client."""
    cache_path = tmp_path / "cache.jsonl"
    key = _cache_key("gemini-2.0-flash", "hello")
    cache_path.write_text(json.dumps({"key": key, "response": "cached-reply"}) + "\n")

    client = LLMClient(
        model="gemini-2.0-flash",
        cache_path=str(cache_path),
        genai_client=_FakeGenAIClient(_RaisingModels()),
    )
    assert client.call("hello") == "cached-reply"


def test_missing_key_returns_none_and_logs(tmp_path, monkeypatch, caplog) -> None:
    """No GEMINI_API_KEY: the client is disabled, calls return None, and one line is logged."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        client = LLMClient(model="gemini-2.0-flash", cache_path=str(tmp_path / "c.jsonl"))
    assert client.enabled is False
    assert client.call("anything") is None
    assert any("GEMINI_API_KEY" in r.message for r in caplog.records)
    # The key's value (there isn't one here) must never appear; assert only the name does.
    assert all("GEMINI_API_KEY not set" in r.message for r in caplog.records if "KEY" in r.message)


def test_call_retries_once_then_returns_none(tmp_path) -> None:
    """A transport error retries exactly once, then returns None without raising."""
    models = _CountingModels(raise_times=2)
    client = LLMClient(
        model="m", cache_path=str(tmp_path / "c.jsonl"), genai_client=_FakeGenAIClient(models)
    )
    assert client.call("prompt") is None
    assert models.calls == 2


def test_call_succeeds_after_one_retry(tmp_path) -> None:
    """A single transient failure is absorbed by the one retry, returning the next reply."""
    models = _CountingModels(texts=["good"], raise_times=1)
    client = LLMClient(
        model="m", cache_path=str(tmp_path / "c.jsonl"), genai_client=_FakeGenAIClient(models)
    )
    assert client.call("prompt") == "good"
    assert models.calls == 2


def test_call_writes_and_reloads_the_cache(tmp_path) -> None:
    """A fresh call is persisted to disk and read back by a brand-new LLMClient instance."""
    cache_path = tmp_path / "cache.jsonl"
    models = _CountingModels(texts=["first-reply"])
    client = LLMClient(model="m", cache_path=str(cache_path), genai_client=_FakeGenAIClient(models))
    assert client.call("prompt") == "first-reply"

    reloaded = LLMClient(
        model="m", cache_path=str(cache_path), genai_client=_FakeGenAIClient(_RaisingModels())
    )
    assert reloaded.call("prompt") == "first-reply"  # served from disk, no network


def test_call_many_resolves_cached_and_fresh_prompts_concurrently(tmp_path) -> None:
    """call_many skips the network for cached prompts and dispatches only the rest."""
    cache_path = tmp_path / "cache.jsonl"
    key = _cache_key("m", "cached-prompt")
    cache_path.write_text(json.dumps({"key": key, "response": "cached"}) + "\n")
    models = _CountingModels(texts=["fresh"])
    client = LLMClient(
        model="m", cache_path=str(cache_path), genai_client=_FakeGenAIClient(models)
    )
    results = client.call_many({"a": "cached-prompt", "b": "new-prompt"})
    assert results == {"a": "cached", "b": "fresh"}
    assert models.calls == 1  # only the uncached prompt reached the fake network


def test_force_refresh_bypasses_the_cache(tmp_path) -> None:
    """force_refresh=True re-fetches from the network even when a cache entry exists."""
    cache_path = tmp_path / "cache.jsonl"
    key = _cache_key("m", "prompt")
    cache_path.write_text(json.dumps({"key": key, "response": "stale"}) + "\n")
    models = _CountingModels(texts=["fresh"])
    client = LLMClient(model="m", cache_path=str(cache_path), genai_client=_FakeGenAIClient(models))
    assert client.call("prompt") == "stale"  # normal call: cache hit
    assert client.call("prompt", force_refresh=True) == "fresh"
    assert models.calls == 1
