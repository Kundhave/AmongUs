"""Cached, concurrent Gemini client: the only place that may touch the network (SPEC §9.3)."""

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_JSON_MIME = "application/json"

# Fatal: retrying burns wall-clock for an error that will never succeed (bad model name,
# bad/missing credentials). Everything else — including 503/UNAVAILABLE/429/RESOURCE_EXHAUSTED
# and any error text we don't recognise — is worth a retry.
_FATAL_RE = re.compile(
    r"\b(404|NOT_FOUND|401|403|UNAUTHENTICATED|PERMISSION_DENIED|invalid[ _-]?api[ _-]?key)\b",
    re.IGNORECASE,
)


def _is_retryable(exc: Exception) -> bool:
    """Classify a transport error: fatal codes never retry; unrecognised ones do (safe default)."""
    text = str(exc)
    if _FATAL_RE.search(text):
        return False
    return True


class _GenAIClient(Protocol):
    """The one method of google.genai.Client this module calls; lets tests inject a fake."""

    models: Any


def _read_api_key() -> str | None:
    """Read GEMINI_API_KEY from the environment only — never from a file or argument."""
    return os.environ.get("GEMINI_API_KEY")


def _build_genai_client(api_key: str) -> _GenAIClient:
    """Construct the real google.genai.Client, imported lazily so `template` needs no SDK."""
    from google import genai

    return genai.Client(api_key=api_key)


class LLMClient:
    """Hashes, caches and dispatches Gemini calls; never raises, never calls out on a hit."""

    def __init__(
        self,
        model: str,
        cache_path: str,
        genai_client: _GenAIClient | None = None,
        max_workers: int = 8,
    ) -> None:
        """Build the client: load the on-disk cache, and connect only if a key is present.

        `genai_client` lets tests inject a fake (or one that raises) instead of a real SDK
        client — the seam that proves a cache hit never reaches the network.
        """
        self.model = model
        self.cache_path = cache_path
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._cache: dict[str, str] = self._load_cache()
        if genai_client is not None:
            self._client: _GenAIClient | None = genai_client
        else:
            key = _read_api_key()
            if key:
                self._client = _build_genai_client(key)
            else:
                self._client = None
                logger.warning(
                    "GEMINI_API_KEY not set; deliberator falls back to template (no network)."
                )

    @property
    def enabled(self) -> bool:
        """True if this client can reach the network (a key or fake client was supplied)."""
        return self._client is not None

    def cache_key(self, prompt: str) -> str:
        """sha256(model + '\\n' + prompt), the cache key required by §9.3."""
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def call(self, prompt: str, force_refresh: bool = False) -> str | None:
        """Return the model's raw text for prompt: cache hit, else one call plus one retry.

        `force_refresh` skips the cache lookup (but still writes the result back under the
        same key) — used for GeminiDeliberator's single parse-failure retry, so a second
        attempt can actually reach the network instead of re-reading the same bad response.
        Never raises: any transport failure, after one retry, returns None.
        """
        key = self.cache_key(prompt)
        if not force_refresh:
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                return cached
        if self._client is None:
            return None
        text = self._call_network(prompt)
        if text is not None:
            self._store(key, text)
        return text

    def call_many(self, prompts: dict[str, str]) -> dict[str, str | None]:
        """Resolve many prompts concurrently (one thread pool round-trip); cache hits skip it.

        Keyed by an arbitrary caller-chosen id (here, an AgentId) rather than the prompt
        itself, so callers can tell results apart even if two prompts happened to collide.
        """
        results: dict[str, str | None] = {}
        pending: dict[str, str] = {}
        for req_id, prompt in prompts.items():
            key = self.cache_key(prompt)
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                results[req_id] = cached
            else:
                pending[req_id] = prompt
        if not pending:
            return results
        if self._client is None:
            for req_id in pending:
                results[req_id] = None
            return results
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending))) as pool:
            futures = {
                pool.submit(self._call_network, prompt): req_id
                for req_id, prompt in pending.items()
            }
            for future in futures:
                req_id = futures[future]
                text = future.result()
                if text is not None:
                    self._store(self.cache_key(pending[req_id]), text)
                results[req_id] = text
        return results

    def _call_network(self, prompt: str) -> str | None:
        """One call plus one retry on a transport error; never raises (§9.3)."""
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=_json_mode_config(),
                )
                return response.text
            except Exception as exc:  # noqa: BLE001 — a transport failure must never crash a run
                logger.info(
                    "Gemini call failed on attempt %d (%s)", attempt + 1, type(exc).__name__
                )
                continue
        return None

    def _store(self, key: str, response: str) -> None:
        """Cache one response in memory and append it to the JSONL cache file on disk."""
        with self._lock:
            self._cache[key] = response
            path = Path(self.cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps({"key": key, "response": response}) + "\n")

    def _load_cache(self) -> dict[str, str]:
        """Read every `{key, response}` line already on disk into memory."""
        cache: dict[str, str] = {}
        path = Path(self.cache_path)
        if not path.exists():
            return cache
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    cache[row["key"]] = row["response"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return cache


def _json_mode_config():
    """Build the google.genai request config asking for strict JSON output (§9.2)."""
    from google.genai import types

    return types.GenerateContentConfig(response_mime_type=_JSON_MIME)
