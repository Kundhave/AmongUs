"""Shared test fixtures: guarantee no test can accidentally reach the live Gemini API."""

import pytest


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip GEMINI_API_KEY from the environment for every test in this suite.

    The suite must never make a network call or need an API key (§17/§9.3). This is a
    belt-and-suspenders guard: even if the ambient shell happens to export a real key,
    every test still exercises the offline `deliberator = "gemini"` no-key fallback path
    (§9.3's "missing key behaves exactly as template") rather than the network.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
