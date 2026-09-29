"""Retry classification and backoff timing for LLMClient's network calls (SPEC §9.3)."""

import re
import time

# Fatal: retrying burns wall-clock for an error that will never succeed (bad model name,
# bad/missing credentials). Everything else — including 503/UNAVAILABLE/429/RESOURCE_EXHAUSTED
# and any error text we don't recognise — is worth a retry.
_FATAL_RE = re.compile(
    r"\b(404|NOT_FOUND|401|403|UNAUTHENTICATED|PERMISSION_DENIED|invalid[ _-]?api[ _-]?key)\b",
    re.IGNORECASE,
)


def is_retryable(exc: Exception) -> bool:
    """Classify a transport error: fatal codes never retry; unrecognised ones do (safe default)."""
    return not _FATAL_RE.search(str(exc))


def jitter() -> float:
    """A small sub-second, non-seeded offset so concurrent workers don't retry in lockstep.

    Derived from the wall clock, not `random` — §0's seeded-RNG rule governs *world*
    randomness so replays stay byte-identical; retry timing never touches world state or
    the cache key, so it carries no determinism obligation.
    """
    return (time.monotonic_ns() % 1_000_000) / 1_000_000 * 0.5


def backoff_delay(backoff_base: float, attempt: int) -> float:
    """Seconds to sleep before retrying `attempt` (0-based): `base * 2**attempt` plus jitter."""
    return backoff_base * (2**attempt) + jitter()
