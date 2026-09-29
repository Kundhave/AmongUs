"""Tests for risk-weighted cost function (SPEC §8.2, §17)."""

import dataclasses

from amongus.config import SimConfig
from amongus.search.costs import make_cost_fn, plain_cost_fn

_CFG = SimConfig()


def test_risk_term_never_negative():
    """cost(u, v, w) never dips below the raw edge weight w."""
    suspicion = {"red": 0.9, "blue": 0.3}
    last_seen = {"red": ("electrical", 10), "blue": ("electrical", 12)}
    cost_fn = make_cost_fn(suspicion, last_seen, tick=15, cfg=_CFG)
    assert cost_fn("storage", "electrical", 4) >= 4.0


def test_alpha_risk_zero_reproduces_plain_weight():
    """With alpha_risk=0, cost(u, v, w) equals the plain edge weight w."""
    cfg = dataclasses.replace(_CFG, alpha_risk=0.0)
    suspicion = {"red": 0.9}
    last_seen = {"red": ("electrical", 10)}
    cost_fn = make_cost_fn(suspicion, last_seen, tick=10, cfg=cfg)
    assert cost_fn("storage", "electrical", 4) == 4.0
    assert cost_fn("storage", "electrical", 4) == plain_cost_fn("storage", "electrical", 4)


def test_stale_sighting_contributes_nothing():
    """A sighting older than last_seen_decay adds no risk."""
    cfg = dataclasses.replace(_CFG, last_seen_decay=5)
    suspicion = {"red": 0.9}
    last_seen = {"red": ("electrical", 0)}
    cost_fn = make_cost_fn(suspicion, last_seen, tick=6, cfg=cfg)  # 6 ticks stale
    assert cost_fn("storage", "electrical", 4) == 4.0


def test_recent_sighting_adds_risk():
    """A sighting within last_seen_decay ticks adds alpha_risk * suspicion to the target room."""
    cfg = dataclasses.replace(_CFG, alpha_risk=2.0, last_seen_decay=5)
    suspicion = {"red": 0.9}
    last_seen = {"red": ("electrical", 0)}
    cost_fn = make_cost_fn(suspicion, last_seen, tick=5, cfg=cfg)  # exactly at decay boundary
    assert cost_fn("storage", "electrical", 4) == 4.0 + 2.0 * 0.9


def test_risk_only_applies_to_destination_room():
    """Risk is added for the destination room v, not the source room u."""
    cfg = dataclasses.replace(_CFG, alpha_risk=2.0)
    suspicion = {"red": 0.9}
    last_seen = {"red": ("storage", 0)}  # seen in u, not v
    cost_fn = make_cost_fn(suspicion, last_seen, tick=0, cfg=cfg)
    assert cost_fn("storage", "electrical", 4) == 4.0


def test_plain_cost_fn_returns_weight():
    """plain_cost_fn simply returns the edge weight, unmodified."""
    assert plain_cost_fn("storage", "electrical", 4) == 4.0
