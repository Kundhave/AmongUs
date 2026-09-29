"""Tests for SimConfig defaults and immutability (SPEC §3, §17)."""

import dataclasses

import pytest

from amongus.config import SimConfig


def test_defaults():
    """SimConfig() matches the documented defaults for key tunables."""
    cfg = SimConfig()
    assert cfg.seed == 0
    assert cfg.n_players == 8
    assert cfg.n_impostors == 2
    assert cfg.alpha_risk == 2.0
    assert cfg.theta_vote == 0.35
    assert cfg.reactor_timer == 30


def test_frozen():
    """SimConfig is frozen: mutating a field raises."""
    cfg = SimConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1  # type: ignore[misc]


def test_replace_for_scenarios():
    """Scenarios override fields via dataclasses.replace without mutating the original."""
    cfg = SimConfig()
    cfg2 = dataclasses.replace(cfg, seed=42)
    assert cfg2.seed == 42
    assert cfg.seed == 0
