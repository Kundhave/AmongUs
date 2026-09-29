"""Single source of randomness: a seeded numpy Generator factory."""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh seeded numpy Generator; the only entry point for randomness."""
    return np.random.default_rng(seed)
