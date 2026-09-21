"""Opponent policies for DuelEnv: a scripted baseline and a self-play checkpoint pool.

An opponent is any callable `obs -> action_index_array` (the same signature SB3's
`model.predict(obs, deterministic=True)[0]` provides). Keeping the interface this thin means
the env never needs to know whether it is fighting a script or a frozen network.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .gym_env import OpponentPolicy

# Observation indices (must match encode_obs order in gym_env.py).
_I_FWD = 0
_I_RIGHT = 1
_I_DIST = 2
_I_SIN_BEARING = 3
_I_COS_BEARING = 4
_I_CHARGE = 11

# Scripted action encodings (indices into ACTION_NVEC factors).
_MOVE_FWD = 2   # _MOVE_LEVELS[2] == +1
_MOVE_NONE = 1
_TURN_LEFT = 0
_TURN_MILD_LEFT = 1
_TURN_NONE = 2
_TURN_MILD_RIGHT = 3
_TURN_RIGHT = 4


def scripted_baseline(obs: np.ndarray) -> np.ndarray:
    """A competent-but-beatable heuristic: face the enemy, close distance, swing in reach.

    This is the Phase-A benchmark the RL agent must exceed. It is intentionally simple —
    no strafing, no crit timing — so a self-play policy has clear room to outclass it.
    """
    dist = obs[_I_DIST] * 12.0  # de-normalise (ARENA_RADIUS)
    bearing = math.atan2(obs[_I_SIN_BEARING], obs[_I_COS_BEARING])
    charge = obs[_I_CHARGE]

    # Turn toward the opponent proportional to bearing error.
    if bearing > 0.35:
        turn = _TURN_RIGHT
    elif bearing > 0.08:
        turn = _TURN_MILD_RIGHT
    elif bearing < -0.35:
        turn = _TURN_LEFT
    elif bearing < -0.08:
        turn = _TURN_MILD_LEFT
    else:
        turn = _TURN_NONE

    aligned = abs(bearing) < 0.4
    in_reach = dist <= 3.1

    move_x = _MOVE_FWD if (not in_reach and aligned) else _MOVE_NONE
    sprint = 1 if (move_x == _MOVE_FWD and dist > 4.0) else 0
    attack = 1 if (in_reach and aligned and charge > 0.9) else 0

    return np.array([move_x, _MOVE_NONE, turn, 0, sprint, attack], dtype=np.int64)


def idle(_obs: np.ndarray) -> np.ndarray:
    """A do-nothing punching bag, useful for smoke tests."""
    return np.array([1, 1, 2, 0, 0, 0], dtype=np.int64)


class SelfPlayPool:
    """A rolling pool of frozen past checkpoints to sample opponents from (league play).

    Sampling from history — not just the latest policy — prevents the agent from overfitting
    a single opponent and reduces strategy-cycling (PSRO / AlphaStar-league intuition).
    """

    def __init__(self, max_size: int, rng: np.random.Generator, latest_bias: float = 0.5) -> None:
        self._policies: List[OpponentPolicy] = []
        self._max_size = max_size
        self._rng = rng
        self._latest_bias = latest_bias  # probability of picking the most recent checkpoint

    def add(self, policy: OpponentPolicy) -> None:
        self._policies.append(policy)
        if len(self._policies) > self._max_size:
            self._policies.pop(0)  # drop oldest

    def sample(self) -> OpponentPolicy:
        if not self._policies:
            return scripted_baseline
        if len(self._policies) > 1 and self._rng.random() < self._latest_bias:
            return self._policies[-1]
        return self._policies[int(self._rng.integers(len(self._policies)))]

    def __len__(self) -> int:
        return len(self._policies)
