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
_I_DIST = 2
_I_SIN_BEARING = 3
_I_COS_BEARING = 4
_I_CHARGE = 11
_I_MY_AIRBORNE = 12
_I_MY_VY = 13

_ARENA_RADIUS = 12.0  # keep in sync with mechanics.ARENA_RADIUS

# Move-factor indices (into _MOVE_LEVELS = (-1, 0, +1)).
_MOVE_FWD = 2
_MOVE_NONE = 1
_STRAFE_RIGHT = 2
_STRAFE_LEFT = 0
# Turn-factor indices (into _TURN_LEVELS).
_TURN_LEFT = 0
_TURN_MILD_LEFT = 1
_TURN_NONE = 2
_TURN_MILD_RIGHT = 3
_TURN_RIGHT = 4


def _turn_toward(bearing: float) -> int:
    """Pick a turn level to reduce the bearing error (bearing 0 == dead ahead)."""
    if bearing > 0.35:
        return _TURN_RIGHT
    if bearing > 0.08:
        return _TURN_MILD_RIGHT
    if bearing < -0.35:
        return _TURN_LEFT
    if bearing < -0.08:
        return _TURN_MILD_LEFT
    return _TURN_NONE


def scripted_baseline(obs: np.ndarray) -> np.ndarray:
    """A competent sword duelist — the Phase-A yardstick the RL agent must exceed.

    Pure-sword skills only (no shields): it faces the target, sprints to close, circle-strafes
    in melee to be a moving target, and **jump-crits** — hopping, then swinging on the way down
    for 1.5x damage (crits require being airborne, descending, and not sprinting). This gives a
    genuine skill ceiling to measure against, unlike a stationary punching bag.
    """
    dist = obs[_I_DIST] * _ARENA_RADIUS
    bearing = math.atan2(obs[_I_SIN_BEARING], obs[_I_COS_BEARING])
    charge = obs[_I_CHARGE]
    airborne = obs[_I_MY_AIRBORNE] > 0.5
    vy = obs[_I_MY_VY]

    turn = _turn_toward(bearing)
    aligned = abs(bearing) < 0.45

    move_x = _MOVE_NONE
    move_z = _MOVE_NONE
    jump = 0
    sprint = 0
    attack = 0

    if aligned and dist > 3.3:
        # Close the gap; sprint from range, strafe a little so approaches aren't straight lines.
        move_x = _MOVE_FWD
        sprint = 1 if dist > 4.5 else 0
        move_z = _STRAFE_RIGHT
    elif dist <= 3.3:
        # Melee: circle-strafe and jump-crit when the swing is charged.
        move_z = _STRAFE_RIGHT
        if dist > 2.6:
            move_x = _MOVE_FWD  # stay glued to reach edge
        if charge > 0.9 and aligned:
            if not airborne:
                jump = 1                       # begin crit hop (no sprint -> crit stays valid)
            elif vy < 0.0:
                attack = 1                     # swing on the descent = crit
    else:
        # Not aligned yet: turn in place, drift sideways.
        move_z = _STRAFE_RIGHT

    return np.array([move_x, move_z, turn, jump, sprint, attack], dtype=np.int64)


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
