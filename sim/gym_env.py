"""Single-agent Gymnasium wrapper around the two-player Arena.

The training agent controls player A. Player B is driven by a *frozen* opponent policy
(scripted baseline early on, then sampled from a self-play checkpoint pool). This turns a
two-player game into a standard single-agent env so we can use an off-the-shelf, battle-tested
PPO (Stable-Baselines3) while still doing league-style self-play at the loop level.

Observation is fully egocentric (expressed in A's local frame), so the same encoding works
verbatim for the opponent by swapping the two players — a symmetric self-play setup.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

# Provider returns the opponent policy to use for the next episode (sampled on reset).
OpponentProvider = Callable[[], "OpponentPolicy"]

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import mechanics as mech
from .arena import Action, Arena, PlayerState

# Action factor sizes: move_x, move_z, turn, jump, sprint, attack.
ACTION_NVEC = [3, 3, 5, 2, 2, 2]
_MOVE_LEVELS = (-1.0, 0.0, 1.0)
_TURN_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)

OBS_DIM = 16
_MAX_EPISODE_TICKS = mech.TICK_RATE * 30  # 30-second round cap

# Opponent policy: obs vector -> action index array (len 6). Same interface SB3 predict gives.
OpponentPolicy = Callable[[np.ndarray], np.ndarray]


def decode_action(idx: np.ndarray) -> Action:
    """Map a MultiDiscrete action index array to a typed Action."""
    return Action(
        move_x=_MOVE_LEVELS[int(idx[0])],
        move_z=_MOVE_LEVELS[int(idx[1])],
        turn=_TURN_LEVELS[int(idx[2])],
        jump=bool(idx[3]),
        sprint=bool(idx[4]),
        attack=bool(idx[5]),
    )


def encode_obs(me: PlayerState, opp: PlayerState) -> np.ndarray:
    """Egocentric observation of `opp` from `me`'s point of view, all in me's local frame."""
    dx = opp.x - me.x
    dz = opp.z - me.z
    dist = math.hypot(dx, dz)

    cos_y, sin_y = math.cos(me.yaw), math.sin(me.yaw)
    # Rotate world delta into local frame (forward = +x_local).
    fwd = dx * cos_y + dz * sin_y
    right = -dx * sin_y + dz * cos_y

    # Opponent velocity in my local frame.
    opp_fwd = opp.vx * cos_y + opp.vz * sin_y
    opp_right = -opp.vx * sin_y + opp.vz * cos_y
    # My own velocity in local frame.
    my_fwd = me.vx * cos_y + me.vz * sin_y
    my_right = -me.vx * sin_y + me.vz * cos_y

    bearing = math.atan2(right, fwd)  # 0 = directly ahead

    obs = np.array(
        [
            fwd / mech.ARENA_RADIUS,
            right / mech.ARENA_RADIUS,
            min(dist / mech.ARENA_RADIUS, 1.5),
            math.sin(bearing),
            math.cos(bearing),
            opp_fwd / mech.SPRINT_SPEED,
            opp_right / mech.SPRINT_SPEED,
            my_fwd / mech.SPRINT_SPEED,
            my_right / mech.SPRINT_SPEED,
            me.health / mech.MAX_HEALTH,
            opp.health / mech.MAX_HEALTH,
            mech.attack_charge(me.cooldown),
            0.0 if me.on_ground else 1.0,
            me.vy / mech.JUMP_VELOCITY,
            0.0 if opp.on_ground else 1.0,
            opp.vy / mech.JUMP_VELOCITY,
        ],
        dtype=np.float32,
    )
    return obs


class DuelEnv(gym.Env):
    """Gymnasium env: agent = player A, opponent = frozen policy on player B."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Optional[OpponentPolicy] = None,
        opponent_provider: Optional[OpponentProvider] = None,
        reward_time_penalty: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._arena = Arena(self._rng)
        self._opponent = opponent
        self._opponent_provider = opponent_provider
        self._time_penalty = reward_time_penalty

    def set_opponent(self, opponent: OpponentPolicy) -> None:
        """Swap the frozen opponent (used by the self-play loop between rollouts)."""
        self._opponent = opponent

    def set_opponent_provider(self, provider: OpponentProvider) -> None:
        """Set a provider that re-samples the opponent at the start of each episode."""
        self._opponent_provider = provider

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._arena = Arena(self._rng)
        if self._opponent_provider is not None:
            self._opponent = self._opponent_provider()
        self._arena.reset()
        return encode_obs(self._arena.a, self._arena.b), {}

    def _opponent_action(self) -> Action:
        if self._opponent is None:
            return Action(0.0, 0.0, 0.0, False, False, False)  # idle dummy
        opp_obs = encode_obs(self._arena.b, self._arena.a)
        idx = self._opponent(opp_obs)
        return decode_action(np.asarray(idx))

    def step(self, action):
        act_a = decode_action(np.asarray(action))
        act_b = self._opponent_action()

        outcome = self._arena.step(act_a, act_b)

        # Reward from A's perspective: normalise damage by full health so a full kill ~= 1.
        reward = (outcome.damage_dealt_by_a - outcome.damage_dealt_by_b) / mech.MAX_HEALTH
        reward -= self._time_penalty  # nudge toward decisive play

        terminated = self._arena.done
        truncated = self._arena.tick >= _MAX_EPISODE_TICKS
        if terminated:
            a_alive = self._arena.a.health > 0.0
            b_alive = self._arena.b.health > 0.0
            if a_alive and not b_alive:
                reward += 1.0
            elif b_alive and not a_alive:
                reward -= 1.0

        obs = encode_obs(self._arena.a, self._arena.b)
        info = {
            "a_health": self._arena.a.health,
            "b_health": self._arena.b.health,
            "a_hit": outcome.a_hit_landed,
            "a_crit": outcome.a_crit,
        }
        return obs, float(reward), terminated, truncated, info
