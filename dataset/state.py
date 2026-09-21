"""Deriving BladeVision state features + input state from Phase-0 captures (Phase 1).

Two jobs:
  1. `egocentric_features` — turn the mod's world-space self/opponent ground-truth into the
     same *egocentric* view the sim policy consumes (opponent position in the player's own
     facing frame), so perception targets line up with the RL brain (docs/DESIGN.md §4.3).
  2. `INPUT_MAP` / held-state reduction helpers — map raw pynput key/button names to the
     canonical action channels.

Minecraft yaw convention: degrees, 0 = +Z (south), increasing clockwise; horizontal facing
vector is (-sin, cos). The player's right-hand vector is (-cos, -sin). Pitch is degrees,
positive downward.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

# Canonical action channels and the raw pynput names that map to them.
# (Sprint via double-tap-W isn't visible as a key; the mod's `sprinting` flag is authoritative
#  for that, so we keep the key channel only for an explicit sprint keybind.)
INPUT_MAP = {
    "w": "forward",
    "a": "left",
    "s": "back",
    "d": "right",
    "Key.space": "jump",
    "Key.shift": "sneak",
    "Key.shift_l": "sneak",
    "Key.ctrl": "sprint",
    "Key.ctrl_l": "sprint",
    "Button.left": "attack",
    "Button.right": "use",
}

# The boolean channels every per-frame input snapshot carries.
INPUT_CHANNELS = ("forward", "left", "back", "right", "jump", "sneak", "sprint", "attack", "use")


def empty_input_state() -> Dict[str, bool]:
    return {c: False for c in INPUT_CHANNELS}


def egocentric_features(self_state: dict, opp_state: Optional[dict]) -> Optional[dict]:
    """Opponent pose relative to self, in self's facing frame. None if no opponent.

    Returns forward/right offset (blocks), horizontal distance, vertical delta, and the
    bearing to the opponent as sin/cos (0 == dead ahead, +right).
    """
    if opp_state is None:
        return None

    yaw = math.radians(self_state["yaw"])
    fx, fz = -math.sin(yaw), math.cos(yaw)      # forward (horizontal)
    rx, rz = -math.cos(yaw), -math.sin(yaw)     # player's right

    dx = opp_state["x"] - self_state["x"]
    dy = opp_state["y"] - self_state["y"]
    dz = opp_state["z"] - self_state["z"]

    forward = dx * fx + dz * fz
    right = dx * rx + dz * rz
    dist = math.hypot(dx, dz)
    bearing = math.atan2(right, forward)

    return {
        "fwd": forward,
        "right": right,
        "dist": dist,
        "dy": dy,
        "bearing_sin": math.sin(bearing),
        "bearing_cos": math.cos(bearing),
    }
