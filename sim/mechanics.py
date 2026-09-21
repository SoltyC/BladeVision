"""Minecraft 1.9+ sword-combat mechanics for the BladeVision fast simulator.

This module holds the constants and the pure hit-resolution logic. It is deliberately
approximate: the goal is a fast, faithful-enough training environment for self-play RL
(Milestone A), not a byte-exact clone of the vanilla server. Values are tuned to reproduce
the *feel* that matters for policy learning — reach limits, the attack-cooldown charge curve,
sprint/W-tap knockback, and jump crits.

All physics run at TICK_RATE ticks/second to match Minecraft.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Time ------------------------------------------------------------------
TICK_RATE = 20  # Hz, matches Minecraft
DT = 1.0 / TICK_RATE

# --- Combat ----------------------------------------------------------------
# Iron sword base damage (hearts are 2 HP each; 20 HP = full health).
SWORD_DAMAGE = 6.0
MAX_HEALTH = 20.0

# Attack reach: entity hitbox interaction distance. Vanilla creative reach is 3.0 blocks to
# the hitbox; survival attack reach is ~3.0 to the entity box. We measure center-to-center and
# subtract a hitbox radius so ~3.0 edge reach corresponds to this center distance.
ATTACK_REACH = 3.0
HITBOX_RADIUS = 0.3

# Attack cooldown: an iron/diamond sword has ~0.625 s between full-power hits (1.6 attack
# speed). Damage scales with charge: hitting before full charge is heavily penalised.
ATTACK_COOLDOWN_TICKS = round(0.625 * TICK_RATE)  # ~13 ticks
MIN_CHARGE_FOR_DAMAGE = 0.10  # below this, the swing does negligible damage

# Facing cone: you must be looking roughly at the target for the swing to connect.
ATTACK_CONE_DEG = 60.0

# Crit: attacker must be airborne and descending, and not sprinting. Deals 1.5x.
CRIT_MULTIPLIER = 1.5

# --- Knockback -------------------------------------------------------------
# Horizontal knockback velocity (blocks/tick-ish, scaled into our units).
BASE_KNOCKBACK = 0.40
SPRINT_KNOCKBACK_BONUS = 0.40  # sprint (W-tap) hits knock back much harder
KNOCKBACK_VERTICAL = 0.36      # upward pop on hit

# --- Movement --------------------------------------------------------------
WALK_SPEED = 4.317 / TICK_RATE   # blocks per tick
SPRINT_SPEED = 5.612 / TICK_RATE
MOVE_ACCEL = 0.9                 # fraction of the speed gap closed per tick (ground)
AIR_CONTROL = 0.20               # reduced steering while airborne
GROUND_FRICTION = 0.55           # velocity retained per tick when not inputting (ground)
AIR_FRICTION = 0.91

# --- Jump / gravity --------------------------------------------------------
JUMP_VELOCITY = 0.42     # initial upward velocity (blocks/tick)
GRAVITY = 0.08           # blocks/tick^2
TERMINAL_FALL = -3.0

# --- Turning ---------------------------------------------------------------
# Max yaw the agent may rotate per tick (radians). Bounds how fast it can flick; the executor
# layer (later phases) replaces this with a human motor model, but for the sim it caps aim.
MAX_TURN_PER_TICK = math.radians(35.0)

# --- Arena -----------------------------------------------------------------
ARENA_RADIUS = 12.0  # circular flat arena; agents are clamped inside


@dataclass(frozen=True)
class HitResult:
    """Outcome of one resolved attack. Immutable."""

    landed: bool
    damage: float
    crit: bool
    # Knockback velocity to apply to the victim (world x/z) and vertical pop.
    kb_x: float
    kb_z: float
    kb_y: float


NO_HIT = HitResult(landed=False, damage=0.0, crit=False, kb_x=0.0, kb_z=0.0, kb_y=0.0)


def attack_charge(cooldown_timer: int) -> float:
    """Charge fraction in [0, 1] given ticks since the last swing.

    0 immediately after a swing, ramping linearly to 1 once the cooldown has fully elapsed.
    """
    if cooldown_timer >= ATTACK_COOLDOWN_TICKS:
        return 1.0
    return max(0.0, cooldown_timer / ATTACK_COOLDOWN_TICKS)


def _angle_to(dx: float, dz: float) -> float:
    """World yaw (radians) pointing from origin toward (dx, dz)."""
    return math.atan2(dz, dx)


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


def resolve_attack(
    *,
    attacker_x: float,
    attacker_z: float,
    attacker_yaw: float,
    attacker_cooldown: int,
    attacker_sprinting: bool,
    attacker_vy: float,
    attacker_on_ground: bool,
    victim_x: float,
    victim_z: float,
) -> HitResult:
    """Pure resolution of a single attack attempt. Returns a HitResult.

    Does not mutate anything; the caller applies damage/knockback. Encapsulating this here
    keeps the arena step readable and makes the mechanics unit-testable in isolation.
    """
    dx = victim_x - attacker_x
    dz = victim_z - attacker_z
    dist = math.hypot(dx, dz)

    if dist > ATTACK_REACH + HITBOX_RADIUS:
        return NO_HIT

    # Must be facing the target within the cone.
    facing_err = abs(angle_diff(_angle_to(dx, dz), attacker_yaw))
    if facing_err > math.radians(ATTACK_CONE_DEG):
        return NO_HIT

    charge = attack_charge(attacker_cooldown)
    if charge < MIN_CHARGE_FOR_DAMAGE:
        return NO_HIT

    crit = (not attacker_on_ground) and (attacker_vy < 0.0) and (not attacker_sprinting)
    damage = SWORD_DAMAGE * charge * (CRIT_MULTIPLIER if crit else 1.0)

    # Knockback points from attacker to victim, normalised.
    if dist > 1e-6:
        nx, nz = dx / dist, dz / dist
    else:
        nx, nz = math.cos(attacker_yaw), math.sin(attacker_yaw)

    kb = BASE_KNOCKBACK + (SPRINT_KNOCKBACK_BONUS if attacker_sprinting else 0.0)
    return HitResult(
        landed=True,
        damage=damage,
        crit=crit,
        kb_x=nx * kb,
        kb_z=nz * kb,
        kb_y=KNOCKBACK_VERTICAL,
    )
