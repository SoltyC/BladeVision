"""Two-player sword-PvP physics core for the BladeVision fast simulator.

No RL / Gym dependency lives here — this is the pure environment mechanics so it can be
unit-tested and reused. The Gymnasium wrapper (`gym_env.py`) drives this for single-agent
self-play training.

Coordinate system: flat circular arena on the (x, z) plane, matching Minecraft's ground
plane. `y` is height (for jump crits only). Yaw is the world-space facing angle in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import mechanics as mech


@dataclass(frozen=True)
class PlayerState:
    """Immutable snapshot of one fighter."""

    x: float
    z: float
    vx: float
    vz: float
    y: float
    vy: float
    yaw: float
    health: float
    on_ground: bool
    sprinting: bool
    cooldown: int  # ticks since last swing (charge ramps with this)

    @staticmethod
    def spawn(x: float, z: float, yaw: float) -> "PlayerState":
        return PlayerState(
            x=x, z=z, vx=0.0, vz=0.0, y=0.0, vy=0.0, yaw=yaw,
            health=mech.MAX_HEALTH, on_ground=True, sprinting=False,
            cooldown=mech.ATTACK_COOLDOWN_TICKS,
        )


@dataclass(frozen=True)
class Action:
    """One tick of intent from an agent.

    move_x / move_z are desired movement in the agent's *local* frame (forward = +x_local),
    each in [-1, 1]. turn is desired yaw change in [-1, 1] scaled to MAX_TURN_PER_TICK.
    """

    move_x: float
    move_z: float
    turn: float
    jump: bool
    sprint: bool
    attack: bool


@dataclass(frozen=True)
class StepOutcome:
    """Per-tick result for bookkeeping and reward shaping."""

    damage_dealt_by_a: float
    damage_dealt_by_b: float
    a_hit_landed: bool
    b_hit_landed: bool
    a_crit: bool
    b_crit: bool


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _apply_movement(p: PlayerState, act: Action) -> PlayerState:
    """Integrate one player's movement/turn/jump for a tick. Returns a new state.

    Hit resolution and knockback are handled by the arena step after both players move, so
    this function only concerns locomotion.
    """
    # Turn (yaw) first so movement uses the updated facing.
    yaw = p.yaw + _clamp(act.turn, -1.0, 1.0) * mech.MAX_TURN_PER_TICK
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi

    # Desired world-space movement direction from local intent, rotated by yaw.
    mx = _clamp(act.move_x, -1.0, 1.0)
    mz = _clamp(act.move_z, -1.0, 1.0)
    mag = math.hypot(mx, mz)
    sprinting = act.sprint and mx > 0.3 and p.on_ground  # sprint requires forward intent
    target_speed = (mech.SPRINT_SPEED if sprinting else mech.WALK_SPEED)

    if mag > 1e-6:
        # Normalise local intent, then rotate into world frame.
        lx, lz = mx / mag, mz / mag
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = lx * cos_y - lz * sin_y
        wz = lx * sin_y + lz * cos_y
        desired_vx = wx * target_speed
        desired_vz = wz * target_speed
    else:
        desired_vx = desired_vz = 0.0

    accel = mech.MOVE_ACCEL if p.on_ground else mech.AIR_CONTROL
    vx = p.vx + (desired_vx - p.vx) * accel
    vz = p.vz + (desired_vz - p.vz) * accel

    # Friction when no input.
    if mag <= 1e-6:
        friction = mech.GROUND_FRICTION if p.on_ground else mech.AIR_FRICTION
        vx *= friction
        vz *= friction

    # Vertical: jump + gravity.
    vy = p.vy
    on_ground = p.on_ground
    if act.jump and on_ground:
        vy = mech.JUMP_VELOCITY
        on_ground = False
    vy = max(vy - mech.GRAVITY, mech.TERMINAL_FALL)

    y = p.y + vy
    if y <= 0.0:
        y = 0.0
        vy = 0.0
        on_ground = True

    x = p.x + vx
    z = p.z + vz

    # Clamp inside the circular arena (soft wall: cancel outward velocity).
    r = math.hypot(x, z)
    if r > mech.ARENA_RADIUS:
        scale = mech.ARENA_RADIUS / r
        x *= scale
        z *= scale
        vx *= 0.0
        vz *= 0.0

    cooldown = min(p.cooldown + 1, mech.ATTACK_COOLDOWN_TICKS)

    return replace(
        p, x=x, z=z, vx=vx, vz=vz, y=y, vy=vy, yaw=yaw,
        on_ground=on_ground, sprinting=sprinting, cooldown=cooldown,
    )


def _resolve(attacker: PlayerState, victim: PlayerState) -> mech.HitResult:
    if attacker.cooldown < 1:  # already swung this tick's window handled by caller
        return mech.NO_HIT
    return mech.resolve_attack(
        attacker_x=attacker.x, attacker_z=attacker.z, attacker_yaw=attacker.yaw,
        attacker_cooldown=attacker.cooldown, attacker_sprinting=attacker.sprinting,
        attacker_vy=attacker.vy, attacker_on_ground=attacker.on_ground,
        victim_x=victim.x, victim_z=victim.z,
    )


def _apply_hit(victim: PlayerState, hit: mech.HitResult) -> PlayerState:
    return replace(
        victim,
        health=max(0.0, victim.health - hit.damage),
        vx=victim.vx + hit.kb_x,
        vz=victim.vz + hit.kb_z,
        vy=hit.kb_y,
        on_ground=False,
    )


class Arena:
    """Stateful two-player duel. Holds current PlayerStates; `step` advances one tick.

    Kept as a small class (rather than pure functions) because the Gym wrapper needs an
    object with clear reset/step semantics. The state itself remains immutable snapshots.
    """

    def __init__(self, rng) -> None:
        self._rng = rng
        self.a: PlayerState = PlayerState.spawn(0.0, 0.0, 0.0)
        self.b: PlayerState = PlayerState.spawn(0.0, 0.0, 0.0)
        self.tick: int = 0
        self.reset()

    def reset(self) -> None:
        """Randomised symmetric spawn facing each other."""
        angle = self._rng.uniform(-math.pi, math.pi)
        dist = self._rng.uniform(4.0, 8.0)
        ax, az = -math.cos(angle) * dist / 2.0, -math.sin(angle) * dist / 2.0
        bx, bz = math.cos(angle) * dist / 2.0, math.sin(angle) * dist / 2.0
        self.a = PlayerState.spawn(ax, az, math.atan2(bz - az, bx - ax))
        self.b = PlayerState.spawn(bx, bz, math.atan2(az - bz, ax - bx))
        self.tick = 0

    def step(self, act_a: Action, act_b: Action) -> StepOutcome:
        """Advance one tick: move both, then resolve simultaneous attacks."""
        # Record whether each intends to (and can) attack this tick before moving, using the
        # pre-move charge — the swing lands based on geometry after movement.
        a_wants = act_a.attack and mech.attack_charge(self.a.cooldown) >= mech.MIN_CHARGE_FOR_DAMAGE
        b_wants = act_b.attack and mech.attack_charge(self.b.cooldown) >= mech.MIN_CHARGE_FOR_DAMAGE

        new_a = _apply_movement(self.a, act_a)
        new_b = _apply_movement(self.b, act_b)

        hit_a = mech.NO_HIT
        hit_b = mech.NO_HIT
        if a_wants:
            hit_a = _resolve(new_a, new_b)  # a attacks b
        if b_wants:
            hit_b = _resolve(new_b, new_a)  # b attacks a

        # Reset attacker cooldown on any swing attempt (even a miss consumes the swing).
        if act_a.attack:
            new_a = replace(new_a, cooldown=0)
        if act_b.attack:
            new_b = replace(new_b, cooldown=0)

        if hit_a.landed:
            new_b = _apply_hit(new_b, hit_a)
        if hit_b.landed:
            new_a = _apply_hit(new_a, hit_b)

        self.a, self.b = new_a, new_b
        self.tick += 1

        return StepOutcome(
            damage_dealt_by_a=hit_a.damage,
            damage_dealt_by_b=hit_b.damage,
            a_hit_landed=hit_a.landed,
            b_hit_landed=hit_b.landed,
            a_crit=hit_a.crit,
            b_crit=hit_b.crit,
        )

    @property
    def done(self) -> bool:
        return self.a.health <= 0.0 or self.b.health <= 0.0
