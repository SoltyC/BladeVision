"""Dependency-free sanity checks for the fast simulator.

These exercise the combat mechanics and arena physics without numpy/torch/SB3, so they run
even before the RL stack is installed. Run with `pytest tests/` or `python tests/test_mechanics.py`.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import mechanics as mech
from sim.arena import Action, Arena, PlayerState


def _no_op() -> Action:
    return Action(0.0, 0.0, 0.0, False, False, False)


def test_charge_ramps_to_full():
    assert mech.attack_charge(0) == 0.0
    assert mech.attack_charge(mech.ATTACK_COOLDOWN_TICKS) == 1.0
    mid = mech.attack_charge(mech.ATTACK_COOLDOWN_TICKS // 2)
    assert 0.3 < mid < 0.7


def test_hit_lands_in_reach_and_facing():
    hit = mech.resolve_attack(
        attacker_x=0.0, attacker_z=0.0, attacker_yaw=0.0,
        attacker_cooldown=mech.ATTACK_COOLDOWN_TICKS, attacker_sprinting=False,
        attacker_vy=0.0, attacker_on_ground=True,
        victim_x=2.0, victim_z=0.0,
    )
    assert hit.landed
    assert hit.damage > 0.0
    assert hit.kb_x > 0.0  # knocked away along +x


def test_miss_out_of_reach():
    hit = mech.resolve_attack(
        attacker_x=0.0, attacker_z=0.0, attacker_yaw=0.0,
        attacker_cooldown=mech.ATTACK_COOLDOWN_TICKS, attacker_sprinting=False,
        attacker_vy=0.0, attacker_on_ground=True,
        victim_x=10.0, victim_z=0.0,
    )
    assert not hit.landed


def test_miss_when_facing_away():
    hit = mech.resolve_attack(
        attacker_x=0.0, attacker_z=0.0, attacker_yaw=math.pi,  # facing -x
        attacker_cooldown=mech.ATTACK_COOLDOWN_TICKS, attacker_sprinting=False,
        attacker_vy=0.0, attacker_on_ground=True,
        victim_x=2.0, victim_z=0.0,  # target is at +x
    )
    assert not hit.landed


def test_crit_requires_airborne_descending_not_sprinting():
    common = dict(
        attacker_x=0.0, attacker_z=0.0, attacker_yaw=0.0,
        attacker_cooldown=mech.ATTACK_COOLDOWN_TICKS,
        victim_x=2.0, victim_z=0.0,
    )
    crit = mech.resolve_attack(attacker_sprinting=False, attacker_vy=-0.2, attacker_on_ground=False, **common)
    ground = mech.resolve_attack(attacker_sprinting=False, attacker_vy=0.0, attacker_on_ground=True, **common)
    assert crit.crit and not ground.crit
    assert crit.damage > ground.damage


def test_sprint_increases_knockback():
    common = dict(
        attacker_x=0.0, attacker_z=0.0, attacker_yaw=0.0,
        attacker_cooldown=mech.ATTACK_COOLDOWN_TICKS, attacker_vy=0.0, attacker_on_ground=True,
        victim_x=2.0, victim_z=0.0,
    )
    sprint = mech.resolve_attack(attacker_sprinting=True, **common)
    walk = mech.resolve_attack(attacker_sprinting=False, **common)
    assert math.hypot(sprint.kb_x, sprint.kb_z) > math.hypot(walk.kb_x, walk.kb_z)


def test_arena_reset_symmetric_and_full_health():
    arena = Arena(random.Random(0))
    arena.reset()
    assert arena.a.health == mech.MAX_HEALTH
    assert arena.b.health == mech.MAX_HEALTH
    # Spawns mirror around the origin.
    assert abs(arena.a.x + arena.b.x) < 1e-6
    assert abs(arena.a.z + arena.b.z) < 1e-6


def test_stationary_players_never_damage_each_other():
    arena = Arena(random.Random(1))
    arena.reset()
    for _ in range(mech.TICK_RATE * 5):
        outcome = arena.step(_no_op(), _no_op())
        assert outcome.damage_dealt_by_a == 0.0
        assert outcome.damage_dealt_by_b == 0.0
    assert not arena.done


def test_adjacent_attacker_eventually_kills():
    """An attacker glued to a passive victim should win within the round."""
    arena = Arena(random.Random(2))
    # Place A right next to B, facing it; B does nothing.
    arena.a = PlayerState.spawn(0.0, 0.0, 0.0)
    arena.b = PlayerState.spawn(1.5, 0.0, math.pi)
    attack = Action(0.0, 0.0, 0.0, False, False, True)
    killed = False
    for _ in range(mech.TICK_RATE * 20):
        # Re-aim A at B each tick and keep it adjacent (B gets knocked back).
        dx = arena.b.x - arena.a.x
        dz = arena.b.z - arena.a.z
        arena.a = PlayerState.spawn(arena.b.x - 1.4 * math.cos(math.atan2(dz, dx)),
                                    arena.b.z - 1.4 * math.sin(math.atan2(dz, dx)),
                                    math.atan2(dz, dx))
        arena.a = _restore_charge(arena.a)
        arena.step(attack, _no_op())
        if arena.b.health <= 0.0:
            killed = True
            break
    assert killed


def _restore_charge(p: PlayerState) -> PlayerState:
    from dataclasses import replace

    return replace(p, cooldown=mech.ATTACK_COOLDOWN_TICKS)


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
