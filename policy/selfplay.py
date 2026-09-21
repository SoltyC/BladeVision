"""PPO self-play training for the BladeVision sword bot (Milestone A).

Recipe (see docs/DESIGN.md §4.3, §10):
  1. Warm up against the scripted baseline until the agent can reliably beat it.
  2. Switch to league self-play: each generation, freeze the current policy into a pool and
     train the next generation against opponents sampled from that pool.

Runs entirely on CPU in the fast simulator — no Minecraft client, no CUDA required. On Apple
Silicon this trains many times faster than real-time.

Usage:
    python -m policy.selfplay --generations 20 --steps-per-gen 200000 --out runs/selfplay
"""

from __future__ import annotations

import argparse
import io
import os
from typing import Optional

import numpy as np

from sim.gym_env import DuelEnv
from sim.opponents import SelfPlayPool, scripted_baseline


def _lazy_imports():
    """Import heavy RL deps lazily so `--help` and unit tests don't require them."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    return PPO, DummyVecEnv, VecMonitor


class FrozenPolicyOpponent:
    """Wraps a snapshot of a PPO model as an opponent callable (deterministic actions).

    The model is reloaded from serialized bytes so it is a fully independent frozen copy —
    later training on the live model cannot mutate this opponent's weights.
    """

    def __init__(self, model_bytes: bytes) -> None:
        from stable_baselines3 import PPO

        self._model = PPO.load(io.BytesIO(model_bytes), device="cpu")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


def _snapshot_bytes(model) -> bytes:
    buf = io.BytesIO()
    model.save(buf)
    return buf.getvalue()


def make_env(pool: SelfPlayPool, rng: np.random.Generator, warmup: bool):
    """Build a DuelEnv factory. During warmup the opponent is always the scripted baseline."""

    def _factory():
        env = DuelEnv(seed=int(rng.integers(2**31)))
        if warmup:
            env.set_opponent(scripted_baseline)
        else:
            env.set_opponent_provider(pool.sample)
        return env

    return _factory


def evaluate(model, opponent, n_episodes: int, seed: int) -> float:
    """Return win rate of `model` (player A) vs `opponent` over n_episodes."""
    env = DuelEnv(opponent=opponent, seed=seed)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if terminated and info["b_health"] <= 0.0 < info["a_health"]:
                wins += 1
    return wins / n_episodes


def train(
    generations: int,
    steps_per_gen: int,
    out_dir: str,
    warmup_gens: int = 2,
    pool_size: int = 8,
    seed: int = 0,
) -> str:
    PPO, DummyVecEnv, VecMonitor = _lazy_imports()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pool = SelfPlayPool(max_size=pool_size, rng=rng)

    # Start in warmup mode against the scripted baseline.
    venv = VecMonitor(DummyVecEnv([make_env(pool, rng, warmup=True)]))
    model = PPO(
        "MlpPolicy",
        venv,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.005,
        learning_rate=3e-4,
        verbose=1,
        seed=seed,
    )

    for gen in range(generations):
        warmup = gen < warmup_gens
        # Rebuild env with the right opponent mode for this generation.
        venv = VecMonitor(DummyVecEnv([make_env(pool, rng, warmup=warmup)]))
        model.set_env(venv)

        print(f"\n=== Generation {gen} ({'warmup' if warmup else 'self-play'}) ===")
        model.learn(total_timesteps=steps_per_gen, reset_num_timesteps=False)

        # Freeze this generation into the league pool.
        pool.add(FrozenPolicyOpponent(_snapshot_bytes(model)))

        wr_baseline = evaluate(model, scripted_baseline, n_episodes=50, seed=seed + gen)
        print(f"[gen {gen}] win rate vs scripted baseline: {wr_baseline:.2%} | pool={len(pool)}")

        ckpt = os.path.join(out_dir, f"gen_{gen:03d}.zip")
        model.save(ckpt)

    final = os.path.join(out_dir, "final.zip")
    model.save(final)
    print(f"\nSaved final model to {final}")
    return final


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="BladeVision PPO self-play (Milestone A)")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--steps-per-gen", type=int, default=200_000)
    parser.add_argument("--warmup-gens", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--out", type=str, default="runs/selfplay")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train(
        generations=args.generations,
        steps_per_gen=args.steps_per_gen,
        out_dir=args.out,
        warmup_gens=args.warmup_gens,
        pool_size=args.pool_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
