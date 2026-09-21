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
import glob
import io
import json
import os
import re
import time
from typing import List, Optional, Tuple

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

    @classmethod
    def from_path(cls, path: str) -> "FrozenPolicyOpponent":
        with open(path, "rb") as fh:
            return cls(fh.read())

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


_CKPT_RE = re.compile(r"gen_(\d+)\.zip$")


def _find_checkpoints(out_dir: str) -> List[Tuple[int, str]]:
    """Return [(generation_index, path), ...] sorted ascending, from an existing run dir."""
    found = []
    for path in glob.glob(os.path.join(out_dir, "gen_*.zip")):
        m = _CKPT_RE.search(path)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def _log_progress(out_dir: str, record: dict) -> None:
    """Append one JSONL line to progress.jsonl and overwrite status.json with the latest."""
    with open(os.path.join(out_dir, "progress.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")
    with open(os.path.join(out_dir, "status.json"), "w") as fh:
        json.dump(record, fh, indent=2)


def train(
    generations: int,
    steps_per_gen: int,
    out_dir: str,
    warmup_gens: int = 2,
    pool_size: int = 8,
    seed: int = 0,
    resume: bool = False,
) -> str:
    """Run (or resume) self-play training, checkpointing and logging every generation.

    Unattended-friendly: a crash or stop loses at most the in-progress generation, and
    `resume=True` picks up from the newest checkpoint in `out_dir`, rebuilding the league
    pool from the most recent saved generations.
    """
    PPO, DummyVecEnv, VecMonitor = _lazy_imports()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pool = SelfPlayPool(max_size=pool_size, rng=rng)

    start_gen = 0
    model = None
    # Fixed early-self reference (gen_000) for a non-saturating progress metric. The
    # scripted-baseline win rate pins at ~100% once the bot is competent, so we also track
    # win rate vs. the earliest self-play checkpoint, which scales with the bot and can't max out.
    reference_opp = None
    gen0_path = os.path.join(out_dir, "gen_000.zip")

    if resume:
        existing = _find_checkpoints(out_dir)
        if existing:
            last_gen, last_path = existing[-1]
            start_gen = last_gen + 1
            print(f"Resuming from {last_path} (next generation = {start_gen})")
            model = PPO.load(last_path, device="cpu")
            # Rebuild the league pool from the most recent checkpoints.
            for _gen, path in existing[-pool_size:]:
                pool.add(FrozenPolicyOpponent.from_path(path))
            if os.path.exists(gen0_path):
                reference_opp = FrozenPolicyOpponent.from_path(gen0_path)

    if model is None:
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
            verbose=0,
            seed=seed,
        )

    for gen in range(start_gen, generations):
        warmup = gen < warmup_gens
        venv = VecMonitor(DummyVecEnv([make_env(pool, rng, warmup=warmup)]))
        model.set_env(venv)

        mode = "warmup" if warmup else "self-play"
        print(f"\n=== Generation {gen} ({mode}) ===", flush=True)
        t0 = time.time()
        model.learn(total_timesteps=steps_per_gen, reset_num_timesteps=False)
        gen_secs = time.time() - t0

        # Freeze this generation into the league pool, then checkpoint.
        pool.add(FrozenPolicyOpponent(_snapshot_bytes(model)))
        ckpt = os.path.join(out_dir, f"gen_{gen:03d}.zip")
        model.save(ckpt)

        # gen_000 becomes the fixed early-self reference for all later generations.
        if reference_opp is None and os.path.exists(gen0_path):
            reference_opp = FrozenPolicyOpponent.from_path(gen0_path)

        wr_baseline = evaluate(model, scripted_baseline, n_episodes=50, seed=seed + gen)
        wr_gen0 = None
        if gen > 0 and reference_opp is not None:
            wr_gen0 = round(evaluate(model, reference_opp, n_episodes=50, seed=seed + 9000 + gen), 4)

        record = {
            "generation": gen,
            "mode": mode,
            "win_rate_vs_baseline": round(wr_baseline, 4),
            "win_rate_vs_gen0": wr_gen0,
            "pool_size": len(pool),
            "total_timesteps": int(model.num_timesteps),
            "gen_seconds": round(gen_secs, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": os.path.basename(ckpt),
        }
        _log_progress(out_dir, record)
        gen0_str = f" | vs gen0: {wr_gen0:.0%}" if wr_gen0 is not None else ""
        print(
            f"[gen {gen}] win vs baseline: {wr_baseline:.2%}{gen0_str} | pool={len(pool)} "
            f"| {gen_secs:.0f}s | steps={model.num_timesteps}",
            flush=True,
        )

    final = os.path.join(out_dir, "final.zip")
    model.save(final)
    print(f"\nSaved final model to {final}", flush=True)
    return final


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="BladeVision PPO self-play (Milestone A)")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--steps-per-gen", type=int, default=200_000)
    parser.add_argument("--warmup-gens", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--out", type=str, default="runs/selfplay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from newest checkpoint in --out")
    parser.add_argument(
        "--forever",
        action="store_true",
        help="run effectively without limit (unattended daemon); stop with Ctrl-C / kill",
    )
    args = parser.parse_args(argv)

    generations = 10_000_000 if args.forever else args.generations
    train(
        generations=generations,
        steps_per_gen=args.steps_per_gen,
        out_dir=args.out,
        warmup_gens=args.warmup_gens,
        pool_size=args.pool_size,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
