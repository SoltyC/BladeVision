"""Head-to-head evaluation between two trained checkpoints (or vs. the scripted baseline).

Usage:
    python -m policy.duel runs/selfplay/final.zip --vs scripted --episodes 200
    python -m policy.duel runs/selfplay/gen_019.zip --vs runs/selfplay/gen_000.zip
"""

from __future__ import annotations

import argparse

import numpy as np

from sim.gym_env import DuelEnv
from sim.opponents import scripted_baseline


def _load_opponent(spec: str):
    if spec == "scripted":
        return scripted_baseline
    if spec == "idle":
        from sim.opponents import idle

        return idle
    from stable_baselines3 import PPO

    model = PPO.load(spec, device="cpu")

    def _policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _policy


def run_duel(model_path: str, opponent_spec: str, episodes: int, seed: int) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    opponent = _load_opponent(opponent_spec)
    env = DuelEnv(opponent=opponent, seed=seed)

    wins = losses = draws = 0
    total_ticks = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            total_ticks += 1
            done = terminated or truncated
        if info["a_health"] > 0.0 >= info["b_health"]:
            wins += 1
        elif info["b_health"] > 0.0 >= info["a_health"]:
            losses += 1
        else:
            draws += 1

    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
        "avg_ticks": total_ticks / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BladeVision duel eval")
    parser.add_argument("model", help="path to the PPO checkpoint under test (player A)")
    parser.add_argument("--vs", default="scripted", help="'scripted', 'idle', or a checkpoint path")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    result = run_duel(args.model, args.vs, args.episodes, args.seed)
    print(f"{args.model}  vs  {args.vs}")
    print(
        f"  win {result['win_rate']:.2%}  "
        f"({result['wins']}W / {result['losses']}L / {result['draws']}D)  "
        f"avg {result['avg_ticks']:.0f} ticks/round"
    )


if __name__ == "__main__":
    main()
