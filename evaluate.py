#!/usr/bin/env python3
"""Evaluate a trained CartPole PPO policy (headless)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import CartPoleMuJoCoEnv
from ppo import PPO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cartpole_ppo_best.pt",
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="use argmax policy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = CartPoleMuJoCoEnv()
    agent = PPO(obs_dim=4, act_dim=2)
    agent.load(str(args.checkpoint))

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            if args.greedy:
                action = agent.net.greedy(obs)
            else:
                action, _, _ = agent.net.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
        print(f"episode {ep+1:02d}: return={ep_ret:.0f}")

    print(f"mean={np.mean(returns):.1f}  std={np.std(returns):.1f}  "
          f"min={np.min(returns):.0f}  max={np.max(returns):.0f}")
    env.close()


if __name__ == "__main__":
    main()
