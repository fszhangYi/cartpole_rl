#!/usr/bin/env python3
"""按配置加载 PPO / Q-Learning 权重并评估（复用 Agent.select_action）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algorithms import create_agent
from config_loader import ROOT, load_config, normalize_algorithm, resolve_checkpoint_path
from env import CartPoleMuJoCoEnv
from train import make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--algorithm", type=str, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="关闭探索（推荐验收）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.algorithm is not None:
        cfg["algorithm"] = normalize_algorithm(args.algorithm)

    env = make_env(cfg)
    agent = create_agent(cfg, obs_dim=4, act_dim=2)

    ckpt = args.checkpoint or resolve_checkpoint_path(cfg, "best")
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"找不到权重: {ckpt}")
    agent.load(str(ckpt))
    print(f"algorithm={cfg['algorithm']}  checkpoint={ckpt}")

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.select_action(obs, greedy=args.greedy)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
        print(f"episode {ep+1:02d}: return={ep_ret:.0f}")

    print(
        f"mean={np.mean(returns):.1f}  std={np.std(returns):.1f}  "
        f"min={np.min(returns):.0f}  max={np.max(returns):.0f}"
    )
    env.close()


if __name__ == "__main__":
    main()
