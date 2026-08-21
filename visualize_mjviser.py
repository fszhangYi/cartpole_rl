#!/usr/bin/env python3
"""Run trained CartPole policy in mjviser (browser viewer).

Default port: 6008.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import viser
from mjviser import Viewer

from env import DEFAULT_XML, CartPoleMuJoCoEnv
from ppo import PPO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CartPole PPO with mjviser")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cartpole_ppo_best.pt",
    )
    p.add_argument("--port", type=int, default=6008)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions; default is greedy (argmax)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = CartPoleMuJoCoEnv(xml_path=DEFAULT_XML, max_episode_steps=100_000)
    agent = PPO(obs_dim=4, act_dim=2)
    if args.checkpoint.exists():
        agent.load(str(args.checkpoint))
        print(f"Loaded policy: {args.checkpoint}")
    else:
        print(f"WARNING: checkpoint not found ({args.checkpoint}), using random policy")

    obs, _ = env.reset(seed=args.seed)
    model, data = env.model, env.data
    episode = 0
    ep_ret = 0.0

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """Replace default mj_step: apply policy action then step physics."""
        nonlocal obs, episode, ep_ret
        if args.stochastic:
            action, _, _ = agent.net.act(obs)
        else:
            action = agent.net.greedy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += float(reward)
        if terminated or truncated:
            print(f"[episode {episode}] return={ep_ret:.0f}")
            episode += 1
            ep_ret = 0.0
            obs, _ = env.reset()

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal obs, ep_ret
        obs, _ = env.reset()
        ep_ret = 0.0
        print("Reset from viewer GUI")

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"mjviser CartPole → http://localhost:{args.port}")
    Viewer(model, data, server=server, step_fn=step_fn, reset_fn=reset_fn).run()


if __name__ == "__main__":
    main()
