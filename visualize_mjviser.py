#!/usr/bin/env python3
"""按配置用 mjviser 可视化当前算法策略（默认端口见 config.yaml）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import viser
from mjviser import Viewer

from algorithms import create_agent
from config_loader import ROOT, load_config, normalize_algorithm, resolve_checkpoint_path
from env import DEFAULT_XML, CartPoleMuJoCoEnv
from train import make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CartPole with mjviser")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--algorithm", type=str, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.algorithm is not None:
        cfg["algorithm"] = normalize_algorithm(args.algorithm)

    viz = cfg.get("visualize") or {}
    port = int(args.port if args.port is not None else viz.get("port", 6008))
    host = str(args.host if args.host is not None else viz.get("host", "0.0.0.0"))

    # 可视化加长单局，复用 make_env 的阈值后再覆盖 max steps
    env = make_env(cfg)
    env.max_episode_steps = 100_000
    # 若 XML 路径需固定
    if env.xml_path != DEFAULT_XML:
        pass

    agent = create_agent(cfg, obs_dim=4, act_dim=2)
    ckpt = args.checkpoint or resolve_checkpoint_path(cfg, "best")
    if Path(ckpt).exists():
        agent.load(str(ckpt))
        print(f"Loaded [{cfg['algorithm']}] {ckpt}")
    else:
        print(f"WARNING: checkpoint not found ({ckpt}), using untrained agent")

    obs, _ = env.reset(seed=args.seed)
    model, data = env.model, env.data
    episode = 0
    ep_ret = 0.0
    greedy = not args.stochastic

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal obs, episode, ep_ret
        action = agent.select_action(obs, greedy=greedy)
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

    server = viser.ViserServer(host=host, port=port)
    print(f"mjviser [{cfg['algorithm']}] → http://localhost:{port}")
    Viewer(model, data, server=server, step_fn=step_fn, reset_fn=reset_fn).run()


if __name__ == "__main__":
    main()
