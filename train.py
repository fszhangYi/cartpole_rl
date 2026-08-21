#!/usr/bin/env python3
"""按 config.yaml 训练；算法分支集中在 match/case。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithms import create_agent
from config_loader import (
    ROOT,
    IMPLEMENTED,
    load_config,
    needs_continuous_action,
    normalize_algorithm,
    resolve_checkpoint_path,
)
from env import CartPoleMuJoCoEnv
from training_loops import (
    train_a2c,
    train_dp,
    train_dqn_family,
    train_monte_carlo,
    train_mpc,
    train_offpolicy_continuous,
    train_ppo_impl,
    train_reinforce,
    train_tabular_online,
    train_trpo,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CartPole RL")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--algorithm", type=str, default=None, help=f"可选: {', '.join(IMPLEMENTED)}")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def make_env(cfg: dict[str, Any]) -> CartPoleMuJoCoEnv:
    e = cfg.get("env") or {}
    angle_deg = float(e.get("angle_threshold_deg", 12.0))
    continuous = needs_continuous_action(cfg["algorithm"])
    return CartPoleMuJoCoEnv(
        max_episode_steps=int(e.get("max_episode_steps", 500)),
        angle_threshold=angle_deg * np.pi / 180.0,
        x_threshold=float(e.get("x_threshold", 1.4)),
        force_mag=float(e.get("force_mag", 1.0)),
        continuous=continuous,
    )


def _train_total_steps(cfg: dict[str, Any], cli_total: int | None) -> int:
    if cli_total is not None:
        return cli_total
    algo = cfg["algorithm"]
    sec = cfg.get(algo) or {}
    if "total_steps" in sec:
        return int(sec["total_steps"])
    # 表格类默认更长
    if algo in {"q_learning", "sarsa", "sarsa_lambda", "dyna_q", "monte_carlo"}:
        return int((cfg.get("q_learning") or {}).get("total_steps", 200_000))
    return int((cfg.get("train") or {}).get("total_steps", 80_000))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.algorithm is not None:
        cfg["algorithm"] = normalize_algorithm(args.algorithm)
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed

    tcfg = cfg.setdefault("train", {})
    seed = int(tcfg.get("seed", 42))
    total_steps = _train_total_steps(cfg, args.total_steps)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(cfg)
    if cfg["algorithm"] in {"ddpg", "td3", "sac", "mpc"}:
        act_dim = int(np.prod(env.action_space.shape))
    else:
        act_dim = int(env.action_space.n)
    agent = create_agent(cfg, obs_dim=env.observation_space.shape[0], act_dim=act_dim)

    print(f"config={cfg['_config_path']}  algorithm={cfg['algorithm']}  continuous={env.continuous}")

    match cfg["algorithm"]:
        case "ppo":
            history, best_avg = train_ppo_impl(env, agent, cfg, total_steps, seed)
        case "q_learning" | "dyna_q":
            history, best_avg = train_tabular_online(env, agent, cfg, total_steps, seed, cfg["algorithm"])
        case "sarsa" | "sarsa_lambda":
            history, best_avg = train_tabular_online(env, agent, cfg, total_steps, seed, cfg["algorithm"])
        case "monte_carlo":
            history, best_avg = train_monte_carlo(env, agent, cfg, total_steps, seed)
        case "value_iteration" | "policy_iteration":
            history, best_avg = train_dp(env, agent, cfg, total_steps, seed)
        case "dqn" | "double_dqn" | "dueling_dqn" | "rainbow":
            history, best_avg = train_dqn_family(env, agent, cfg, total_steps, seed)
        case "reinforce":
            history, best_avg = train_reinforce(env, agent, cfg, total_steps, seed)
        case "a2c":
            history, best_avg = train_a2c(env, agent, cfg, total_steps, seed)
        case "trpo":
            history, best_avg = train_trpo(env, agent, cfg, total_steps, seed)
        case "ddpg" | "td3" | "sac":
            history, best_avg = train_offpolicy_continuous(
                env, agent, cfg, total_steps, seed, cfg["algorithm"]
            )
        case "mpc":
            history, best_avg = train_mpc(env, agent, cfg, total_steps, seed)
        case _:
            raise ValueError(cfg["algorithm"])

    save_dir = ROOT / str(tcfg.get("save_dir", "checkpoints"))
    log_dir = ROOT / str(tcfg.get("log_dir", "runs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    final_path = resolve_checkpoint_path(cfg, "final")
    best_path = resolve_checkpoint_path(cfg, "best")
    agent.save(str(final_path))
    agent.save(str(best_path))

    log_path = log_dir / f"train_history_{cfg['algorithm']}.json"
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    env.close()
    print(f"Saved final → {final_path}")
    print(f"Saved best  → {best_path} (avgR={best_avg:.1f})")
    print(f"History     → {log_path}")


if __name__ == "__main__":
    main()
