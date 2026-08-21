"""按 config.algorithm 用 match/case 创建智能体。"""

from __future__ import annotations

from typing import Any

from algorithms.ppo import PPO
from algorithms.qlearning import QLearningAgent


def create_agent(cfg: dict[str, Any], obs_dim: int, act_dim: int):
    """工厂：算法分支集中在此 switch（match/case）。"""
    algo = cfg["algorithm"]

    match algo:
        case "ppo":
            p = cfg.get("ppo") or {}
            return PPO(
                obs_dim=obs_dim,
                act_dim=act_dim,
                lr=float(p.get("lr", 3e-4)),
                gamma=float(p.get("gamma", 0.99)),
                gae_lambda=float(p.get("gae_lambda", 0.95)),
                clip_eps=float(p.get("clip_eps", 0.2)),
                ent_coef=float(p.get("ent_coef", 0.01)),
                vf_coef=float(p.get("vf_coef", 0.5)),
                max_grad_norm=float(p.get("max_grad_norm", 0.5)),
                update_epochs=int(p.get("update_epochs", 10)),
                minibatch_size=int(p.get("minibatch_size", 64)),
                device=str(p.get("device", "cpu")),
            )
        case "q_learning":
            q = cfg.get("q_learning") or {}
            return QLearningAgent(
                n_actions=act_dim,
                n_bins=q.get("n_bins", [8, 8, 16, 16]),
                state_low=q.get("state_low", [-1.4, -3.0, -0.22, -3.5]),
                state_high=q.get("state_high", [1.4, 3.0, 0.22, 3.5]),
                alpha=float(q.get("alpha", 0.1)),
                gamma=float(q.get("gamma", 0.99)),
                epsilon_start=float(q.get("epsilon_start", 1.0)),
                epsilon_end=float(q.get("epsilon_end", 0.02)),
                epsilon_decay_steps=int(q.get("epsilon_decay_steps", 100_000)),
            )
        case _:
            raise ValueError(f"create_agent: 未知算法 {algo!r}")
