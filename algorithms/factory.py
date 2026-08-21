"""智能体工厂：按 config["algorithm"] 用 match/case 创建对应实现。

训练入口只需传入完整配置字典与观测/动作维度，无需手写各算法构造参数。
未知算法名会抛出 ValueError。
"""

from __future__ import annotations

from typing import Any

from algorithms.a2c import A2CAgent
from algorithms.ddpg import DDPGAgent
from algorithms.dp import DPAgent
from algorithms.dqn import DQNAgent
from algorithms.dynaq import DynaQAgent
from algorithms.monte_carlo import MonteCarloAgent
from algorithms.mpc import MPCAgent
from algorithms.ppo import PPO
from algorithms.qlearning import QLearningAgent
from algorithms.reinforce import ReinforceAgent
from algorithms.sac import SACAgent
from algorithms.sarsa import SarsaAgent
from algorithms.td3 import TD3Agent
from algorithms.td_lambda import SarsaLambdaAgent
from algorithms.trpo import TRPOAgent


def _tabular_kwargs(section: dict) -> dict:
    """从配置段提取表格型算法共用超参（分箱、学习率、ε 退火等）。"""
    return dict(
        n_bins=section.get("n_bins", [8, 8, 16, 16]),
        state_low=section.get("state_low", [-1.4, -3.0, -0.22, -3.5]),
        state_high=section.get("state_high", [1.4, 3.0, 0.22, 3.5]),
        alpha=float(section.get("alpha", 0.1)),
        gamma=float(section.get("gamma", 0.99)),
        epsilon_start=float(section.get("epsilon_start", 1.0)),
        epsilon_end=float(section.get("epsilon_end", 0.02)),
        epsilon_decay_steps=int(section.get("epsilon_decay_steps", 100_000)),
    )


def create_agent(cfg: dict[str, Any], obs_dim: int, act_dim: int):
    """根据 cfg['algorithm'] 实例化智能体。

    Args:
        cfg: 全局配置，至少含 "algorithm"；各算法超参在同名子字典中。
        obs_dim: 观测维度（深度方法需要）。
        act_dim: 离散动作数或连续动作维。

    Returns:
        对应算法的 Agent 实例。
    """
    algo = cfg["algorithm"]
    # 默认设备取 train.device，各算法段可再覆盖
    device = str((cfg.get("train") or {}).get("device", "cpu"))

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
                device=str(p.get("device", device)),
            )
        case "q_learning":
            return QLearningAgent(n_actions=act_dim, **_tabular_kwargs(cfg.get("q_learning") or {}))
        case "sarsa":
            # 无独立 sarsa 段时回退到 q_learning 超参
            return SarsaAgent(n_actions=act_dim, **_tabular_kwargs(cfg.get("sarsa") or cfg.get("q_learning") or {}))
        case "sarsa_lambda":
            sec = cfg.get("sarsa_lambda") or cfg.get("q_learning") or {}
            return SarsaLambdaAgent(
                n_actions=act_dim,
                lam=float(sec.get("lam", 0.8)),  # λ：资格迹衰减
                **_tabular_kwargs(sec),
            )
        case "dyna_q":
            sec = cfg.get("dyna_q") or cfg.get("q_learning") or {}
            return DynaQAgent(
                n_actions=act_dim,
                planning_steps=int(sec.get("planning_steps", 10)),
                **_tabular_kwargs(sec),
            )
        case "monte_carlo":
            return MonteCarloAgent(n_actions=act_dim, **_tabular_kwargs(cfg.get("monte_carlo") or cfg.get("q_learning") or {}))
        case "value_iteration" | "policy_iteration":
            # 两种 DP 模式共用 DPAgent，用 mode=algo 区分
            sec = cfg.get(algo) or cfg.get("q_learning") or {}
            return DPAgent(
                n_actions=act_dim,
                mode=algo,
                vi_iters=int(sec.get("vi_iters", 200)),
                **_tabular_kwargs(sec),
            )
        case "dqn" | "double_dqn" | "dueling_dqn" | "rainbow":
            d = cfg.get("dqn") or {}
            return DQNAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                variant=algo if algo != "rainbow" else "rainbow",
                lr=float(d.get("lr", 1e-3)),
                gamma=float(d.get("gamma", 0.99)),
                batch_size=int(d.get("batch_size", 64)),
                buffer_size=int(d.get("buffer_size", 50_000)),
                start_learning=int(d.get("start_learning", 1000)),
                target_update=int(d.get("target_update", 500)),
                epsilon_start=float(d.get("epsilon_start", 1.0)),
                epsilon_end=float(d.get("epsilon_end", 0.05)),
                epsilon_decay_steps=int(d.get("epsilon_decay_steps", 50_000)),
                # rainbow 默认至少 3-step 回报
                n_step=int(d.get("n_step", 1 if algo != "rainbow" else 3)),
                device=str(d.get("device", device)),
            )
        case "reinforce":
            r = cfg.get("reinforce") or {}
            return ReinforceAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                lr=float(r.get("lr", 3e-3)),
                gamma=float(r.get("gamma", 0.99)),
                use_baseline=bool(r.get("use_baseline", True)),
                device=str(r.get("device", device)),
            )
        case "a2c":
            a = cfg.get("a2c") or {}
            return A2CAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                lr=float(a.get("lr", 3e-4)),
                gamma=float(a.get("gamma", 0.99)),
                ent_coef=float(a.get("ent_coef", 0.01)),
                vf_coef=float(a.get("vf_coef", 0.5)),
                device=str(a.get("device", device)),
            )
        case "trpo":
            t = cfg.get("trpo") or {}
            return TRPOAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                gamma=float(t.get("gamma", 0.99)),
                max_kl=float(t.get("max_kl", 0.01)),
                damping=float(t.get("damping", 0.1)),
                vf_lr=float(t.get("vf_lr", 1e-3)),
                device=str(t.get("device", device)),
            )
        case "ddpg":
            d = cfg.get("ddpg") or {}
            # 仅透传配置中已出现的键，其余用类默认值
            keys = {
                "lr_actor", "lr_critic", "gamma", "tau", "noise_std", "batch_size",
                "buffer_size", "start_learning",
            }
            return DDPGAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=str(d.get("device", device)),
                **{k: d[k] for k in keys if k in d},
            )
        case "td3":
            d = cfg.get("td3") or {}
            keys = {
                "lr", "gamma", "tau", "policy_noise", "noise_clip", "exploration_noise",
                "policy_delay", "batch_size", "buffer_size", "start_learning",
            }
            return TD3Agent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=str(d.get("device", device)),
                **{k: d[k] for k in keys if k in d},
            )
        case "sac":
            d = cfg.get("sac") or {}
            keys = {
                "lr", "gamma", "tau", "alpha", "batch_size", "buffer_size", "start_learning",
            }
            return SACAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=str(d.get("device", device)),
                **{k: d[k] for k in keys if k in d},
            )
        case "mpc":
            m = cfg.get("mpc") or {}
            return MPCAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                horizon=int(m.get("horizon", 8)),
                n_samples=int(m.get("n_samples", 64)),
                gamma=float(m.get("gamma", 0.99)),
            )
        case _:
            raise ValueError(f"create_agent: 未知算法 {algo!r}")
