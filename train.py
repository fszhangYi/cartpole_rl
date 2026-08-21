#!/usr/bin/env python3
"""按 config.yaml 训练 CartPole；算法分支用 match/case，环境与存盘逻辑复用。"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithms import PPO, QLearningAgent, RolloutBatch, create_agent
from config_loader import (
    ROOT,
    load_config,
    normalize_algorithm,
    resolve_checkpoint_path,
)
from env import CartPoleMuJoCoEnv

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CartPole (PPO / Q-Learning)")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="覆盖配置文件中的 algorithm：ppo | q_learning",
    )
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def make_env(cfg: dict[str, Any]) -> CartPoleMuJoCoEnv:
    e = cfg.get("env") or {}
    angle_deg = float(e.get("angle_threshold_deg", 12.0))
    return CartPoleMuJoCoEnv(
        max_episode_steps=int(e.get("max_episode_steps", 500)),
        angle_threshold=angle_deg * np.pi / 180.0,
        x_threshold=float(e.get("x_threshold", 1.4)),
        force_mag=float(e.get("force_mag", 1.0)),
    )


def _train_total_steps(cfg: dict[str, Any], cli_total: int | None) -> int:
    if cli_total is not None:
        return cli_total
    match cfg["algorithm"]:
        case "q_learning":
            q = cfg.get("q_learning") or {}
            if "total_steps" in q:
                return int(q["total_steps"])
            return int((cfg.get("train") or {}).get("total_steps", 200_000))
        case "ppo":
            return int((cfg.get("train") or {}).get("total_steps", 80_000))
        case _:
            return int((cfg.get("train") or {}).get("total_steps", 80_000))


def train_ppo(
    env: CartPoleMuJoCoEnv,
    agent: PPO,
    cfg: dict[str, Any],
    total_steps: int,
    seed: int,
) -> tuple[list[dict], float]:
    """复用原 PPO rollout → GAE → update 流程。"""
    tcfg = cfg.get("train") or {}
    pcfg = cfg.get("ppo") or {}
    rollout_steps = int(pcfg.get("rollout_steps", 2048))
    solved_reward = float(tcfg.get("solved_reward", 475.0))
    solved_window = int(tcfg.get("solved_window", 20))
    save_dir = ROOT / str(tcfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    ep_lens: deque[int] = deque(maxlen=100)
    recent_for_solve: deque[float] = deque(maxlen=solved_window)
    history: list[dict] = []

    global_step = 0
    ep_ret = 0.0
    ep_len = 0
    best_avg = -1e9
    t0 = time.time()
    best_path = resolve_checkpoint_path(cfg, "best")

    print(f"[ppo] total_steps={total_steps} rollout={rollout_steps} device={pcfg.get('device', 'cpu')}")

    while global_step < total_steps:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        for _ in range(rollout_steps):
            action, logprob, value = agent.net.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logprob)
            rew_buf.append(float(reward))
            done_buf.append(bool(done))
            val_buf.append(value)

            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1
            global_step += 1

            if done:
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                recent_for_solve.append(ep_ret)
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_len = 0

            if global_step >= total_steps:
                break

        with torch.no_grad():
            _, last_value = agent.net.forward(torch.as_tensor(obs, dtype=torch.float32))
            last_value = float(last_value.item())

        advantages, returns = agent.compute_gae(rew_buf, done_buf, val_buf, last_value)
        batch = RolloutBatch(
            obs=torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32),
            actions=torch.as_tensor(np.asarray(act_buf), dtype=torch.int64),
            logprobs=torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32),
            rewards=torch.as_tensor(np.asarray(rew_buf), dtype=torch.float32),
            dones=torch.as_tensor(np.asarray(done_buf), dtype=torch.float32),
            values=torch.as_tensor(np.asarray(val_buf), dtype=torch.float32),
            advantages=torch.as_tensor(advantages, dtype=torch.float32),
            returns=torch.as_tensor(returns, dtype=torch.float32),
        )
        metrics = agent.update(batch)

        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        row = {"step": global_step, "avg_return_100": avg_ret, "avg_len_100": avg_len, **metrics}
        history.append(row)
        print(
            f"step={global_step:6d}  avgR100={avg_ret:7.1f}  avgLen={avg_len:6.1f}  "
            f"pi={metrics['policy_loss']:.3f}  v={metrics['value_loss']:.3f}  "
            f"H={metrics['entropy']:.3f}  t={time.time() - t0:.1f}s"
        )

        if avg_ret > best_avg and len(ep_returns) >= 10:
            best_avg = avg_ret
            agent.save(str(best_path))

        if len(recent_for_solve) == solved_window and np.mean(recent_for_solve) >= solved_reward:
            print(
                f"Solved: mean return over last {solved_window} eps "
                f"= {np.mean(recent_for_solve):.1f} >= {solved_reward}"
            )
            best_avg = float(np.mean(recent_for_solve))
            agent.save(str(best_path))
            break

    return history, best_avg


def train_q_learning(
    env: CartPoleMuJoCoEnv,
    agent: QLearningAgent,
    cfg: dict[str, Any],
    total_steps: int,
    seed: int,
) -> tuple[list[dict], float]:
    """逐步交互 + 即时 TD 更新（离策略表格 Q-Learning）。"""
    tcfg = cfg.get("train") or {}
    solved_reward = float(tcfg.get("solved_reward", 475.0))
    solved_window = int(tcfg.get("solved_window", 20))
    best_path = resolve_checkpoint_path(cfg, "best")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent_for_solve: deque[float] = deque(maxlen=solved_window)
    history: list[dict] = []

    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    td_abs_sum = 0.0
    log_every = 2000

    print(
        f"[q_learning] total_steps={total_steps} alpha={agent.alpha} "
        f"eps={agent.epsilon_start}→{agent.epsilon_end} bins={agent.n_bins}"
    )

    while global_step < total_steps:
        action = agent.select_action(obs, greedy=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        td_abs_sum += agent.update(obs, action, float(reward), next_obs, done)

        obs = next_obs
        ep_ret += float(reward)
        global_step += 1

        if done:
            ep_returns.append(ep_ret)
            recent_for_solve.append(ep_ret)
            obs, _ = env.reset()
            ep_ret = 0.0

        if global_step % log_every == 0 or global_step >= total_steps:
            avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
            mean_td = td_abs_sum / log_every
            td_abs_sum = 0.0
            row = {
                "step": global_step,
                "avg_return_100": avg_ret,
                "epsilon": agent.epsilon,
                "td_abs": mean_td,
            }
            history.append(row)
            print(
                f"step={global_step:6d}  avgR100={avg_ret:7.1f}  "
                f"eps={agent.epsilon:.3f}  td={mean_td:.3f}  t={time.time() - t0:.1f}s"
            )

            if avg_ret > best_avg and len(ep_returns) >= 10:
                best_avg = avg_ret
                agent.save(str(best_path))

            if (
                len(recent_for_solve) == solved_window
                and np.mean(recent_for_solve) >= solved_reward
            ):
                print(
                    f"Solved: mean return over last {solved_window} eps "
                    f"= {np.mean(recent_for_solve):.1f} >= {solved_reward}"
                )
                best_avg = float(np.mean(recent_for_solve))
                agent.save(str(best_path))
                break

    return history, best_avg


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
    agent = create_agent(cfg, obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n)

    print(f"config={cfg['_config_path']}  algorithm={cfg['algorithm']}")

    # ---------- 大改动集中处：按算法切换训练循环 ----------
    match cfg["algorithm"]:
        case "ppo":
            assert isinstance(agent, PPO)
            history, best_avg = train_ppo(env, agent, cfg, total_steps, seed)
        case "q_learning":
            assert isinstance(agent, QLearningAgent)
            history, best_avg = train_q_learning(env, agent, cfg, total_steps, seed)
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

    log_name = f"train_history_{cfg['algorithm']}.json"
    log_path = log_dir / log_name
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    env.close()
    print(f"Saved final → {final_path}")
    print(f"Saved best  → {best_path} (avgR={best_avg:.1f})")
    print(f"History     → {log_path}")


if __name__ == "__main__":
    main()
