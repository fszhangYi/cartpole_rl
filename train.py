#!/usr/bin/env python3
"""Train PPO on MuJoCo CartPole (平衡车)."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from env import CartPoleMuJoCoEnv
from ppo import PPO, RolloutBatch

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO CartPole with MuJoCo")
    p.add_argument("--total-steps", type=int, default=80_000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--log-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--solved-reward", type=float, default=475.0, help="avg return to early-stop")
    p.add_argument("--solved-window", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CartPoleMuJoCoEnv()
    obs, _ = env.reset(seed=args.seed)
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        lr=args.lr,
        device=args.device,
    )

    ep_returns: deque[float] = deque(maxlen=100)
    ep_lens: deque[int] = deque(maxlen=100)
    recent_for_solve: deque[float] = deque(maxlen=args.solved_window)
    history = []

    global_step = 0
    ep_ret = 0.0
    ep_len = 0
    best_avg = -1e9
    t0 = time.time()

    print(
        f"Train CartPole | total_steps={args.total_steps} rollout={args.rollout_steps} "
        f"device={args.device}"
    )

    while global_step < args.total_steps:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        for _ in range(args.rollout_steps):
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

            if global_step >= args.total_steps:
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
        row = {
            "step": global_step,
            "avg_return_100": avg_ret,
            "avg_len_100": avg_len,
            "episodes": len(ep_returns),
            **metrics,
        }
        history.append(row)
        elapsed = time.time() - t0
        print(
            f"step={global_step:6d}  avgR100={avg_ret:7.1f}  avgLen={avg_len:6.1f}  "
            f"pi={metrics['policy_loss']:.3f}  v={metrics['value_loss']:.3f}  "
            f"H={metrics['entropy']:.3f}  t={elapsed:.1f}s"
        )

        if avg_ret > best_avg and len(ep_returns) >= 10:
            best_avg = avg_ret
            best_path = args.save_dir / "cartpole_ppo_best.pt"
            agent.save(str(best_path))

        if len(recent_for_solve) == args.solved_window and np.mean(recent_for_solve) >= args.solved_reward:
            print(
                f"Solved: mean return over last {args.solved_window} eps "
                f"= {np.mean(recent_for_solve):.1f} >= {args.solved_reward}"
            )
            agent.save(str(args.save_dir / "cartpole_ppo_best.pt"))
            best_avg = float(np.mean(recent_for_solve))
            break

    final_path = args.save_dir / "cartpole_ppo_final.pt"
    agent.save(str(final_path))
    # Keep best in sync with final if final is at least as good
    agent.save(str(args.save_dir / "cartpole_ppo_best.pt"))
    log_path = args.log_dir / "train_history.json"
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    env.close()
    print(f"Saved final → {final_path}")
    print(f"Saved best  → {args.save_dir / 'cartpole_ppo_best.pt'} (avgR={best_avg:.1f})")
    print(f"History     → {log_path}")

if __name__ == "__main__":
    main()
