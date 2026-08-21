"""各算法训练循环（供 train.py 的 match/case 调用）。"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import torch

from algorithms.a2c import A2CAgent
from algorithms.dqn import DQNAgent
from algorithms.monte_carlo import MonteCarloAgent
from algorithms.mpc import MPCAgent
from algorithms.ppo import PPO, RolloutBatch
from algorithms.reinforce import ReinforceAgent
from algorithms.trpo import TRPOAgent
from config_loader import resolve_checkpoint_path
from env import CartPoleMuJoCoEnv


def _stats_ctx(cfg, total_steps, seed, name):
    tcfg = cfg.get("train") or {}
    return {
        "solved_reward": float(tcfg.get("solved_reward", 475.0)),
        "solved_window": int(tcfg.get("solved_window", 20)),
        "best_path": resolve_checkpoint_path(cfg, "best"),
        "log_every": int(tcfg.get("log_every", 2000)),
        "total_steps": total_steps,
        "seed": seed,
        "name": name,
    }


def _maybe_solved(recent, window, thr) -> bool:
    return len(recent) == window and float(np.mean(recent)) >= thr


def train_ppo_impl(env: CartPoleMuJoCoEnv, agent: PPO, cfg, total_steps, seed):
    pcfg = cfg.get("ppo") or {}
    rollout_steps = int(pcfg.get("rollout_steps", 2048))
    ctx = _stats_ctx(cfg, total_steps, seed, "ppo")
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    print(f"[ppo] steps={total_steps} rollout={rollout_steps}")

    while global_step < total_steps:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        for _ in range(rollout_steps):
            action, logprob, value = agent.net.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_buf.append(obs); act_buf.append(action); logp_buf.append(logprob)
            rew_buf.append(float(reward)); done_buf.append(bool(done)); val_buf.append(value)
            obs = next_obs
            ep_ret += float(reward)
            global_step += 1
            if done:
                ep_returns.append(ep_ret); recent.append(ep_ret)
                obs, _ = env.reset(); ep_ret = 0.0
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
        avg = float(np.mean(ep_returns)) if ep_returns else 0.0
        history.append({"step": global_step, "avg_return_100": avg, **metrics})
        print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
        if avg > best_avg and len(ep_returns) >= 10:
            best_avg = avg; agent.save(str(ctx["best_path"]))
        if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
            best_avg = float(np.mean(recent)); agent.save(str(ctx["best_path"])); break
    return history, best_avg


def train_tabular_online(env, agent, cfg, total_steps, seed, kind: str):
    """Q / SARSA / SARSA(λ) / Dyna-Q。"""
    ctx = _stats_ctx(cfg, total_steps, seed, kind)
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    action = agent.select_action(obs, greedy=False)
    print(f"[{kind}] steps={total_steps}")

    while global_step < total_steps:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if kind in ("sarsa", "sarsa_lambda"):
            next_action = agent.select_action(next_obs, greedy=False) if not done else 0
            agent.update(obs, action, float(reward), next_obs, done, next_action=next_action)
            action = next_action if not done else agent.select_action(next_obs, greedy=False)
        else:
            agent.update(obs, action, float(reward), next_obs, done)
            action = agent.select_action(next_obs, greedy=False)

        obs = next_obs
        ep_ret += float(reward)
        global_step += 1
        if done:
            ep_returns.append(ep_ret); recent.append(ep_ret)
            obs, _ = env.reset(); ep_ret = 0.0
            action = agent.select_action(obs, greedy=False)

        if global_step % ctx["log_every"] == 0 or global_step >= total_steps:
            avg = float(np.mean(ep_returns)) if ep_returns else 0.0
            history.append({"step": global_step, "avg_return_100": avg, "epsilon": getattr(agent, "epsilon", 0)})
            print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
            if avg > best_avg and len(ep_returns) >= 10:
                best_avg = avg; agent.save(str(ctx["best_path"]))
            if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
                best_avg = float(np.mean(recent)); agent.save(str(ctx["best_path"])); break
    return history, best_avg


def train_monte_carlo(env, agent: MonteCarloAgent, cfg, total_steps, seed):
    ctx = _stats_ctx(cfg, total_steps, seed, "monte_carlo")
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    best_avg = -1e9
    t0 = time.time()
    print(f"[monte_carlo] steps={total_steps}")

    while global_step < total_steps:
        traj = []
        ep_ret = 0.0
        done = False
        while not done and global_step < total_steps:
            a = agent.select_action(obs, greedy=False)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            traj.append((obs, a, float(reward)))
            obs = next_obs
            ep_ret += float(reward)
            global_step += 1
        agent.update_episode(traj)
        ep_returns.append(ep_ret); recent.append(ep_ret)
        obs, _ = env.reset()
        if len(ep_returns) % 20 == 0:
            avg = float(np.mean(ep_returns))
            history.append({"step": global_step, "avg_return_100": avg})
            print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
            if avg > best_avg:
                best_avg = avg; agent.save(str(ctx["best_path"]))
            if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
                break
    return history, best_avg


def train_dp(env, agent, cfg, total_steps, seed):
    """先探索收集模型，再 value/policy iteration。"""
    ctx = _stats_ctx(cfg, total_steps, seed, cfg["algorithm"])
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    explore_steps = min(total_steps, int((cfg.get(cfg["algorithm"]) or {}).get("explore_steps", total_steps // 2)))
    obs, _ = env.reset(seed=seed)
    print(f"[{cfg['algorithm']}] explore={explore_steps} then plan")
    for step in range(explore_steps):
        a = agent.select_action(obs, greedy=False)
        next_obs, reward, terminated, truncated, _ = env.step(a)
        agent.observe_transition(obs, a, float(reward), next_obs, terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    agent.plan()
    agent.save(str(ctx["best_path"]))

    # 评估贪婪策略若干局写入 history
    returns = []
    for ep in range(20):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += float(r)
            done = term or trunc
        returns.append(ep_ret)
    avg = float(np.mean(returns))
    history = [{"step": explore_steps, "avg_return_100": avg, "phase": "after_plan"}]
    print(f"after plan greedy mean={avg:.1f}")
    return history, avg


def train_dqn_family(env, agent: DQNAgent, cfg, total_steps, seed):
    ctx = _stats_ctx(cfg, total_steps, seed, agent.variant)
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    print(f"[{agent.variant}] steps={total_steps}")

    while global_step < total_steps:
        a = agent.select_action(obs, greedy=False)
        next_obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        metrics = agent.observe(obs, a, float(reward), next_obs, done)
        obs = next_obs
        ep_ret += float(reward)
        global_step += 1
        if done:
            ep_returns.append(ep_ret); recent.append(ep_ret)
            obs, _ = env.reset(); ep_ret = 0.0
        if global_step % ctx["log_every"] == 0:
            avg = float(np.mean(ep_returns)) if ep_returns else 0.0
            row = {"step": global_step, "avg_return_100": avg, **metrics}
            history.append(row)
            print(f"step={global_step:6d} avgR100={avg:7.1f} eps={agent.epsilon:.3f} t={time.time()-t0:.1f}s")
            if avg > best_avg and len(ep_returns) >= 10:
                best_avg = avg; agent.save(str(ctx["best_path"]))
            if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
                best_avg = float(np.mean(recent)); agent.save(str(ctx["best_path"])); break
    return history, best_avg


def train_reinforce(env, agent: ReinforceAgent, cfg, total_steps, seed):
    ctx = _stats_ctx(cfg, total_steps, seed, "reinforce")
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    best_avg = -1e9
    t0 = time.time()
    print(f"[reinforce] steps={total_steps}")

    while global_step < total_steps:
        obs_l, act_l, rew_l = [], [], []
        ep_ret = 0.0
        done = False
        while not done and global_step < total_steps:
            a = agent.select_action(obs, greedy=False)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            obs_l.append(obs); act_l.append(a); rew_l.append(float(reward))
            obs = next_obs
            ep_ret += float(reward)
            global_step += 1
        metrics = agent.update_episode(obs_l, act_l, rew_l)
        ep_returns.append(ep_ret); recent.append(ep_ret)
        obs, _ = env.reset()
        if len(ep_returns) % 10 == 0:
            avg = float(np.mean(ep_returns))
            history.append({"step": global_step, "avg_return_100": avg, **metrics})
            print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
            if avg > best_avg:
                best_avg = avg; agent.save(str(ctx["best_path"]))
            if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
                break
    return history, best_avg


def train_a2c(env, agent: A2CAgent, cfg, total_steps, seed):
    n_steps = int((cfg.get("a2c") or {}).get("n_steps", 16))
    ctx = _stats_ctx(cfg, total_steps, seed, "a2c")
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    print(f"[a2c] steps={total_steps} n_steps={n_steps}")

    while global_step < total_steps:
        obs_b, act_b, rew_b, done_b, val_b = [], [], [], [], []
        for _ in range(n_steps):
            a, logp, v = agent.net.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            obs_b.append(obs); act_b.append(a); rew_b.append(float(reward))
            done_b.append(bool(done)); val_b.append(v)
            obs = next_obs
            ep_ret += float(reward)
            global_step += 1
            if done:
                ep_returns.append(ep_ret); recent.append(ep_ret)
                obs, _ = env.reset(); ep_ret = 0.0
            if global_step >= total_steps:
                break
        with torch.no_grad():
            _, last_v = agent.net.forward(torch.as_tensor(obs, dtype=torch.float32))
            last_v = 0.0 if done_b[-1] else float(last_v.item())
        metrics = agent.update(obs_b, act_b, rew_b, done_b, val_b, last_v)
        if global_step % max(n_steps * 20, 1) == 0:
            avg = float(np.mean(ep_returns)) if ep_returns else 0.0
            history.append({"step": global_step, "avg_return_100": avg, **metrics})
            print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
            if avg > best_avg and len(ep_returns) >= 10:
                best_avg = avg; agent.save(str(ctx["best_path"]))
            if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
                best_avg = float(np.mean(recent)); agent.save(str(ctx["best_path"])); break
    return history, best_avg


def train_trpo(env, agent: TRPOAgent, cfg, total_steps, seed):
    rollout = int((cfg.get("trpo") or {}).get("rollout_steps", 2048))
    ctx = _stats_ctx(cfg, total_steps, seed, "trpo")
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    print(f"[trpo] steps={total_steps}")

    while global_step < total_steps:
        obs_b, act_b, rew_b, done_b, logp_b, val_b = [], [], [], [], [], []
        for _ in range(rollout):
            a, logp, v = agent.net.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            obs_b.append(obs); act_b.append(a); rew_b.append(float(reward))
            done_b.append(bool(done)); logp_b.append(logp); val_b.append(v)
            obs = next_obs; ep_ret += float(reward); global_step += 1
            if done:
                ep_returns.append(ep_ret); recent.append(ep_ret)
                obs, _ = env.reset(); ep_ret = 0.0
            if global_step >= total_steps:
                break
        # GAE 简易：用回报 - value
        returns = []
        R = 0.0
        for r, d in zip(reversed(rew_b), reversed(done_b)):
            R = r + agent.gamma * R * (0.0 if d else 1.0)
            returns.append(R)
        returns.reverse()
        adv = np.asarray(returns, dtype=np.float32) - np.asarray(val_b, dtype=np.float32)
        metrics = agent.update(obs_b, act_b, adv, returns, logp_b)
        avg = float(np.mean(ep_returns)) if ep_returns else 0.0
        history.append({"step": global_step, "avg_return_100": avg, **metrics})
        print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
        if avg > best_avg and len(ep_returns) >= 10:
            best_avg = avg; agent.save(str(ctx["best_path"]))
        if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
            best_avg = float(np.mean(recent)); agent.save(str(ctx["best_path"])); break
    return history, best_avg


def train_offpolicy_continuous(env, agent, cfg, total_steps, seed, name: str):
    ctx = _stats_ctx(cfg, total_steps, seed, name)
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    ep_returns: deque[float] = deque(maxlen=100)
    recent: deque[float] = deque(maxlen=ctx["solved_window"])
    history = []
    global_step = 0
    ep_ret = 0.0
    best_avg = -1e9
    t0 = time.time()
    print(f"[{name}] steps={total_steps} (continuous)")

    while global_step < total_steps:
        a = agent.select_action(obs, greedy=False)
        next_obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        metrics = agent.observe(obs, a, float(reward), next_obs, done)
        obs = next_obs
        ep_ret += float(reward)
        global_step += 1
        if done:
            ep_returns.append(ep_ret); recent.append(ep_ret)
            obs, _ = env.reset(); ep_ret = 0.0
        if global_step % ctx["log_every"] == 0:
            avg = float(np.mean(ep_returns)) if ep_returns else 0.0
            history.append({"step": global_step, "avg_return_100": avg, **metrics})
            print(f"step={global_step:6d} avgR100={avg:7.1f} t={time.time()-t0:.1f}s")
            if avg > best_avg and len(ep_returns) >= 10:
                best_avg = avg; agent.save(str(ctx["best_path"]))
            if _maybe_solved(recent, ctx["solved_window"], ctx["solved_reward"]):
                best_avg = float(np.mean(recent)); agent.save(str(ctx["best_path"])); break
    return history, best_avg


def train_mpc(env, agent: MPCAgent, cfg, total_steps, seed):
    ctx = _stats_ctx(cfg, total_steps, seed, "mpc")
    ctx["best_path"].parent.mkdir(parents=True, exist_ok=True)
    explore = min(total_steps // 2, int((cfg.get("mpc") or {}).get("explore_steps", 5000)))
    obs, _ = env.reset(seed=seed)
    print(f"[mpc] collect={explore} then fit+plan")
    for _ in range(explore):
        a = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(a)
        agent.observe_transition(obs, a, float(reward), next_obs, terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    agent.fit_dynamics()
    agent.save(str(ctx["best_path"]))

    returns = []
    for ep in range(10):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += float(r)
            done = term or trunc
        returns.append(ep_ret)
    avg = float(np.mean(returns))
    print(f"[mpc] after fit mean return={avg:.1f}")
    return [{"step": explore, "avg_return_100": avg}], avg
