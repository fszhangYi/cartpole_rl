"""REINFORCE（带可选状态价值 baseline 的蒙特卡洛策略梯度）。

回合结束后用折扣回报 G_t 估计回报；策略梯度：
    ∇J ≈ E[ ∇log π(a|s) · (G_t - b(s)) ]
baseline b(s) 不改变期望梯度但可降低方差；本实现同时用 MSE 拟合 b。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.common import mlp


class ReinforceAgent:
    """离散动作 REINFORCE；可选 MLP baseline。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = torch.device(device)
        # 策略头输出 logits → Categorical
        self.policy = mlp([obs_dim, 64, 64, act_dim]).to(self.device)
        self.baseline = mlp([obs_dim, 64, 1]).to(self.device) if use_baseline else None
        params = list(self.policy.parameters())
        if self.baseline is not None:
            params += list(self.baseline.parameters())
        self.opt = optim.Adam(params, lr=lr)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """greedy 时取 argmax logits，否则按 Categorical 采样。"""
        logits = self.policy(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        if greedy:
            return int(torch.argmax(logits).item())
        dist = Categorical(logits=logits)
        return int(dist.sample().item())

    def update_episode(self, obs_list, act_list, rew_list) -> dict:
        """用一整回合数据做一次策略（+baseline）更新。"""
        # 从后向前：G_t = r_{t+1} + γ G_{t+1}
        returns = []
        G = 0.0
        for r in reversed(rew_list):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        obs_t = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(act_list), dtype=torch.int64, device=self.device)

        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_t)

        if self.baseline is not None:
            b = self.baseline(obs_t).squeeze(-1)
            # 优势：G - b；b 用 detach 以免策略梯度穿过 baseline 目标
            adv = returns_t - b.detach()
            bl_loss = F_mse(b, returns_t)  # 拟合回报作为价值目标
        else:
            adv = returns_t
            bl_loss = torch.tensor(0.0, device=self.device)
            # 标准化
        # 标准化优势，稳定尺度
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # 策略损失为 -E[logπ · A]；加上 0.5 * baseline MSE
        loss = -(logp * adv).mean() + 0.5 * bl_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.item())}

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict(), "baseline": None if self.baseline is None else self.baseline.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        if self.baseline is not None and ckpt.get("baseline") is not None:
            self.baseline.load_state_dict(ckpt["baseline"])


def F_mse(pred, target):
    """均方误差封装，供 baseline 回归使用。"""
    return nn.functional.mse_loss(pred, target)
