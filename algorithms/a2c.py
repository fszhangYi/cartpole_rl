"""A2C（同步 Actor-Critic）；A3C 见文档说明未实现异步多进程版。

在一段轨迹上用 n-step 回报估计优势 A = R - V(s)，同时优化：
    L = -E[logπ(a|s) A] + c_v ||V - R||^2 - c_e H[π]
复用 PPO 模块中的 ActorCritic 网络结构。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.ppo import ActorCritic


class A2CAgent:
    """同步优势 Actor-Critic（单环境 / 批量轨迹版）。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.ent_coef = ent_coef  # 熵奖励系数，鼓励探索
        self.vf_coef = vf_coef  # 价值损失权重
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if greedy:
            return self.net.greedy(obs)
        a, _, _ = self.net.act(obs)
        return a

    def update(self, obs, actions, rewards, dones, values, last_value) -> dict:
        """n-step 回报：对一段 trajectory 做 A2C 更新。

        从段末 bootstrap：R_T = last_value，再反向
            R_t = r_t + γ R_{t+1} · (1 - done_t)
        优势 A_t = R_t - V_t（采集时的旧价值）。
        """
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            # 终止则切断 bootstrap
            R = r + self.gamma * R * (0.0 if done else 1.0)
            returns.append(R)
        returns.reverse()

        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device)
        ret_t = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)
        val_t = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)
        adv = ret_t - val_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dist, values_pred = self.net(obs_t)
        logp = dist.log_prob(act_t)
        ent = dist.entropy().mean()
        # 策略：最大化 E[logπ A] → 最小化 -logπ A（A detach 不回传到旧 V）
        policy_loss = -(logp * adv.detach()).mean()
        value_loss = 0.5 * (values_pred - ret_t).pow(2).mean()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(ent.item()),
        }

    def save(self, path: str) -> None:
        torch.save({"model": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model"])
