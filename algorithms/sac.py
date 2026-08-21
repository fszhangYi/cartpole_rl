"""SAC（Soft Actor-Critic）— 连续力，对角高斯策略。

最大化熵正则化目标：E[Σ γ^t (r_t + α H(π(·|s_t)))]
策略为 tanh-squash 的高斯；双 Q + 软目标；固定温度 α（本实现不自动调 α）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from algorithms.common import ReplayBuffer, hard_update, soft_update
from algorithms.ddpg import Critic


# log σ 裁剪范围，防止数值爆炸/坍缩
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class SquashedGaussianActor(nn.Module):
    """输出均值与 log_std，采样 z~N(μ,σ) 后 a=tanh(z)；并修正 logπ。"""

    def __init__(self, obs_dim, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        if deterministic:
            z = mu  # 评估：取均值再 squash
        else:
            z = dist.rsample()  # 重参数化，可对 μ,σ 反传
        a = torch.tanh(z)
        # log π(a|s) = log N(z) - Σ log(1 - tanh(z)^2)  （Jacobian 修正）
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        return a, logp


class SACAgent:
    """SAC：熵正则化离策略 Actor-Critic。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        start_learning: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.gamma, self.tau, self.alpha = gamma, tau, alpha  # α：温度 / 熵权重
        self.batch_size = batch_size
        self.start_learning = start_learning
        self.device = torch.device(device)
        self.actor = SquashedGaussianActor(obs_dim, act_dim).to(self.device)
        self.critic1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim).to(self.device)
        self.critic1_tgt = Critic(obs_dim, act_dim).to(self.device)
        self.critic2_tgt = Critic(obs_dim, act_dim).to(self.device)
        hard_update(self.critic1_tgt, self.critic1)
        hard_update(self.critic2_tgt, self.critic2)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.global_steps = 0

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """greedy=True 时用均值动作（deterministic）。"""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a, _ = self.actor(o, deterministic=greedy)
            return a.cpu().numpy()[0].astype(np.float32)

    def observe(self, s, a, r, s2, done) -> dict:
        self.global_steps += 1
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, done)
        if len(self.buffer) < max(self.start_learning, self.batch_size):
            return {}
        return self.learn()

    def learn(self) -> dict:
        """软 Bellman：y = r + γ (min Q' - α logπ)；策略：min α logπ - Q。"""
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            # 软状态价值：Ṽ = min_i Q_i' (s',a') - α log π(a'|s')
            q_tgt = torch.min(self.critic1_tgt(s2, a2), self.critic2_tgt(s2, a2)) - self.alpha * logp2
            y = r + self.gamma * q_tgt * (1 - d)
        q1, q2 = self.critic1(s, a), self.critic2(s, a)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()

        # 重采样当前策略动作，最大化 Q - α logπ（即最小化 α logπ - Q）
        a_new, logp = self.actor(s)
        q_new = torch.min(self.critic1(s, a_new), self.critic2(s, a_new))
        a_loss = (self.alpha * logp - q_new).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()
        soft_update(self.critic1_tgt, self.critic1, self.tau)
        soft_update(self.critic2_tgt, self.critic2, self.tau)
        return {"critic_loss": float(c_loss.item()), "actor_loss": float(a_loss.item())}

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "c1": self.critic1.state_dict(), "c2": self.critic2.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["c1"])
        self.critic2.load_state_dict(ckpt["c2"])
        hard_update(self.critic1_tgt, self.critic1)
        hard_update(self.critic2_tgt, self.critic2)
