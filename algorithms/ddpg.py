"""DDPG（Deep Deterministic Policy Gradient）— 连续力控制。

确定性策略 μ(s) + 加性探索噪声；Critic 学习 Q(s,a)，
Actor 沿 ∇_a Q 提升动作。目标网络软更新稳定 TD 目标。
动作经 Tanh 压到 [-1, 1]（力归一化）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common import ReplayBuffer, hard_update, mlp, soft_update


class Actor(nn.Module):
    """确定性策略：s → a ∈ (-1,1)^{act_dim}。"""

    def __init__(self, obs_dim, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """动作价值：拼接 (s,a) → Q(s,a)。"""

    def __init__(self, obs_dim, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class DDPGAgent:
    """DDPG 智能体：离策略 Actor-Critic + 回放池。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        start_learning: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.tau = tau  # 软更新系数
        self.noise_std = noise_std  # 探索高斯噪声标准差
        self.batch_size = batch_size
        self.start_learning = start_learning
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor_tgt = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_tgt = Critic(obs_dim, act_dim).to(self.device)
        hard_update(self.actor_tgt, self.actor)
        hard_update(self.critic_tgt, self.critic)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer = ReplayBuffer(buffer_size)
        self.global_steps = 0

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """返回连续动作向量；非 greedy 时加高斯噪声并 clip 到 [-1,1]。"""
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            a = a.cpu().numpy()[0]
        if not greedy:
            a = a + np.random.normal(0, self.noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def observe(self, s, a, r, s2, done) -> dict:
        """存转移；缓冲足够后调用 learn。"""
        self.global_steps += 1
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, done)
        if len(self.buffer) < max(self.start_learning, self.batch_size):
            return {}
        return self.learn()

    def learn(self) -> dict:
        """Critic：MSE(Q, r+γ Q'(s', μ'(s')))；Actor：最大化 Q(s, μ(s))。"""
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)
        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            q_tgt = self.critic_tgt(s2, a2)
            y = r + self.gamma * q_tgt * (1 - d)  # Bellman 目标
        q = self.critic(s, a)
        c_loss = nn.functional.mse_loss(q, y)
        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()

        # 确定性策略梯度：−E[Q(s, μ(s))]
        a_loss = -self.critic(s, self.actor(s)).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()
        soft_update(self.actor_tgt, self.actor, self.tau)
        soft_update(self.critic_tgt, self.critic, self.tau)
        return {"critic_loss": float(c_loss.item()), "actor_loss": float(a_loss.item())}

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        hard_update(self.actor_tgt, self.actor)
        hard_update(self.critic_tgt, self.critic)
