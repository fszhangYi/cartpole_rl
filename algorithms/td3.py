"""TD3（Twin Delayed DDPG）— 连续力控制。

相对 DDPG 的三项改进：
    1) 双 Critic 取 min，抑制 Q 过估计
    2) 目标动作加截断噪声（target policy smoothing）
    3) 延迟策略更新（每 policy_delay 步才更新 Actor 与目标网）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common import ReplayBuffer, hard_update, soft_update
from algorithms.ddpg import Actor, Critic


class TD3Agent:
    """TD3 智能体。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        policy_delay: int = 2,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        start_learning: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.start_learning = start_learning
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor_tgt = Actor(obs_dim, act_dim).to(self.device)
        # 双 Q 网络
        self.critic1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim).to(self.device)
        self.critic1_tgt = Critic(obs_dim, act_dim).to(self.device)
        self.critic2_tgt = Critic(obs_dim, act_dim).to(self.device)
        hard_update(self.actor_tgt, self.actor)
        hard_update(self.critic1_tgt, self.critic1)
        hard_update(self.critic2_tgt, self.critic2)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.global_steps = 0
        self.learn_steps = 0

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """确定性策略 + 探索噪声（clip 到 [-1,1]）。"""
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            a = a.cpu().numpy()[0]
        if not greedy:
            a = a + np.random.normal(0, self.exploration_noise, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def observe(self, s, a, r, s2, done) -> dict:
        self.global_steps += 1
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, done)
        if len(self.buffer) < max(self.start_learning, self.batch_size):
            return {}
        return self.learn()

    def learn(self) -> dict:
        """更新双 Critic；每隔 policy_delay 步更新 Actor 与软目标。"""
        self.learn_steps += 1
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)
        with torch.no_grad():
            # 目标策略平滑：ã = clip(μ'(s') + ε, -1, 1)，ε ~ clip(N(0,σ), -c, c)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_tgt(s2) + noise).clamp(-1, 1)
            # 取两个目标 Critic 的较小值抑制过估计
            q_tgt = torch.min(self.critic1_tgt(s2, a2), self.critic2_tgt(s2, a2))
            y = r + self.gamma * q_tgt * (1 - d)
        q1, q2 = self.critic1(s, a), self.critic2(s, a)
        c_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()
        metrics = {"critic_loss": float(c_loss.item())}
        if self.learn_steps % self.policy_delay == 0:
            # 延迟策略：只用 critic1 提供策略梯度
            a_loss = -self.critic1(s, self.actor(s)).mean()
            self.a_opt.zero_grad()
            a_loss.backward()
            self.a_opt.step()
            soft_update(self.actor_tgt, self.actor, self.tau)
            soft_update(self.critic1_tgt, self.critic1, self.tau)
            soft_update(self.critic2_tgt, self.critic2, self.tau)
            metrics["actor_loss"] = float(a_loss.item())
        return metrics

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "c1": self.critic1.state_dict(), "c2": self.critic2.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["c1"])
        self.critic2.load_state_dict(ckpt["c2"])
        hard_update(self.actor_tgt, self.actor)
        hard_update(self.critic1_tgt, self.critic1)
        hard_update(self.critic2_tgt, self.critic2)
