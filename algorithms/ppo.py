"""最小 PPO（Actor-Critic）实现，面向离散 CartPole。

近端策略优化：用裁剪重要性采样比约束策略更新幅度，
配合 GAE(λ) 估计优势，多 epoch 小批量重复利用同一段 rollout。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """正交初始化权重（策略输出层常用小 std），偏置置常数。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """共享观测、分离的策略头（logits）与价值头 V(s)。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),  # 小初始化 → 近均匀策略
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return Categorical(logits=logits), value

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """采样动作，并返回 logπ(a|s) 与 V(s)（供 GAE / PPO ratio）。"""
        with torch.no_grad():
            dist, value = self.forward(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def greedy(self, obs: np.ndarray) -> int:
        """评估用：取最大 logit 对应动作。"""
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())


@dataclass
class RolloutBatch:
    """一段采集数据及 GAE 后的优势/回报，供 PPO.update 使用。"""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class PPO:
    """PPO-Clip：L = -min(rA, clip(r)A) + c_v MSE(V,R) - c_e H。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps  # 重要性比裁剪半径 ε
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

    def compute_gae(
        self,
        rewards: List[float],
        dones: List[bool],
        values: List[float],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """广义优势估计 GAE(λ)。

        δ_t = r_t + γ V_{t+1} (1-d_t) - V_t
        A_t = δ_t + γλ (1-d_t) A_{t+1}
        returns = A + V  （作价值回归目标）
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - float(dones[t])
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """多轮小批量 PPO-Clip 更新，返回平均损失指标。"""
        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        old_logprobs = batch.logprobs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = obs.shape[0]
        idxs = np.arange(n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                mb = idxs[start : start + self.minibatch_size]
                dist, values = self.net(obs[mb])
                logprobs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                # r_t(θ) = π_θ(a|s) / π_old(a|s)
                ratio = torch.exp(logprobs - old_logprobs[mb])
                adv = advantages[mb]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                # 取悲观下界；负号因要最小化
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - returns[mb]) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in metrics.items()}

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """统一 Agent 接口：供 evaluate / visualize 复用。"""
        if greedy:
            return self.net.greedy(obs)
        action, _, _ = self.net.act(obs)
        return action

    def save(self, path: str) -> None:
        torch.save({"model": self.net.state_dict(), "algorithm": "ppo"}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
