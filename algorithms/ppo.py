"""Minimal PPO (Actor-Critic) for discrete CartPole."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
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
        with torch.no_grad():
            dist, value = self.forward(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class PPO:
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
        self.clip_eps = clip_eps
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
                ratio = torch.exp(logprobs - old_logprobs[mb])
                adv = advantages[mb]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
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
