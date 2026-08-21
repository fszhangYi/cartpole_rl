"""简化版离散 TRPO（共轭梯度 + 线搜索）。完整二阶 TRPO 的工程近似。

在 KL 约束下最大化代理目标：
    max_θ E[ (π_θ/π_old) A ]  s.t.  E[ KL(π_old || π_θ) ] ≤ δ
用 Fisher 信息矩阵向量积 + CG 求自然梯度，再线搜索保证 KL 与目标改进。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.ppo import ActorCritic


class TRPOAgent:
    """离散动作 TRPO；价值网络单独用 Adam 回归，策略用自然梯度更新。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        max_kl: float = 0.01,
        damping: float = 0.1,
        vf_lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.max_kl = max_kl  # 平均 KL 上界 δ
        self.damping = damping  # FVP 中的阻尼项，改善数值稳定性
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.vf_opt = optim.Adam(self.net.critic.parameters(), lr=vf_lr)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if greedy:
            return self.net.greedy(obs)
        a, _, _ = self.net.act(obs)
        return a

    def _flat_grads(self, loss, params):
        """将 loss 对 params 的梯度展平为一维向量。"""
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
        return torch.cat([g.reshape(-1) for g in grads])

    def _cg(self, Avp_fn, b, iters=10):
        """共轭梯度求解 Ax = b，其中 A 通过矩阵-向量积 Avp_fn 隐式给出。"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(iters):
            Avp = Avp_fn(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def update(self, obs, actions, advantages, returns, old_logprobs) -> dict:
        """一次 TRPO 更新：先拟合 V，再对 actor 做自然梯度 + 线搜索。"""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device)
        adv_t = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.asarray(old_logprobs), dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # 价值函数先回归（与策略更新解耦）
        for _ in range(5):
            _, v = self.net(obs_t)
            vloss = 0.5 * (v - ret_t).pow(2).mean()
            self.vf_opt.zero_grad()
            vloss.backward()
            self.vf_opt.step()

        params = list(self.net.actor.parameters())
        dist, _ = self.net(obs_t)
        logp = dist.log_prob(act_t)
        ratio = torch.exp(logp - old_logp)  # π_new / π_old
        surr = (ratio * adv_t).mean()  # 代理目标 L
        # 最大化 surr → 对 -surr 求梯度得到上升方向 g
        flat_g = self._flat_grads(-surr, params).detach()

        def fisher_vector_product(v):
            """计算 (F + λI) v，F 为平均 KL 的 Hessian（Fisher 信息）。"""
            dist2, _ = self.net(obs_t)
            # KL(old||new) 用当前策略相对 old logits：用 old_logp 近似
            # 简化：用当前分布的熵相关二阶；更稳的做法是固定 old_dist
            kl = torch.distributions.kl.kl_divergence(
                Categorical(logits=dist2.logits.detach()), Categorical(logits=dist2.logits)
            ).mean()
            # 上面 kl 对 logits 为 0；改用：采样 old probs
            old_dist = Categorical(logits=dist2.logits.detach())
            new_dist = Categorical(logits=dist2.logits)
            kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
            grads = torch.autograd.grad(kl, params, create_graph=True)
            flat_grad_kl = torch.cat([g.reshape(-1) for g in grads])
            # Hessian-vector：∇(∇KL · v)
            kl_v = (flat_grad_kl * v).sum()
            grads2 = torch.autograd.grad(kl_v, params, retain_graph=True)
            flat = torch.cat([g.contiguous().reshape(-1) for g in grads2])
            return flat + self.damping * v

        # 求解 F x = g 得自然梯度方向
        step_dir = self._cg(fisher_vector_product, flat_g)
        # 缩放使二次近似 KL ≈ δ：max_kl = 0.5 x^T F x → 步长因子
        shs = 0.5 * torch.dot(step_dir, fisher_vector_product(step_dir))
        lm = torch.sqrt(shs / (self.max_kl + 1e-8))
        full_step = step_dir / (lm + 1e-8)

        # 线搜索：从满步长起折半，直至 surrogate 提升且 KL 可接受
        with torch.no_grad():
            old_params = torch.cat([p.data.view(-1) for p in params])
        success = False
        for coef in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            new_flat = old_params + coef * full_step
            offset = 0
            for p in params:
                numel = p.numel()
                p.data.copy_(new_flat[offset : offset + numel].view_as(p))
                offset += numel
            dist_new, _ = self.net(obs_t)
            logp_new = dist_new.log_prob(act_t)
            ratio_new = torch.exp(logp_new - old_logp)
            surr_new = (ratio_new * adv_t).mean()
            old_dist = Categorical(logits=dist.logits.detach())
            kl = torch.distributions.kl.kl_divergence(old_dist, dist_new).mean()
            if surr_new > surr and kl <= self.max_kl * 1.5:
                success = True
                break
            # revert：不满足约束则恢复旧参数再试更小步长
            offset = 0
            for p in params:
                numel = p.numel()
                p.data.copy_(old_params[offset : offset + numel].view_as(p))
                offset += numel
        return {"surr": float(surr.item()), "trpo_ok": float(success)}

    def save(self, path: str) -> None:
        torch.save({"model": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model"])
