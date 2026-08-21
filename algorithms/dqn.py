"""DQN / Double DQN / Dueling DQN / Rainbow-lite（离散动作）。

价值函数逼近 + 经验回放 + 目标网络。variant 控制是否启用：
    - double：在线网选动作、目标网估价值，减轻最大化偏差
    - dueling：V + A - mean(A) 分解
    - rainbow：上两者 + n-step + 简化 PER
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.common import PrioritizedReplayBuffer, ReplayBuffer, hard_update, mlp


class QNet(nn.Module):
    """状态 → 各离散动作 Q 值；可选 Dueling 结构。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, dueling: bool = False) -> None:
        super().__init__()
        self.dueling = dueling
        self.act_dim = act_dim
        if dueling:
            # 共享特征后分叉：标量 V(s) 与优势 A(s,a)
            self.feature = mlp([obs_dim, hidden, hidden])
            self.val = nn.Linear(hidden, 1)
            self.adv = nn.Linear(hidden, act_dim)
        else:
            self.net = mlp([obs_dim, hidden, hidden, act_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(x)
        h = self.feature(x)
        v = self.val(h)
        adv = self.adv(h)
        # Q = V + A - mean_a(A)，保证 A 可辨识（减去均值）
        return v + adv - adv.mean(dim=-1, keepdim=True)


class DQNAgent:
    """深度 Q 网络智能体（含若干增强变体）。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        variant: str = "dqn",  # dqn | double_dqn | dueling_dqn | rainbow
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        start_learning: int = 1000,
        target_update: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        n_step: int = 1,
        device: str = "cpu",
    ) -> None:
        self.variant = variant
        self.act_dim = act_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_learning = start_learning
        self.target_update = target_update  # 每多少 learn_steps 硬更新目标网
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.n_step = max(1, int(n_step))
        self.device = torch.device(device)
        self.global_steps = 0
        self.learn_steps = 0

        dueling = variant in ("dueling_dqn", "rainbow")
        double = variant in ("double_dqn", "dueling_dqn", "rainbow")
        self.double = double
        self.use_per = variant == "rainbow"
        if variant == "rainbow":
            self.n_step = max(self.n_step, 3)  # rainbow-lite 至少 3-step

        self.q = QNet(obs_dim, act_dim, dueling=dueling).to(self.device)
        self.q_tgt = QNet(obs_dim, act_dim, dueling=dueling).to(self.device)
        hard_update(self.q_tgt, self.q)
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer: ReplayBuffer | PrioritizedReplayBuffer
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.buffer = ReplayBuffer(buffer_size)
        # n-step 未满窗口前的临时转移缓冲
        self._nstep_buf: list = []

    def _anneal(self) -> None:
        """线性退火探索率 ε。"""
        t = min(1.0, self.global_steps / max(self.epsilon_decay_steps, 1))
        self.epsilon = self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy：探索时均匀随机，否则 argmax Q(s,·)。"""
        if (not greedy) and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.act_dim))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(torch.argmax(q, dim=-1).item())

    def _push_nstep(self, s, a, r, s2, done) -> None:
        """累积 n-step 转移；回合结束时冲刷剩余缓冲。"""
        self._nstep_buf.append((s, a, r, s2, done))
        if done:
            while self._nstep_buf:
                self._flush_nstep(force_len=len(self._nstep_buf))
            return
        if len(self._nstep_buf) >= self.n_step:
            self._flush_nstep(force_len=self.n_step)

    def _flush_nstep(self, force_len: int) -> None:
        """将缓冲前缀压成一条 n-step 样本：R = Σ_{i=0}^{n-1} γ^i r_i。"""
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(self._nstep_buf[:force_len]):
            R += (self.gamma**i) * r
            if d:
                break
        s0, a0, _, _, _ = self._nstep_buf[0]
        _, _, _, sn, dn = self._nstep_buf[force_len - 1]
        # 若中途 done，用实际终点状态与 done=True
        for j in range(force_len):
            if self._nstep_buf[j][4]:
                sn, dn = self._nstep_buf[j][3], True
                break
        self.buffer.push(s0, np.array([a0], dtype=np.float32), R, sn, float(dn))
        self._nstep_buf.pop(0)

    def observe(self, s, a, r, s2, done) -> dict:
        """环境一步：写入缓冲，样本足够后 learn，并周期性同步目标网。"""
        self.global_steps += 1
        self._anneal()
        self._push_nstep(s, a, r, s2, done)
        metrics = {}
        if len(self.buffer) >= max(self.start_learning, self.batch_size):
            metrics = self.learn()
        if self.learn_steps > 0 and self.learn_steps % self.target_update == 0:
            hard_update(self.q_tgt, self.q)
        return metrics

    def learn(self) -> dict:
        """从回放池采样，最小化加权 TD 平方误差。

        标准：y = r + γ^n max_a' Q_tgt(s',a')
        Double：a' = argmax Q_online(s')，再用 Q_tgt 取值。
        """
        self.learn_steps += 1
        if self.use_per:
            s, a, r, s2, d, w, idxs = self.buffer.sample(self.batch_size)  # type: ignore
            w = w.to(self.device)
        else:
            s, a, r, s2, d = self.buffer.sample(self.batch_size)
            w = torch.ones_like(r)
            idxs = None
        s, a, r, s2, d = s.to(self.device), a.to(self.device).long().view(-1), r.to(self.device), s2.to(self.device), d.to(self.device)

        # 当前 Q(s,a)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            gamma_n = self.gamma**self.n_step  # n-step 折扣
            if self.double:
                next_a = self.q(s2).argmax(dim=1)  # 在线网选 a'
                next_q = self.q_tgt(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.q_tgt(s2).max(dim=1).values
            # done 时切断 bootstrap
            target = r + gamma_n * next_q * (1.0 - d)
        td = target - q
        loss = (w * td.pow(2)).mean()  # PER 时用 IS 权重
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.use_per and idxs is not None:
            self.buffer.update_priorities(idxs, td.detach().cpu().numpy())  # type: ignore
        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        torch.save({"q": self.q.state_dict(), "variant": self.variant}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["q"])
        hard_update(self.q_tgt, self.q)
        self.epsilon = self.epsilon_end  # 加载后默认低探索，便于评估
