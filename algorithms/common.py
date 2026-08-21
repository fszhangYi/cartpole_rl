"""算法共用组件：回放池、MLP 构造器、软/硬目标网络更新等。

被 DQN、DDPG、TD3、SAC 等深度强化学习实现复用，避免各文件重复造轮子。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU, output_activation=None) -> nn.Sequential:
    """按层宽列表构建全连接 MLP。

    Args:
        sizes: 各层神经元数，如 [obs_dim, 128, 128, act_dim]。
        activation: 隐层激活函数类（实例化时无参调用）。
        output_activation: 若不为 None，则在最后一层后追加该激活。

    Returns:
        nn.Sequential 网络；最后一层前不加隐层激活（便于输出 logits / Q）。
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # 中间层加激活；最后一层仅在指定 output_activation 时加
        if i < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """软更新目标网络：θ' ← (1-τ)θ' + τθ。

    τ 很小时目标网络缓慢跟踪在线网络，稳定 TD 目标（DDPG/TD3/SAC）。
    """
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            # 原地：tp = (1-tau)*tp + tau*sp
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """硬拷贝：将 source 的全部参数直接复制到 target（如 DQN 周期性同步）。"""
    target.load_state_dict(source.state_dict())


class ReplayBuffer:
    """均匀采样经验回放池（FIFO，容量用 deque.maxlen 限制）。

    存储转移 (s, a, r, s', done)，sample 时转为 torch.Tensor 供离策略学习。
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        # 每条：状态、动作（可为向量）、标量奖励、下一状态、done 标志
        self.buf: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, s, a, r, s2, done) -> None:
        """写入一条转移；满容量时自动丢弃最旧样本。"""
        self.buf.append(
            (
                np.asarray(s, dtype=np.float32),
                np.asarray(a, dtype=np.float32),
                float(r),
                np.asarray(s2, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        """均匀随机采样 batch，返回 (s, a, r, s2, d) 五个 Tensor。"""
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        # zip(*batch) 再 stack → 各字段堆成 batch 维
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(s),
            torch.as_tensor(a),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(s2),
            torch.as_tensor(d, dtype=torch.float32),
        )


class PrioritizedReplayBuffer:
    """简化版优先经验回放（PER）：按 TD 误差幅度优先采样，并校正重要性采样权重。

    供 rainbow-lite DQN 使用。采样概率 ∝ p_i^α；IS 权重 w_i ∝ (N·P(i))^(-β)。
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)  # 优先程度：0=均匀，1=完全按优先级
        self.buf: list = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0  # 环形写入指针

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, s, a, r, s2, done) -> None:
        """新样本以当前最大优先级入队，保证至少被采样一次。"""
        max_p = self.priorities.max() if self.buf else 1.0
        transition = (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(r),
            np.asarray(s2, dtype=np.float32),
            float(done),
        )
        if len(self.buf) < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """按优先级采样，并返回归一化 IS 权重与索引（供更新优先级）。

        Args:
            beta: 重要性采样校正强度，训练中常从 0.4 退火到 1.0。
        """
        n = len(self.buf)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        idxs = np.random.choice(n, batch_size, p=probs)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s2, d = map(np.stack, zip(*batch))
        # IS 权重：纠正优先采样带来的分布偏移，再按 batch 内最大值归一化
        weights = (n * probs[idxs]) ** (-beta)
        weights = weights / weights.max()
        return (
            torch.as_tensor(s),
            torch.as_tensor(a),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(s2),
            torch.as_tensor(d, dtype=torch.float32),
            torch.as_tensor(weights, dtype=torch.float32),
            idxs,
        )

    def update_priorities(self, idxs, td_errors) -> None:
        """用最新 |TD error| 更新对应样本优先级（+ε 避免零概率）。"""
        for i, err in zip(idxs, td_errors):
            self.priorities[int(i)] = abs(float(err)) + 1e-6
