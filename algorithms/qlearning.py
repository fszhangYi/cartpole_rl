"""表格 Q-Learning（连续状态离散化）— 经典离策略价值方法。

将 CartPole 连续 4 维状态分箱后维护 Q 表，用 ε-greedy 探索，
按 Bellman 最优方程做单步 TD 更新。是 SARSA / Dyna-Q / MC / DP 等
表格算法的基类实现。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class QLearningAgent:
    """ε-greedy + 一维/多维分箱后的 Q 表更新。

    更新公式（离策略，bootstrap 用 max）：
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') · (1-done) - Q(s,a) ]
    """

    def __init__(
        self,
        n_actions: int = 2,
        n_bins: Sequence[int] = (8, 8, 16, 16),
        state_low: Sequence[float] = (-1.4, -3.0, -0.22, -3.5),
        state_high: Sequence[float] = (1.4, 3.0, 0.22, 3.5),
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 100_000,
    ) -> None:
        self.n_actions = int(n_actions)
        self.n_bins = tuple(int(x) for x in n_bins)
        self.state_low = np.asarray(state_low, dtype=np.float64)
        self.state_high = np.asarray(state_high, dtype=np.float64)
        self.alpha = float(alpha)  # 学习率 α
        self.gamma = float(gamma)  # 折扣因子 γ
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.epsilon = self.epsilon_start
        self.global_steps = 0  # 用于线性退火 ε

        if len(self.n_bins) != 4 or self.state_low.shape != (4,) or self.state_high.shape != (4,):
            raise ValueError("n_bins / state_low / state_high 均需为长度 4")

        # Q 表形状：(bin0, bin1, bin2, bin3, n_actions)
        self.q_table = np.zeros(self.n_bins + (self.n_actions,), dtype=np.float64)

    def _anneal_epsilon(self) -> None:
        """按全局步数将 ε 从 start 线性降到 end。"""
        if self.epsilon_decay_steps <= 0:
            self.epsilon = self.epsilon_end
            return
        t = min(1.0, self.global_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def discretize(self, obs: np.ndarray) -> tuple[int, int, int, int]:
        """将连续观测映射到离散箱下标（先 clip 再线性分箱）。"""
        x = np.clip(np.asarray(obs, dtype=np.float64), self.state_low, self.state_high)
        # 线性映射到 [0, n_bins-1]：ratio ∈ [0,1] → idx = floor(ratio * n)
        ratios = (x - self.state_low) / (self.state_high - self.state_low + 1e-12)
        idxs = []
        for i, n in enumerate(self.n_bins):
            idx = int(ratios[i] * n)
            idx = min(max(idx, 0), n - 1)  # 边界 ratio=1 时防越界
            idxs.append(idx)
        return idxs[0], idxs[1], idxs[2], idxs[3]

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy：以 ε 概率随机，否则取 argmax_a Q(s,a)。"""
        s = self.discretize(obs)
        if (not greedy) and (np.random.random() < self.epsilon):
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self.q_table[s]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> float:
        """单步 TD 更新，返回 |TD error|（便于日志）。

        target = r + γ max_a' Q(s',a')  （终端则无 bootstrap）
        """
        s = self.discretize(obs)
        s2 = self.discretize(next_obs)
        q_sa = self.q_table[s][action]
        target = reward
        if not done:
            # 离策略：用下一状态最优动作价值，而非行为策略实际动作
            target += self.gamma * float(np.max(self.q_table[s2]))
        td_error = target - q_sa
        self.q_table[s][action] = q_sa + self.alpha * td_error

        self.global_steps += 1
        self._anneal_epsilon()
        return float(abs(td_error))

    def save(self, path: str) -> None:
        """压缩保存 Q 表与离散化/退火超参。"""
        np.savez_compressed(
            path,
            q_table=self.q_table,
            n_bins=np.asarray(self.n_bins),
            state_low=self.state_low,
            state_high=self.state_high,
            alpha=np.asarray([self.alpha]),
            gamma=np.asarray([self.gamma]),
            epsilon=np.asarray([self.epsilon]),
            epsilon_start=np.asarray([self.epsilon_start]),
            epsilon_end=np.asarray([self.epsilon_end]),
            epsilon_decay_steps=np.asarray([self.epsilon_decay_steps]),
            global_steps=np.asarray([self.global_steps]),
            algorithm=np.asarray(["q_learning"]),
        )

    def load(self, path: str) -> None:
        """从 npz 恢复 Q 表与超参；n_actions 由 Q 表最后一维推断。"""
        data = np.load(path, allow_pickle=False)
        self.q_table = data["q_table"]
        self.n_bins = tuple(int(x) for x in data["n_bins"].tolist())
        self.state_low = data["state_low"].astype(np.float64)
        self.state_high = data["state_high"].astype(np.float64)
        self.alpha = float(data["alpha"][0])
        self.gamma = float(data["gamma"][0])
        self.epsilon = float(data["epsilon"][0])
        self.epsilon_start = float(data["epsilon_start"][0])
        self.epsilon_end = float(data["epsilon_end"][0])
        self.epsilon_decay_steps = int(data["epsilon_decay_steps"][0])
        self.global_steps = int(data["global_steps"][0])
        self.n_actions = int(self.q_table.shape[-1])
