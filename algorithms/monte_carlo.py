"""First-visit Monte Carlo 控制（ε-greedy 软策略）。

不 bootstrap：用完整回合回报 G_t 估计 Q；每个 (s,a) 在回合内首次出现时
用该时刻起的回报更新（First-visit）。方差大但无偏差（相对真回报）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Tuple

import numpy as np

from algorithms.qlearning import QLearningAgent


class MonteCarloAgent(QLearningAgent):
    """用完整回报更新 Q；仍复用离散化与 ε-greedy。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # 记录每个 (s,a) 历史上所有 first-visit 回报，取均值作为 Q
        self.returns: DefaultDict[Tuple[tuple, int], List[float]] = defaultdict(list)

    def update_episode(self, trajectory: list[tuple]) -> float:
        """整回合更新。trajectory: list of (obs, action, reward)。

        从后向前递推 G ← r + γG；仅首次访问的 (s,a) 写入 returns 并更新 Q。
        返回各首次访问 |ΔQ| 的均值，供日志。
        """
        # 从后向前算 G；首次访问更新
        G = 0.0
        visited = set()
        abs_td = 0.0
        for obs, action, reward in reversed(trajectory):
            G = float(reward) + self.gamma * G  # G_t = r_{t+1} + γ G_{t+1}
            s = self.discretize(obs)
            key = (s, int(action))
            if key in visited:
                continue  # every-visit 会再次更新；此处是 first-visit
            visited.add(key)
            self.returns[key].append(G)
            new_q = float(np.mean(self.returns[key]))  # 样本均值 ≈ Q(s,a)
            abs_td += abs(new_q - self.q_table[s][action])
            self.q_table[s][action] = new_q
        self.global_steps += len(trajectory)
        # 按本回合步数多次退火，使 ε 与交互步数对齐
        for _ in range(len(trajectory)):
            self._anneal_epsilon()
        return abs_td / max(len(visited), 1)

    def update(self, *args, **kwargs):
        """MC 不支持逐步 TD；请调用 update_episode。"""
        raise RuntimeError("MonteCarloAgent 请用 update_episode，不要逐步 update")
