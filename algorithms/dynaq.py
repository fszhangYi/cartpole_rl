"""Dyna-Q：Q-Learning + 学到的表格环境模型做规划（planning）。

真实交互一步后，除直接 TD 更新外，再从模型中随机回放若干 (s,a)→(r,s')
做额外 Q 更新，用有限样本加速价值传播（相对纯采样更高效）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from algorithms.qlearning import QLearningAgent


class DynaQAgent(QLearningAgent):
    """Dyna-Q：直接 RL + 基于确定性表格模型的随机规划。"""

    def __init__(self, planning_steps: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.planning_steps = int(planning_steps)
        # 确定性模型：model[(s_tuple, a)] = (r, s2_tuple, done)
        self.model: Dict[Tuple[tuple, int], Tuple[float, tuple, bool]] = {}

    def update(self, obs, action, reward, next_obs, done) -> float:
        """先做真实一步 Q-Learning，再写入模型并做 planning_steps 次模拟更新。"""
        td = super().update(obs, action, reward, next_obs, done)
        s = self.discretize(obs)
        s2 = self.discretize(next_obs)
        # 用最近一次观测覆盖该 (s,a) 的模型（确定性假设）
        self.model[(s, int(action))] = (float(reward), s2, bool(done))

        keys = list(self.model.keys())
        if not keys:
            return td
        # 规划：均匀随机抽已见过的 (s,a)，用模型给出的转移做 Q-Learning 式更新
        for _ in range(self.planning_steps):
            (sp, ap) = keys[np.random.randint(0, len(keys))]
            rp, s2p, done_p = self.model[(sp, ap)]
            q_sa = self.q_table[sp][ap]
            # 与真实更新相同：target = r + γ max Q(s',·)（终端无 bootstrap）
            target = rp if done_p else rp + self.gamma * float(np.max(self.q_table[s2p]))
            self.q_table[sp][ap] = q_sa + self.alpha * (target - q_sa)
        return td

    def save(self, path: str) -> None:
        # 复用父类存 Q；表格模型可在下次交互中重建，不单独持久化
        super().save(path)

    def load(self, path: str) -> None:
        super().load(path)
        self.model = {}  # 加载后清空模型，需重新收集转移
