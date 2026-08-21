"""智能体统一接口：评估 / 可视化只依赖本协议，不关心 PPO 或 Q-Learning。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """返回离散动作；greedy=True 时不做探索。"""

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...
