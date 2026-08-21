"""智能体统一接口协议（Protocol）。

评估 / 可视化 / 训练循环只依赖本模块定义的 Agent 协议，
不关心底层是 PPO、Q-Learning、DDPG 等具体算法。

约定：
    - select_action: 根据观测选动作；greedy=True 时关闭探索
    - save / load: 持久化与恢复参数（路径由调用方决定）
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """所有可评估智能体应满足的结构化协议（duck typing）。

    用 @runtime_checkable 以便 isinstance(obj, Agent) 做运行时检查。
    """

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """根据观测选择动作。

        Args:
            obs: 环境观测向量（如 CartPole 的 4 维状态）。
            greedy: True 时不做随机探索（评估模式）。

        Returns:
            离散动作时为 int；连续动作算法（DDPG/TD3/SAC/MPC 等）
            实际可能返回 np.ndarray（力/力矩向量）。协议注解写 int
            仅为离散默认约定，调用方勿假定返回类型恒为标量。
        """

    def save(self, path: str) -> None:
        """将模型/表格参数保存到 path。"""
        ...

    def load(self, path: str) -> None:
        """从 path 加载模型/表格参数。"""
        ...
