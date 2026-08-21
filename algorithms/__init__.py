"""强化学习算法包（algorithms）。

对外本包：
    对外暴露统一智能体协议 Agent、工厂 create_agent，以及常用实现
    （如 PPO、QLearningAgent），供训练脚本、评估与可视化按名称创建算法。

说明：
    具体算法实现分布在同目录各模块；本文件仅做再导出（re-export），
    不包含训练逻辑。
"""

from algorithms.base import Agent
from algorithms.factory import create_agent
from algorithms.ppo import PPO, RolloutBatch
from algorithms.qlearning import QLearningAgent

# 公开 API：外部应优先 from algorithms import ... 使用下列符号
__all__ = ["Agent", "PPO", "RolloutBatch", "QLearningAgent", "create_agent"]
