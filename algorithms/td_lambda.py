"""SARSA(λ) / 资格迹：对应经典 TD(λ) 控制。

用资格迹 E 把 TD 误差沿最近访问的 (s,a) 回溯分配，λ∈[0,1] 在
单步 TD(0) 与 Monte Carlo 之间插值；本实现为累积迹（accumulating traces）。
"""

from __future__ import annotations

import numpy as np

from algorithms.qlearning import QLearningAgent


class SarsaLambdaAgent(QLearningAgent):
    """同策略 SARSA(λ)：δ 乘资格迹更新整张 Q 表，再按 γλ 衰减迹。"""

    def __init__(self, lam: float = 0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lam = float(lam)  # λ：迹衰减；0≈TD(0)，1≈接近 MC
        # 资格迹 E 与 Q 同形；episode 结束时清零
        self.E = np.zeros_like(self.q_table)

    def reset_traces(self) -> None:
        """回合结束时重置资格迹，避免跨 episode 错误归因。"""
        self.E.fill(0.0)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> float:
        """一步 SARSA(λ) 更新。

        δ = r + γ Q(s',a') - Q(s,a)   （done 时无 bootstrap）
        E(s,a) ← E(s,a) + 1          （累积迹）
        Q ← Q + α δ E
        E ← γ λ E
        """
        s = self.discretize(obs)
        s2 = self.discretize(next_obs)
        q_sa = self.q_table[s][action]
        if done:
            target = reward
        else:
            if next_action is None:
                next_action = self.select_action(next_obs, greedy=False)
            target = reward + self.gamma * float(self.q_table[s2][int(next_action)])
        delta = target - q_sa
        self.E[s][action] += 1.0  # 当前访问状态-动作对迹 +1
        # 所有带迹位置一并更新（误差沿历史访问回溯）
        self.q_table += self.alpha * delta * self.E
        self.E *= self.gamma * self.lam  # 迹衰减
        if done:
            self.reset_traces()
        self.global_steps += 1
        self._anneal_epsilon()
        return float(abs(delta))

    def load(self, path: str) -> None:
        super().load(path)
        # 加载后迹与 Q 表同形重新置零
        self.E = np.zeros_like(self.q_table)
