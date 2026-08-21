"""SARSA：同策略（on-policy）表格 TD 控制。

与 Q-Learning 的区别：下一状态价值用行为策略实际选出的 a'，
而非 max_a' Q(s',a')，因此估计的是当前 ε-greedy 策略的 Q^π。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from algorithms.qlearning import QLearningAgent


class SarsaAgent(QLearningAgent):
    """在 QLearningAgent 基础上把更新改成同策略 TD。

    更新公式：
        Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]，a' ~ π(·|s')
    """

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> float:
        """单步 SARSA 更新；若未传入 next_action 则用当前 ε-greedy 现抽。

        注意：训练循环应传入与环境交互时已选好的 a'，以保持同策略一致性。
        """
        s = self.discretize(obs)
        s2 = self.discretize(next_obs)
        q_sa = self.q_table[s][action]
        target = reward
        if not done:
            if next_action is None:
                # 兜底：现场选 a'（可能与轨迹中真实下一步不完全一致）
                next_action = self.select_action(next_obs, greedy=False)
            # 同策略：bootstrap 用 Q(s', a')，不用 max
            target += self.gamma * float(self.q_table[s2][int(next_action)])
        td_error = target - q_sa
        self.q_table[s][action] = q_sa + self.alpha * td_error
        self.global_steps += 1
        self._anneal_epsilon()
        return float(abs(td_error))
