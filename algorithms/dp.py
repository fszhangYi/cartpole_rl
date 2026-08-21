"""离散化 MDP 上的动态规划：价值迭代 / 策略迭代。

先用环境交互统计经验转移与奖励（近似模型），再在表格 MDP 上做
value_iteration 或简化 policy_iteration，得到贪婪策略（执行时 ε=0）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from algorithms.qlearning import QLearningAgent


class DPAgent(QLearningAgent):
    """先随机/ε 探索收集转移计数，再 VI 或 PI；执行阶段 ε=0 贪婪。"""

    def __init__(self, mode: str = "value_iteration", vi_iters: int = 200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mode = mode  # "value_iteration" | "policy_iteration"
        self.vi_iters = int(vi_iters)
        # counts[(s,a)][s2] = 转移次数；reward_sum / reward_n 估计 r(s,a)
        self.trans_counts: Dict[Tuple[tuple, int], Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self.reward_sum: Dict[Tuple[tuple, int], float] = defaultdict(float)
        self.reward_n: Dict[Tuple[tuple, int], int] = defaultdict(int)
        self.fitted = False

    def observe_transition(self, obs, action, reward, next_obs, done) -> None:
        """记录一次转移以构建经验模型（不做 Q 的在线 TD）。"""
        s = self.discretize(obs)
        s2 = self.discretize(next_obs)
        # done 时用吸收：自环并 reward 已在 env 给 0
        key = (s, int(action))
        self.trans_counts[key][s2] += 1
        self.reward_sum[key] += float(reward)
        self.reward_n[key] += 1
        self.global_steps += 1
        self._anneal_epsilon()

    def _expected_q(self, s, a) -> float:
        """在经验模型下计算 Q(s,a) 的一步期望备份。

        Q(s,a) ≈ r̄(s,a) + γ Σ_{s'} P̂(s'|s,a) max_{a'} Q(s',a')
        """
        key = (s, a)
        if self.reward_n[key] == 0:
            return 0.0
        r = self.reward_sum[key] / self.reward_n[key]
        counts = self.trans_counts[key]
        total = sum(counts.values())
        v = 0.0
        for s2, c in counts.items():
            # 经验转移概率 P̂(s'|s,a) = count / total；下一状态用 max Q 作 V
            v += (c / total) * float(np.max(self.q_table[s2]))
        return r + self.gamma * v

    def value_iteration(self) -> None:
        """价值迭代：反复对所有已见 (s,a) 做同步式期望备份 vi_iters 次。"""
        for _ in range(self.vi_iters):
            for key in list(self.reward_n.keys()):
                s, a = key
                self.q_table[s][a] = self._expected_q(s, a)
        self.fitted = True
        self.epsilon = 0.0  # 规划完成后纯贪婪执行

    def policy_iteration(self, eval_iters: int = 50, improve_rounds: int = 20) -> None:
        """简化策略迭代：交替「按当前贪婪策略评估」与改进。

        此处用多轮期望备份近似策略评估（V≈maxQ），再隐式改进为贪婪策略。
        """
        # 简单：交替「按当前贪婪策略评估 V≈maxQ」与改进（此处用 VI 近似）
        for _ in range(improve_rounds):
            for _ in range(eval_iters):
                for key in list(self.reward_n.keys()):
                    s, a = key
                    self.q_table[s][a] = self._expected_q(s, a)
        self.fitted = True
        self.epsilon = 0.0

    def plan(self) -> None:
        """根据 mode 调用 VI 或 PI。"""
        if self.mode == "policy_iteration":
            self.policy_iteration()
        else:
            self.value_iteration()

    def update(self, *args, **kwargs):
        """DP 流程为 observe_transition + plan，禁止逐步 TD update。"""
        raise RuntimeError("DPAgent 用 observe_transition + plan")
