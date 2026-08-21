"""简单 MPC：学线性动力学，再用随机射击选动作（连续力）。

非神经网络模型预测控制：最小二乘拟合 s'≈ A s + B a + c，
在线对 horizon 内随机动作序列滚动预测，选累计启发式回报最高的首步动作。
适合教学演示，非工业级 MPC。
"""

from __future__ import annotations

import numpy as np


class MPCAgent:
    """
    在线规划器：用收集的 (s,a,s') 拟合 s'≈As+Ba+c，再 horizon 内随机采样动作序列最大化回报预测。
    不是经典 NN-MPC；适合教学演示。
    """

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        horizon: int = 8,
        n_samples: int = 64,
        gamma: float = 0.99,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon  # 规划时域长度
        self.n_samples = n_samples  # 随机射击候选序列数
        self.gamma = gamma
        # 线性动力学初值：近似恒等转移、零控制效应
        self.A = np.eye(obs_dim)
        self.B = np.zeros((obs_dim, act_dim))
        self.c = np.zeros(obs_dim)
        self.dataset_s: list = []
        self.dataset_a: list = []
        self.dataset_s2: list = []
        self.fitted = False

    def observe_transition(self, s, a, r, s2, done) -> None:
        """收集动力学拟合数据（奖励 r/done 不进入线性模型）。"""
        self.dataset_s.append(np.asarray(s, dtype=np.float64))
        self.dataset_a.append(np.asarray(a, dtype=np.float64).reshape(-1))
        self.dataset_s2.append(np.asarray(s2, dtype=np.float64))

    def fit_dynamics(self) -> None:
        """最小二乘：S2 ≈ [S, A, 1] θ，再拆出 A,B,c。样本不足 50 则跳过。"""
        if len(self.dataset_s) < 50:
            return
        S = np.stack(self.dataset_s)
        A = np.stack(self.dataset_a)
        S2 = np.stack(self.dataset_s2)
        # 设计矩阵 [s, a, 1]
        X = np.concatenate([S, A, np.ones((len(S), 1))], axis=1)
        # 最小二乘
        theta, _, _, _ = np.linalg.lstsq(X, S2, rcond=None)
        # theta 行对应 [s | a | 1] 的系数；转置得到 A,B 作用在列向量 s,a 上
        self.A = theta[: self.obs_dim].T
        self.B = theta[self.obs_dim : self.obs_dim + self.act_dim].T
        self.c = theta[-1]
        self.fitted = True

    def _predict(self, s, a):
        """一步线性预测：s' = A s + B a + c。"""
        return self.A @ s + self.B @ a + self.c

    def _rollout_return(self, s0, actions):
        """沿候选动作序列滚动，用存活启发式累计折扣回报。"""
        s = s0.copy()
        G = 0.0
        disc = 1.0
        for a in actions:
            s = self._predict(s, a)
            # 启发式：角度/位置惩罚 → 存活奖励近似（与 CartPole 失败阈值相近）
            x, _, theta, _ = s
            alive = 1.0 if (abs(x) < 1.4 and abs(theta) < 0.22) else 0.0
            G += disc * alive
            disc *= self.gamma
            if alive <= 0:
                break
        return G

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """随机射击 MPC：采样 n_samples 条动作序列，执行最优序列的第一步。

        未拟合前返回均匀随机力；greedy 参数保留接口一致性（本实现不区分）。
        """
        s0 = np.asarray(obs, dtype=np.float64)
        if not self.fitted:
            return np.array([np.random.uniform(-1, 1)], dtype=np.float32)
        best_a0 = None
        best_g = -1e9
        for _ in range(self.n_samples):
            actions = np.random.uniform(-1, 1, size=(self.horizon, self.act_dim))
            g = self._rollout_return(s0, actions)
            if g > best_g:
                best_g = g
                best_a0 = actions[0]
        return np.asarray(best_a0, dtype=np.float32)

    def save(self, path: str) -> None:
        """保存线性动力学参数。"""
        np.savez_compressed(path, A=self.A, B=self.B, c=self.c, fitted=np.array([self.fitted]))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.A, self.B, self.c = data["A"], data["B"], data["c"]
        self.fitted = bool(data["fitted"][0])
