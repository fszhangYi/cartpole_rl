# 经典强化学习算法详解与本仓库实现清单

本仓库在 **MuJoCo CartPole（平衡车）** 上实现多种经典 RL 算法。用 `config.yaml` 的 `algorithm:` 或命令行切换：

```bash
python train.py --algorithm ppo
python train.py --algorithm dqn
python train.py --algorithm sac    # 自动使用连续力环境
```

代码位置：`algorithms/`（算法）· `training_loops.py`（训练循环）· `algorithms/factory.py`（`match/case` 工厂）。

---

## 0. 读前共识：本环境在学什么

| 项 | 离散模式（默认） | 连续模式（`ddpg/td3/sac/mpc`） |
| --- | --- | --- |
| 状态 \(s\) | \([x,\dot x,\theta,\dot\theta]\) | 同左 |
| 动作 \(a\) | \(\{0,1\}\) 左/右恒力 | \(a\in[-1,1]\) 力比例 |
| 奖励 | 未倒且未出轨每步 +1，否则当步 0 | 同左 |
| 目标 | 尽量撑满一局（默认最长 500 步） | 同左 |

连续状态上的**表格方法**必须先把 \(s\) 分箱（离散化），这是近似，不是精确 MDP。

---

## 1. 已实现算法详解

### 1.1 Q-Learning（`q_learning`）

**类别：** 离策略 · 基于价值 · 表格方法  

**核心思想：** 学习动作价值 \(Q(s,a)\)，用「下一状态的最大 Q」做备份（即使实际没执行那个动作），因此是 **off-policy**。

**更新公式：**

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[r+\gamma\max_{a'}Q(s',a')\cdot(1-\mathrm{done})-Q(s,a)\bigr]
\]

**行为策略：** \(\varepsilon\)-greedy（以 \(\varepsilon\) 概率随机探索）。  

**本仓库要点：**

- 文件：`algorithms/qlearning.py`
- 四维连续状态按 `n_bins` 线性分箱；\(\varepsilon\) 随训练步数线性退火
- 适合理解 TD 与探索；分箱粗时难满 500 分，通常需很多步

**与 SARSA 区别：** 目标里用 \(\max_{a'}Q\)，不依赖下一步真实动作。

---

### 1.2 SARSA（`sarsa`）

**类别：** 同策略 · 基于价值 · 表格方法  

**核心思想：** 备份时使用**实际将要执行**的下一动作 \(a'\)，因此学的是「当前 \(\varepsilon\)-greedy 策略」的价值，是 **on-policy**。

**更新公式：**

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[r+\gamma Q(s',a')\cdot(1-\mathrm{done})-Q(s,a)\bigr]
\]

其中 \(a'\sim\pi(\cdot|s')\)（本实现中在 `update` 时传入 `next_action`）。

**本仓库要点：**

- 文件：`algorithms/sarsa.py`（继承 `QLearningAgent`，只改更新）
- 训练循环必须「先选 \(a'\) 再更新」，见 `training_loops.train_tabular_online`

**直觉：** 比 Q-Learning 更「保守」——会把探索时的随机动作也算进价值，有时在有危险动作的环境更稳。

---

### 1.3 SARSA(λ) / TD(λ) 控制（`sarsa_lambda`）

**类别：** 同策略 · 资格迹 · 表格方法  

**核心思想：** 一步 TD 只改当前 \((s,a)\)；资格迹让**最近访问过**的状态-动作也对误差 \(\delta\)「负责」，等价于多步回报的高效实现。

**要点公式：**

\[
\delta_t=r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)
\]

\[
E(s,a)\leftarrow \gamma\lambda E(s,a),\quad
E(s_t,a_t)\leftarrow E(s_t,a_t)+1
\]

\[
Q\leftarrow Q+\alpha\,\delta\,E
\]

\(\lambda\in[0,1]\)：\(\lambda=0\) 退化为一步 SARSA；\(\lambda\to 1\) 接近蒙特卡洛。

**本仓库要点：**

- 文件：`algorithms/td_lambda.py`
- 配置项 `lam`（默认 0.8）；episode 结束清空迹 \(E\)

**说明：** 表格项里的「TD(0)」是更新规则；控制问题请用 `sarsa`（λ=0）或本算法。

---

### 1.4 Dyna-Q（`dyna_q`）

**类别：** 基于模型 · 规划 + Q-Learning  

**核心思想：** 与环境交互时既更新 Q，也记住转移 \((s,a)\mapsto(r,s')\)；再从记忆里**随机回放**做额外 Q 更新（planning），提高样本效率。

**流程：**

1. 真实一步：Q-Learning 更新 + 写入模型  
2. 重复 `planning_steps` 次：从模型采样 \((s,a)\)，用模型给出的 \(r,s'\) 做 Q 更新  

**本仓库要点：**

- 文件：`algorithms/dynaq.py`
- 模型为确定性字典（最后一次观测覆盖）；教学用途，非概率模型

---

### 1.5 蒙特卡洛控制（`monte_carlo`）

**类别：** 无模型 · 完整回合回报  

**核心思想：** 不使用 bootstrapping；等一整局结束后，用实际回报 \(G_t=\sum_{k\ge t}\gamma^{k-t}r_k\) 更新 Q。本实现为 **First-visit**：每个 \((s,a)\) 每局只在第一次访问时更新。

\[
Q(s,a)\leftarrow \mathrm{mean}\{\text{所有首次访问得到的 }G\}
\]

**本仓库要点：**

- 文件：`algorithms/monte_carlo.py`
- 必须整局结束才 `update_episode`；方差大、收敛慢，但无偏（在给定策略下）

---

### 1.6 价值迭代 / 策略迭代（`value_iteration` / `policy_iteration`）

**类别：** 动态规划（需模型）  

**核心思想：** 若已知 \(P(s'|s,a)\) 与奖励，可用贝尔曼最优方程迭代求解。本仓库**没有解析模型**，因此：

1. 阶段 A：\(\varepsilon\)-greedy 探索，用计数估计转移与平均奖励  
2. 阶段 B：在估计的离散 MDP 上做 VI 或 PI，得到 Q 表  
3. 执行：\(\varepsilon=0\) 贪婪  

**价值迭代：** 反复对所有 \((s,a)\)  

\[
Q(s,a)\leftarrow \hat r(s,a)+\gamma\sum_{s'}\hat P(s'|s,a)\max_{a'}Q(s',a')
\]

**策略迭代：** 在估计模型上交替「评估 / 改进」（本实现用多次 VI 式扫描近似）。

**本仓库要点：**

- 文件：`algorithms/dp.py`
- 配置 `explore_steps`、`vi_iters`；模型不准时规划质量差

---

### 1.7 DQN（`dqn`）

**类别：** 离策略 · 深度价值函数  

**核心思想：** 用神经网络 \(Q_\theta(s,\cdot)\) 近似 Q；两大稳定技巧：

1. **经验回放**：打乱样本，打破时间相关  
2. **目标网络** \(Q_{\theta^-}\)：定期从 \(\theta\) 硬拷贝，减轻自举目标漂移  

\[
L=\mathbb{E}\bigl[(r+\gamma\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a))^2\bigr]
\]

**本仓库要点：**

- 文件：`algorithms/dqn.py`（`variant="dqn"`）
- \(\varepsilon\)-greedy + Adam；CartPole 状态为向量，非像素

---

### 1.8 Double DQN（`double_dqn`）

**问题：** 普通 DQN 用同一网络既选动作又估价值，易**过高估计**。  

**改法：** 用在线网络选动作，用目标网络估值：

\[
y=r+\gamma Q_{\theta^-}\bigl(s',\arg\max_{a'}Q_\theta(s',a')\bigr)
\]

**本仓库：** 同一 `DQNAgent`，`variant="double_dqn"`。

---

### 1.9 Dueling DQN（`dueling_dqn`）

**思想：** 把 Q 拆成状态价值 \(V(s)\) 与优势 \(A(s,a)\)：

\[
Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a')
\]

利于「许多动作价值相近、主要差在状态好坏」的情形。本实现同时使用 Double 目标。

---

### 1.10 Rainbow 精简版（`rainbow`）

**原版 Rainbow** 集成：Double、Dueling、Prioritized Replay、n-step、NoisyNet、C51 等。  

**本仓库 `rainbow` 包含：**

| 组件 | 有无 |
| --- | --- |
| Double + Dueling | ✅ |
| n-step return（默认 3） | ✅ |
| Prioritized Experience Replay | ✅（比例优先） |
| NoisyNet / C51 | ❌ |

故称为 **rainbow-lite**。完整版见文末「未实现」。

---

### 1.11 REINFORCE（`reinforce`）

**类别：** 同策略 · 纯策略梯度  

**思想：** 用完整回报加权提高/压低 \(\log\pi(a|s)\)：

\[
\nabla J\approx\mathbb{E}\bigl[\nabla\log\pi_\theta(a_t|s_t)\,G_t\bigr]
\]

可选 **baseline** \(b(s)\)（本仓为可学习 \(V\)）减小方差：用 \(G_t-b(s_t)\)。

**本仓库要点：**

- 文件：`algorithms/reinforce.py`
- 按 episode 更新；无信任域，步长大时易不稳

---

### 1.12 TRPO（`trpo`）

**类别：** 同策略 · 信任域策略优化  

**思想：** 在约束 \(\mathbb{E}[\mathrm{KL}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})]\le\delta\) 下最大化替代目标；用**共轭梯度**解约束二次型，再**线搜索**保证 KL 与性能。

**本仓库要点：**

- 文件：`algorithms/trpo.py`
- 离散 Categorical 的简化实现，便于教学；非工业级二阶全套细节
- PPO 可视为用 Clip 近似同一「别更新太猛」的思想

---

### 1.13 PPO（`ppo`）

**类别：** 同策略 · Actor-Critic  

**思想：** 用概率比 \(r_t(\theta)=\pi_\theta/\pi_{\mathrm{old}}\) 的 **Clip** 目标限制更新幅度；配合 **GAE** 估优势。详见教程 [CartPole_PPO_手把手教程.md](./CartPole_PPO_手把手教程.md) §1。

\[
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_t\hat A_t,\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat A_t)\big]
\]

**本仓库要点：**

- 文件：`algorithms/ppo.py`
- 本环境上已验证可稳定达到回报 500

---

### 1.14 A2C（`a2c`）

**类别：** 同步 Actor-Critic  

**思想：** 策略网络与价值网络同时学；用 n-step 回报减去 \(V(s)\) 得优势，更新 \(\log\pi\) 与价值回归，并加熵奖励鼓励探索。

**本仓库要点：**

- 文件：`algorithms/a2c.py`（复用 PPO 的 `ActorCritic` 骨干）
- **不是 A3C**：无异步多线程/进程

---

### 1.15 DDPG（`ddpg`）

**类别：** 离策略 · 连续动作 · 确定性策略  

**思想：** Actor 输出确定性动作 \(\mu_\theta(s)\)；Critic 学 \(Q(s,a)\)；用目标网络软更新；训练时对动作加高斯噪声探索。

\[
y=r+\gamma Q_{\phi'}(s',\mu_{\theta'}(s'))
\]

**本仓库：** 连续力 \(a\in[-1,1]\)；文件 `algorithms/ddpg.py`。

---

### 1.16 TD3（`td3`）

**在 DDPG 上的三项改进：**

1. **双 Critic**，取 min 减轻过高估计  
2. **延迟策略更新**（critic 多更几步再更 actor）  
3. **目标动作平滑**（目标动作加裁剪噪声）

文件：`algorithms/td3.py`。

---

### 1.17 SAC（`sac`）

**类别：** 最大熵 RL · 随机策略 · 连续动作  

**思想：** 最大化回报的同时最大化策略熵，鼓励探索、提高鲁棒性。Actor 为对角高斯再 `tanh` 压缩到 \([-1,1]\)；双 Q；软更新。

\[
J(\pi)=\mathbb{E}\big[\sum_t r_t+\alpha\mathcal H(\pi(\cdot|s_t))\big]
\]

文件：`algorithms/sac.py`（固定温度 \(\alpha\)，未做自动调温）。

---

### 1.18 MPC（`mpc`）教学版

**类别：** 基于模型的在线规划（非神经网络策略）  

**本仓库流程：**

1. 随机力采集转移，最小二乘拟合线性模型 \(s'\approx As+Ba+c\)  
2. 决策时：随机采样多条动作序列，用模型滚动预测启发式回报，取最优序列的**第一步**  

**局限：** 线性模型对非线性倒立摆很粗；启发式「存活」回报非真实最优控制。用于理解「先建模再规划」。

文件：`algorithms/mpc.py`。

---

## 2. 未实现算法及原因

| 名称 | 原因 |
| --- | --- |
| **A3C** | 依赖异步多进程 Actor 与共享参数的 Hogwild 更新；单机 CartPole 收益低、调试成本高。请用 **`a2c`**。 |
| **IMPALA** | 面向大规模分布式（多 actor、独立 learner、V-trace 修正）；超出本教学仓库范围。 |
| **TD(0) 作为独立算法名** | TD(0) 是时序差分**更新规则**；控制任务请选 `sarsa` / `q_learning` / `sarsa_lambda`。 |
| **完整 Rainbow** | 还需 **Categorical DQN (C51)**、**NoisyNet** 等；当前 `rainbow` 仅为子集（见 §1.10）。 |

若命令行传入 `a3c` / `impala` / `td0`，`config_loader.normalize_algorithm` 会抛出上述说明。

---

## 3. 算法 ↔ 代码 ↔ 动作空间

| `algorithm` | 主要文件 | 动作 |
| --- | --- | --- |
| `q_learning` | `qlearning.py` | 离散 |
| `sarsa` | `sarsa.py` | 离散 |
| `sarsa_lambda` | `td_lambda.py` | 离散 |
| `dyna_q` | `dynaq.py` | 离散 |
| `monte_carlo` | `monte_carlo.py` | 离散 |
| `value_iteration` / `policy_iteration` | `dp.py` | 离散 |
| `dqn` / `double_dqn` / `dueling_dqn` / `rainbow` | `dqn.py` | 离散 |
| `reinforce` | `reinforce.py` | 离散 |
| `trpo` | `trpo.py` | 离散 |
| `ppo` | `ppo.py` | 离散 |
| `a2c` | `a2c.py` | 离散 |
| `ddpg` | `ddpg.py` | 连续 |
| `td3` | `td3.py` | 连续 |
| `sac` | `sac.py` | 连续 |
| `mpc` | `mpc.py` | 连续 |

公共组件：`common.py`（回放池、MLP、软更新）。统一接口：`base.py` 的 `select_action` / `save` / `load`。

---

## 4. 推荐阅读与实验顺序

1. `q_learning` → `sarsa` → `sarsa_lambda` → `dyna_q`（表格 TD 谱系）  
2. `monte_carlo` 与 `value_iteration`（无 bootstrap vs 有模型）  
3. `dqn` → `double_dqn` → `dueling_dqn` → `rainbow`  
4. `reinforce` → `a2c` → `trpo` → `ppo`  
5. `ddpg` → `td3` → `sac` → `mpc`  

冒烟（不追求满分）：

```bash
python train.py --algorithm sarsa --total-steps 3000
python train.py --algorithm dqn --total-steps 3000
python train.py --algorithm ppo          # 易达 500
python train.py --algorithm sac --total-steps 5000
```

---

## 5. 参考（经典文献）

- Watkins & Dayan, Q-Learning, 1992  
- Rummery & Niranjan, SARSA  
- Sutton & Barto, *Reinforcement Learning: An Introduction*（MC、TD(λ)、Dyna）  
- Mnih et al., DQN, 2015；van Hasselt et al., Double DQN；Wang et al., Dueling；Hessel et al., Rainbow  
- Williams, REINFORCE；Schulman et al., TRPO / PPO / GAE  
- Lillicrap et al., DDPG；Fujimoto et al., TD3；Haarnoja et al., SAC  
- Mnih et al., A3C；Espeholt et al., IMPALA  

---

*文档与 `/root/autodl-tmp/cartpole_rl/algorithms/` 实现对应；若行为与文档冲突，以代码为准。*
