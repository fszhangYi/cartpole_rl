# 强化学习算法零基础详解（本仓库对照版）

> 读者假设：你可能听说过「AI 玩游戏」，但没系统学过强化学习。  
> 本文目标：用大白话讲清「智能体怎么通过试错变强」，再对照本仓库每一个算法说明它在干什么、怎么跑。  
> 公式采用 GitHub 可渲染写法：行内 `$`公式`$`，块级用 ` ```math ` 代码块。

切换算法：

```bash
cd /root/autodl-tmp/cartpole_rl
python train.py --algorithm ppo
python train.py --algorithm q_learning
python train.py --algorithm dqn
```

代码位置：`algorithms/` · 训练循环：`training_loops.py` · 工厂：`algorithms/factory.py`。

---

# 第一部分：从零认识强化学习

## A.1 监督学习 vs 强化学习（先建立直觉）

| | 监督学习（常见「分类/回归」） | 强化学习（RL） |
| --- | --- | --- |
| 数据从哪来 | 老师给好「题目 + 标准答案」 | **没有标准答案**，只有「做得好不好」的分数 |
| 例子 | 看图识猫：每张图已标好是不是猫 | 骑自行车：摔了疼、稳住了舒服，自己摸索 |
| 反馈 | 每一步都有对错 | 往往要过一会儿才知道这一串动作值不值 |
| 目标 | 拟合标签 | **最大化长期总分** |

强化学习的核心角色：

1. **智能体（Agent）**：做决定的「大脑」（本仓库里的 PPO、DQN 等）。  
2. **环境（Environment）**：世界规则（本仓库是 MuJoCo 里的平衡车）。  
3. **状态（State）$`s`$**：此刻观察到的情况。  
4. **动作（Action）$`a`$**：智能体可选的操作。  
5. **奖励（Reward）$`r`$**：环境立刻打的分（可正可负）。  
6. **策略（Policy）$`\pi`$**：在状态 $`s`$ 下怎么选动作（可以是「表格」或「神经网络」）。

交互循环（非常重要，后面每个算法都绕不开）：

```text
观察状态 s → 选动作 a → 环境变成 s' 并给奖励 r → 再观察 → …
```

智能体要学的是：**在什么状态下该做什么动作，才能让以后加起来的奖励尽量大。**

---

## A.2 本仓库的「游戏」：CartPole 平衡车

想象一根杆竖在小车上，杆容易倒。你只能让小车往左或往右推（或连续调节推力）。

| 概念 | 在本仓库里是什么 |
| --- | --- |
| 状态 $`s`$ | 四个数：$`[x,\dot x,\theta,\dot\theta]`$（车位置、车速、杆角度、杆角速度） |
| 离散动作 | `0` = 向左用力，`1` = 向右用力 |
| 连续动作 | 一个实数 $`a\in[-1,1]`$，表示推力方向与大小比例（DDPG/TD3/SAC/MPC 用） |
| 奖励 | 杆没倒、车没冲出轨道：这一步 **+1**；倒了或出界：这一步 **0** 并结束 |
| 一局结束 | 杆倾角太大、车位移太大，或已经撑满 500 步 |
| 「学得好」 | 很多局里都能接近撑满 500 步 |

你可以先跑可视化感受任务（默认 PPO）：

```bash
python visualize_mjviser.py --port 6008
```

---

## A.3 三个必须先懂的词

### （1）回报（Return）：不只看眼前

一步奖励只是「这一下」的分。真正在意的是从现在起以后能攒多少分：

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
```

$`\gamma`$（gamma，折扣因子，本仓库常用 `0.99`）：

- 接近 1：很在乎长远（适合「要撑很久」的平衡车）。  
- 接近 0：只在乎眼前几步。

### （2）探索 vs 利用

- **利用**：按目前认为最好的动作来（贪心）。  
- **探索**：故意试试别的动作，否则可能永远发现不了更好策略。

常见做法：$`\varepsilon`$-greedy —— 以概率 $`\varepsilon`$ 随机乱动，其余时间选当前最优。训练初期 $`\varepsilon`$ 大，后期变小。

### （3）同策略 vs 离策略（on-policy / off-policy）

- **同策略（on-policy）**：边用策略 A 收集数据，边改进**同一个**策略 A（如 SARSA、PPO）。数据用完通常就扔。  
- **离策略（off-policy）**：用策略 A 收集的数据，去学/改进策略 B（如 Q-Learning、DQN）。旧数据可以放进**经验回放池**反复用。

小白记忆法：

- 同策略 ≈ 「自己练自己，练完这套招式就改招式」。  
- 离策略 ≈ 「看别人（或过去的自己）怎么玩，也能学」。

### （4）表格方法 vs 神经网络方法

CartPole 的状态是**连续实数**，理论上状态无穷多。

- **表格法**：先把连续状态**切成格子（分箱）**，每个格子一张「动作得分表」。实现简单、好懂，但格子太粗就不准。  
- **神经网络法**：直接输入四个实数，输出动作价值或动作概率。表达能力强，但更难调、更「黑盒」。

---

## A.4 算法地图（先看位置再深入）

```text
基于「价值」（先估「这个动作值多少分」）
  ├─ 表格：Q-Learning / SARSA / SARSA(λ) / Dyna-Q / 蒙特卡洛 / 动态规划
  └─ 深度：DQN / Double DQN / Dueling / Rainbow(精简)

基于「策略」（直接学「该怎么做」）
  └─ REINFORCE / TRPO / PPO

Actor-Critic（策略 + 价值一起学）
  └─ A2C / DDPG / TD3 / SAC

基于模型 / 规划
  └─ Dyna-Q（学转移再规划）/ MPC（先拟合动力学再在线规划）
```

---

# 第二部分：每个算法讲透（小白版）

下面每个算法都按同一结构写：**它是谁 → 生活类比 → 在干什么 → 步骤 → 公式白话 → 和谁像 → 本仓库怎么用 → 常见疑问**。

---

## 1. Q-Learning（配置名：`q_learning`）

### 它是谁

最经典的「查表学动作好坏」的算法之一。维护一张表 $`Q(s,a)`$，意思是：**在状态 s 选动作 a，以后大概能拿多少分。**

### 生活类比

你在一个陌生城市找餐厅：每个「路口 + 往哪走」都记一个「这条路最终吃得有多爽」的分数。下次到同一路口，选分数最高的方向。但你记分时会假设「下一步我会选看起来最爽的那条」——即使你当时其实在随便逛。

### 在干什么（离策略的关键）

更新时用的是：

> 「下一状态里，**所有动作里 Q 最大的那个**」

而不是「我下一步真的会选的那个动作」。所以叫 **off-policy**：学的是「理想贪心玩家」的价值，而你收集数据时可能还在乱探索。

### 一步更新（白话）

1. 看到状态 $`s`$，用 $`\varepsilon`$-greedy 选动作 $`a`$（大多时候选当前 Q 最大，偶尔随机）。  
2. 执行后得到奖励 $`r`$ 和新状态 $`s'`$。  
3. 算一个「更靠谱的目标分」：立刻的 $`r`$ + 打折后的「未来最好动作的分」。  
4. 把原来的 $`Q(s,a)`$ 往这个目标挪一点点（学习率 $`\alpha`$）。

```math
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr]
```

白话翻译：

- 括号里：`目标 − 旧估计` = 误差（TD 误差）。  
- $`\alpha`$：一次别改太狠，慢慢纠正。  
- $`\max_{a'}`$：幻想下一步选最好的。

### 本仓库注意

- 文件：`algorithms/qlearning.py`  
- 连续状态要先**分箱**（`n_bins`），表才能建得出来。  
- 跑法：`python train.py --algorithm q_learning`

### 小白常见疑问

- **Q：为什么我练很久还是很快倒？**  
  A：分箱太粗、或步数不够。表格法在 CartPole 上往往要很多交互步。  
- **Q：和下面的 SARSA 差在哪？**  
  A：差在「目标里用 max 还是用真实下一步动作」。见下一节。

---

## 2. SARSA（配置名：`sarsa`）

### 它是谁

名字来自五个字母串起来的过程：**S**tate → **A**ction → **R**eward → **S**tate' → **A**ction'。

### 生活类比

还是记「路口怎么走爽」，但记分时更诚实：  
「我下一步**真的会怎么走**（包括可能乱逛），就按那个来估未来」，不幻想自己永远完美贪心。

### 在干什么（同策略）

目标里用 $`Q(s',a')`$，其中 $`a'`$ 是**按当前策略实际选出的下一步动作**。你学的是「带着探索习惯的自己」的价值，所以是 **on-policy**。

```math
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[r+\gamma Q(s',a')-Q(s,a)\bigr]
```

### 训练时必须多记住一个动作

循环大致是：

1. 在 $`s`$ 选 $`a`$  
2. 走出 $`r,s'`$  
3. **在 $`s'`$ 先选好 $`a'`$**  
4. 用 $`(s,a,r,s',a')`$ 更新  
5. 令 $`s\leftarrow s'`$，$`a\leftarrow a'`$，继续

### 本仓库

- 文件：`algorithms/sarsa.py`  
- 跑法：`python train.py --algorithm sarsa`

### 和 Q-Learning 怎么选（直觉）

- 想学「如果我以后很贪心会怎样」→ Q-Learning。  
- 想学「我现在这种会乱探索的行为到底值多少」→ SARSA（有时更保守、更安全）。

---

## 3. SARSA(λ) / 资格迹（配置名：`sarsa_lambda`）

### 它是谁

SARSA 的「加强版」：一次误差不只改当前格子，还会按「最近走过的路径」往回传责。

### 生活类比

考试错了一题，不只改这一题的笔记，还会把**刚复习过、导致你犯这错的相关知识点**一起标红。标红强度随时间淡去——这就是**资格迹** $`E(s,a)`$。

### 为什么需要它

一步 SARSA 很「近视」：只改刚刚那一对 $`(s,a)`$。  
蒙特卡洛又太「远视」：必须等整局结束才更新，方差大。  
$`\lambda`$ 在中间滑动：

- $`\lambda=0`$：就是普通一步 SARSA。  
- $`\lambda`$ 靠近 1：更像把整段轨迹的责任都算进去。

### 白话步骤

1. 像 SARSA 一样算误差 $`\delta`$（目标 − 当前 Q）。  
2. 当前访问的 $`(s,a)`$ 的迹 +1。  
3. **整张表**按 $`Q \leftarrow Q + \alpha \delta E`$ 更新。  
4. 所有迹乘上 $`\gamma\lambda`$ 衰减；一局结束迹清零。

```math
\delta_t = r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)
```

```math
Q \leftarrow Q + \alpha\,\delta\,E,\quad
E \leftarrow \gamma\lambda E
```

（访问时还要对当前格 $`E(s_t,a_t)\leftarrow E(s_t,a_t)+1`$。）

### 本仓库

- 文件：`algorithms/td_lambda.py`  
- 配置里的 `lam` 就是 $`\lambda`$  
- 跑法：`python train.py --algorithm sarsa_lambda`

### 关于「TD(0)」

TD(0) 不是另一个独立「游戏通关算法名」，而是「只用一步 bootstrapping 的更新规则」。  
在本仓库：想用 TD(0) 做控制 → 用 `sarsa` 或 `q_learning`。

---

## 4. Dyna-Q（配置名：`dyna_q`）

### 它是谁

「一边跟真环境玩，一边在脑子里开小灶复习」的 Q-Learning。

### 生活类比

下棋：你下了真实一局（真交互），同时把棋谱记下来；空下来反复在棋谱上推演（planning），不用每一步都找真人陪练。

### 在干什么

1. **真实一步**：和环境交互，做一次 Q-Learning 更新，并把 $`(s,a)\to(r,s')`$ 记进简易模型。  
2. **规划若干步**：从记忆里随机抽过去的 $`(s,a)`$，用记下的结果再做 Q 更新（本仓库默认 `planning_steps=10`）。

这样同样的真实样本，能「榨」出更多学习信号。

### 本仓库局限（小白也要知道）

模型是「最后一次见到的结果」字典，不是完整概率模型；够教学，不等于工业级世界模型。

- 文件：`algorithms/dynaq.py`  
- 跑法：`python train.py --algorithm dyna_q`

---

## 5. 蒙特卡洛控制（配置名：`monte_carlo`）

### 它是谁

**整局结束才结算**的学习：不靠「用当前 Q 估计未来」，而是用真实一路走下来的总分。

### 生活类比

打完一整局游戏才看战绩，用整局得分回头评价「刚才每个决策好不好」，而不是每杀一人就立刻改战术。

### 优点与代价

- 优点：不依赖对未来的 Gu 计（在固定策略下，估计可以无偏）。  
- 代价：必须等一局结束；分数波动大；学得慢。

### First-visit 是什么

一局里同一个 $`(s,a)`$ 可能出现多次。First-visit = **只在第一次出现时**用本局回报去更新，避免重复计数纠结。

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
```

对首次访问的 $`(s,a)`$，把所有见到的 $`G`$ 取平均当作 $`Q(s,a)`$。

### 本仓库

- 文件：`algorithms/monte_carlo.py`  
- 跑法：`python train.py --algorithm monte_carlo`

---

## 6. 价值迭代 / 策略迭代（`value_iteration` / `policy_iteration`）

### 它们是谁

属于**动态规划（DP）**：如果你已经知道「在任意状态做任意动作，会以什么概率到下一状态、拿多少奖励」，就可以用数学迭代算出最优策略——**甚至不必再跟环境试错**（理想情况）。

### 小白难点：我们并没有现成模型

真实 CartPole 的转移公式很复杂，本仓库做法分两阶段：

1. **探索收集**：先 $`\varepsilon`$-greedy 乱跑，用计数估计「大概会去哪、平均奖励多少」。  
2. **规划求解**：在这张「估出来的格子世界地图」上做价值迭代或策略迭代，得到 Q 表。  
3. **执行**：把探索关掉，按 Q 贪心玩。

### 价值迭代（直觉）

反复刷所有格子：  
「这个动作的分 = 立刻奖励 + 折扣 × 下一格最好动作的分」。刷很多次后数字稳定。

```math
Q(s,a)\leftarrow \hat{r}(s,a)+\gamma\sum_{s'}\hat{P}(s'|s,a)\max_{a'}Q(s',a')
```

### 策略迭代（直觉）

更像「固定一种玩法 → 评估这种玩法值多少分 → 改成更好玩法 → 再评估」的交替。本仓库用多次扫描近似实现。

### 本仓库

- 文件：`algorithms/dp.py`  
- 跑法：
  - `python train.py --algorithm value_iteration`
  - `python train.py --algorithm policy_iteration`

### 常见疑问

- **Q：为什么有时规划完还是一般？**  
  A：探索不够 → 地图估歪了 → 再精确的迭代也是在错误地图上算最优。

---

## 7. DQN（配置名：`dqn`）

### 它是谁

**Deep Q-Network**：用神经网络代替巨大的 Q 表。输入状态向量，输出每个离散动作的 Q 值。

### 为什么表格不够用

状态一连续、维度一高，格子数爆炸（维数灾难）。神经网络可以「泛化」：没见过的相似状态，也能给个靠谱估值。

### 两个救命稻草（不懂这两点就看不懂 DQN）

1. **经验回放（Replay Buffer）**  
   把过去的 $`(s,a,r,s')`$ 存进池子，随机抽样训练。  
   作用：打破「连续几帧长得很像」的相关性，数据还能多用几次（离策略）。

2. **目标网络（Target Network）**  
   另存一份更新较慢的网络 $`Q_{\theta^-}`$ 专门用来算训练目标。  
   作用：目标别跟在线网络一起晃，否则像「用自己晃动的成绩单给自己打分」。

```math
L=\mathbb{E}\big[(r+\gamma\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a))^2\big]
```

白话：让在线网络的 Q，去靠近「奖励 + 目标网络估的未来」。

### 本仓库

- 文件：`algorithms/dqn.py`（`variant=dqn`）  
- 跑法：`python train.py --algorithm dqn`  
- 仍用 $`\varepsilon`$-greedy 探索。

### 小白实验建议

先短跑冒烟：`--total-steps 5000`。要接近满分通常需要更多步、耐心调 $`\varepsilon`$ 退火。

---

## 8. Double DQN（配置名：`double_dqn`）

### 它要修什么病

普通 DQN 在算目标时：同一套价值既「选谁最大」又「这个最大到底多高」，容易**盲目乐观（过高估计）**。

### 改法（一句话）

- **在线网络**：负责选出 $`a^*=\arg\max Q_\theta(s',\cdot)`$  
- **目标网络**：只负责回答「$`a^*`$ 到底值多少」$`Q_{\theta^-}(s',a^*)`$

```math
y=r+\gamma Q_{\theta^-}\big(s',\arg\max_{a'}Q_\theta(s',a')\big)
```

### 本仓库

同一套 `DQNAgent`，换 variant 即可：

```bash
python train.py --algorithm double_dqn
```

---

## 9. Dueling DQN（配置名：`dueling_dqn`）

### 它是谁

网络结构改版：先估计「这个状态本身有多好」$`V(s)`$，再估计「各个动作相对好多少」$`A(s,a)`$，合成 Q：

```math
Q(s,a)=V(s)+A(s,a)-\mathrm{mean}_{a'}A(s,a')
```

减均值是为了让 $`V`$ 和 $`A`$ 别抢同一件事、数值可辨识。

### 生活类比

先判断「这局面整体危不危险」（V），再判断「在此局面下左推还是右推相对更好」（A）。

### 本仓库

与 Double 目标一起用：

```bash
python train.py --algorithm dueling_dqn
```

---

## 10. Rainbow 精简版（配置名：`rainbow`）

### 原版 Rainbow 是什么

论文把多种 DQN 改进「彩虹式」叠在一起（Double、Dueling、优先回放、多步回报、NoisyNet、分布价值 C51 等）。

### 本仓库实现了哪些（务必读清）

| 组件 | 本仓库 `rainbow` |
| --- | --- |
| Double + Dueling | ✅ |
| n-step 回报（默认 3 步） | ✅ |
| Prioritized Experience Replay（优先抽「误差大」的样本） | ✅ |
| NoisyNet（用噪声参数探索） | ❌ |
| C51（学回报分布而不只是期望） | ❌ |

所以文档里也叫 **rainbow-lite**。它仍然比裸 DQN 强一截，但**不是**论文全套。

```bash
python train.py --algorithm rainbow
```

---

## 11. REINFORCE（配置名：`reinforce`）

### 它是谁

最经典的**策略梯度**：不建 Q 表，直接让神经网络输出「每个动作的概率」，用整局回报告诉它该提高还是压低这些概率。

### 生活类比

演小品：整场演完观众打分。得分高 → 强化刚才那些即兴动作；得分低 → 以后少那么干。中间没有「逐步标准答案」。

### 核心思想（极重要）

```math
\nabla J \approx \mathbb{E}\big[\nabla\log\pi_\theta(a_t|s_t)\,G_t\big]
```

白话：

- 若这一局从某步看总回报 $`G_t`$ 很好 → 增大当时动作概率。  
- 若很差 → 减小。  
- $`\log\pi`$ 的梯度给出「怎么改网络参数才能让该动作更可能/更不可能」。

### Baseline（本仓库可选）

$`G_t`$ 波动很大。减去一个只依赖状态的基线 $`b(s)`$（常取状态价值 V）可减小方差，不系统性改变最优策略方向。本仓库可用可学习的 V 网络当 baseline。

### 本仓库

- 文件：`algorithms/reinforce.py`  
- 跑法：`python train.py --algorithm reinforce`  
- **按整局更新**，同策略；通常比 PPO 更抖、更慢。

---

## 12. TRPO（配置名：`trpo`）

### 它要解决什么

纯策略梯度若一步改太猛，策略可能「崩盘」：突然变得很差，而且因为是同策略，接下来采到的数据也变差，更难爬回来。

### 核心思想

在「新策略不能离旧策略太远」的**信任域**里，尽量提升性能。距离常用 KL 散度衡量：

> 最大化替代目标，同时约束 $`\mathrm{KL}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})\le\delta`$

工程上用共轭梯度等技巧求更新方向，再用线搜索保证约束大致满足。

### 和 PPO 的关系（小白版）

- TRPO：约束写得很「正统」，实现重。  
- PPO：用裁剪（Clip）等方式**近似**「别改太猛」，实现简单，本仓库主推。

### 本仓库

- 文件：`algorithms/trpo.py`（离散动作简化版，偏教学）  
- 跑法：`python train.py --algorithm trpo`

---

## 13. PPO（配置名：`ppo`）★★ 建议精读

### 它是谁

**Proximal Policy Optimization（近端策略优化）**：目前工程里极常用的同策略 Actor-Critic 算法。本仓库 CartPole 上已验证可稳定打到满分 500。

更细的公式推导见：[CartPole_PPO_手把手教程.md](./CartPole_PPO_手把手教程.md)。这里给零基础总览。

### 三个角色

1. **Actor（演员）**：策略网络，输出左右动作的概率。  
2. **Critic（评论家）**：价值网络 $`V(s)`$，估计「这个状态大概能拿多少分」。  
3. **优势 $`A`$**：`实际表现 − Critic 预期`。A>0 说明这步比预期好，应加强；A<0 应减弱。

### 为什么叫 Proximal（近端）

更新时看概率比：

```math
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}
```

若 $`r_t`$ 离开 1 太远，说明策略改太猛。PPO 用 **Clip** 把比率限制在 $`[1-\varepsilon,1+\varepsilon]`$ 附近再优化，避免一次走极端。

```math
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat{A}_t)\big]
```

### GAE 是什么（只需建立印象）

用一种平滑方式估计优势 $`\hat{A}`$，在「偏差」和「方差」之间折中（参数 $`\lambda`$）。本仓库默认 `gae_lambda=0.95`。

### 训练循环长什么样

1. 用当前策略与环境玩一段时间（rollout，如 2048 步），记下 $`s,a,\log\pi,r,V`$。  
2. 算优势与回报。  
3. 把这批数据反复小批量更新多次（同一批 on-policy 数据可多用几轮，但仍受 Clip 约束）。  
4. 丢掉旧数据，重新采集。

### 本仓库

- 文件：`algorithms/ppo.py`  
- 跑法：`python train.py --algorithm ppo`  
- 可视化：`python visualize_mjviser.py --algorithm ppo --port 6008`

### 小白为什么优先学 PPO

实现相对友好、超参鲁棒、在本任务上效果最好；同时覆盖「策略梯度 + 价值函数 + 信任域思想」三条主线。

---

## 14. A2C（配置名：`a2c`）

### 它是谁

**Advantage Actor-Critic** 的同步版本：一边用 Actor 选动作，一边用 Critic 打分（优势），两者一起更新。

### 和 A3C 的区别（常考）

| | A2C | A3C |
| --- | --- | --- |
| 含义 | Asynchronous 的「同步实现」思路 / 同步多环境常见叫法 | Asynchronous Advantage Actor-Critic |
| 并行 | 通常同步等齐 | 多线程/进程异步推梯度 |
| 本仓库 | ✅ 实现 | ❌ 不实现（见文末） |

小白只需记：本仓库的 `a2c` = **单机同步 Actor-Critic**，不是多进程 A3C。

### 本仓库

- 文件：`algorithms/a2c.py`（骨干网络与 PPO 同类 ActorCritic）  
- 跑法：`python train.py --algorithm a2c`  
- 用短的 n-step 片段估计回报再更新。

---

## 15. DDPG（配置名：`ddpg`）

### 它是谁

面向**连续动作**的经典算法：Deep Deterministic Policy Gradient。

平衡车若动作用「左右两键」是离散；若动作用「推力是 −1 到 1 的实数」则是连续。DDPG 适合后者。

### 核心直觉

- **Actor** 直接输出一个确定的力 $`\mu_\theta(s)`$（不是概率）。  
- **Critic** 打分 $`Q(s,a)`$：「在这状态用这力，未来分多少」。  
- Actor 的目标：输出能让 Critic 分更高的力。  
- 探索：在输出的力上加噪声（否则确定性策略不会试新动作）。  
- 同样使用回放池 + 目标网络（软更新）。

### 本仓库

- 环境会自动 `continuous=True`  
- 文件：`algorithms/ddpg.py`  
- 跑法：`python train.py --algorithm ddpg`

---

## 16. TD3（配置名：`td3`）

### 它是谁

Twin Delayed DDPG：专治 DDPG 爱「过高估计」和「脆」的问题。

### 三板斧（记住口诀）

1. **双 Critic，取较小的 Q 当目标** → 更保守，少吹牛。  
2. **延迟更新 Actor** → Critic 先学稳一点再改策略。  
3. **目标动作加一点裁剪噪声** → 平滑目标，别对尖峰过拟合。

### 本仓库

```bash
python train.py --algorithm td3
```

文件：`algorithms/td3.py`。

---

## 17. SAC（配置名：`sac`）

### 它是谁

Soft Actor-Critic：在连续控制里非常常用。策略是**随机**的（输出动作分布），并且在追求高回报的同时鼓励**熵（随机性）**更大。

### 为什么要「又准又随机」

- 纯确定性策略容易卡在窄道局部最优。  
- 保持一定随机性 = 持续探索 + 对模型误差更鲁棒。

目标直觉：

```math
\text{最大化}\quad \mathbb{E}\big[\text{回报} + \alpha\cdot\text{策略熵}\big]
```

$`\alpha`$ 控制「有多鼓励折腾」。本仓库用固定 $`\alpha`$（未做自动调温度的完整版）。

### 本仓库

- 文件：`algorithms/sac.py`  
- 跑法：`python train.py --algorithm sac`  
- 动作经 tanh 压到 $`[-1,1]`$。

---

## 18. MPC 教学版（配置名：`mpc`）

### 它是谁

**Model Predictive Control（模型预测控制）** 思想：先有一个「世界怎么变」的模型，每次决策时在脑子里试很多未来动作序列，挑看起来最好的**第一步**执行，然后滚动重算。

### 本仓库教学简化（请降低预期）

1. 先随机用力采集数据。  
2. 用最小二乘拟合线性模型：$`s' \approx As + Ba + c`$（真实倒立摆是非线性的，这只是粗近似）。  
3. 在线时随机采样许多动作序列，用模型滚模拟，用「杆是否还算直立」这类启发式打分，选最优序列的第一个动作。

它**不是**神经网络策略，也不是工业级机器人 MPC。目的是让你理解「先建模，再规划」。

```bash
python train.py --algorithm mpc
```

文件：`algorithms/mpc.py`。

---

# 第三部分：本仓库明确不实现的算法

| 名称 | 为什么小白阶段先跳过 / 本仓不做 |
| --- | --- |
| **A3C** | 核心卖点是异步多进程一起推梯度；单机 CartPole 收益小、调试难。请用 **`a2c`**。 |
| **IMPALA** | 为大规模分布式游戏训练设计（很多 Actor + 中央 Learner + V-trace）。超出本教学仓库。 |
| **把 TD(0) 当成单独算法名** | TD(0) 是更新规则。控制任务请用 `sarsa` / `q_learning` / `sarsa_lambda`。 |
| **完整 Rainbow** | 还需 NoisyNet、C51 等。本仓 `rainbow` 只含 Double+Dueling+n-step+PER。 |

若执行 `python train.py --algorithm a3c`，程序会报错并打印上述原因。

---

# 第四部分：小白学习路线与怎么跑

## 推荐阅读顺序

1. **先玩起来**：`ppo` 训练 + 6008 可视化，建立「奖励变高」的体感。  
2. **表格价值**：`q_learning` → `sarsa` → `sarsa_lambda` → `dyna_q`。  
3. **对比整局 vs 逐步**：`monte_carlo` vs 上面的 TD。  
4. **有模型的古典派**：`value_iteration`。  
5. **深度价值**：`dqn` → `double_dqn` → `dueling_dqn` → `rainbow`。  
6. **策略梯度**：`reinforce` → `a2c` → `trpo` → 回到 `ppo` 对照。  
7. **连续动作**：`ddpg` → `td3` → `sac` → `mpc`。

## 命令速查

```bash
# 列表见 config.yaml 顶部注释，或本文件目录
python train.py --algorithm <名字> --total-steps 5000   # 冒烟
python evaluate.py --algorithm <名字> --greedy
python visualize_mjviser.py --algorithm <名字> --port 6008
```

## 心理预期（很重要）

- **不是**每个算法短训都能满分。PPO 在本任务上最容易漂亮。  
- 表格法依赖分箱，常要更多步。  
- 连续控制（DDPG/TD3/SAC）需要足够探索与回放，短训只有「能跑通」的意义。  
- 先追求「懂思想 + 代码能跑」，再追求「刷分」。

---

# 第五部分：术语表（随时翻）

| 术语 | 一句话 |
| --- | --- |
| 智能体 Agent | 做决策的算法 |
| 环境 Env | 给出状态与奖励的世界 |
| 状态 s | 观测到的情况 |
| 动作 a | 可执行的控制 |
| 奖励 r | 即时反馈分数 |
| 回报 G | 未来奖励的折扣和 |
| 策略 π | 状态到动作（分布）的映射 |
| Q(s,a) | 状态-动作价值 |
| V(s) | 状态价值 |
| 优势 A | 相对平均水平好多少 |
| on-policy | 数据策略 = 正在改进的策略 |
| off-policy | 可用别的行为策略产生的数据学习 |
| TD | 用估计值 bootstrapping 的时序差分 |
| 经验回放 | 存转移再随机采样训练 |
| 目标网络 | 缓慢更新的网络，用于稳定训练目标 |
| Actor-Critic | 策略网络 + 价值网络一起学 |
| 熵 | 随机性的度量；SAC 会最大化它 |

---

# 参考（想继续深挖时）

- Sutton & Barto,《Reinforcement Learning: An Introduction》（免费草稿广泛流传）—— MC/TD/Dyna 的圣经级入门。  
- DQN / Double / Dueling / Rainbow 系列论文。  
- Schulman et al.：TRPO、PPO、GAE。  
- Lillicrap et al.：DDPG；Fujimoto et al.：TD3；Haarnoja et al.：SAC。  

PPO 公式逐步推导：[CartPole_PPO_手把手教程.md](./CartPole_PPO_手把手教程.md)  
环境安装：[MuJoCo_mjviser_安装教程.md](./MuJoCo_mjviser_安装教程.md)  
返回：[README.md](./README.md) · 项目首页：[../README.md](../README.md)

---

*本文面向零基础读者扩写；实现细节以 `algorithms/` 源码为准。*
