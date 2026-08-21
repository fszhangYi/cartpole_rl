# MuJoCo 平衡车（CartPole）+ PPO 手把手教程（本机实测细化版）

> 任务：经典强化学习案例 **CartPole / 倒立摆 / 平衡车**  
> 物理：[MuJoCo](https://mujoco.readthedocs.io/) · 算法：**PPO**（Proximal Policy Optimization）  
> 可视化：[mjviser](https://github.com/mujocolab/mjviser)（Viser 网页查看器）  
> 代码目录：`/root/autodl-tmp/cartpole_rl`  
> 环境：`/root/autodl-tmp/conda-envs/mjviser`（承接《MuJoCo + mjviser 安装教程》）  
> **本教程验收端口：`6008`**  
> **进度（2026-08-21）：** 训练约 59k 步达标；贪婪评估 20 局全部回报 **500**；`visualize_mjviser.py --port 6008` 已验收。

---

## 这项工作是什么：你复现完能证明什么

| 层级 | 内容 | 本教程验收 |
| --- | --- | --- |
| 控制问题 | 小车左右施力，让细杆尽量保持竖直 | 单局撑满 500 步 |
| 仿真 | MuJoCo XML + `mj_step` 积分 | `env.py` 可 `reset/step` |
| 学习算法 | 离散动作 PPO（Actor-Critic + Clip） | `train.py` 平均回报冲到 ≥475 |
| 部署观看 | 策略驱动物理，浏览器实时看 | **6008** 打开能看到车杆平衡 |

读操作之前，建议先读完 **§1 PPO 详解**：后面每个超参、每行 `policy_loss` 日志才有落地点。

---

## 0. 读前须知与前置

### 0.1 你需要已经具备

1. 已按《MuJoCo_mjviser_安装教程》建好环境：  
   `conda activate /root/autodl-tmp/conda-envs/mjviser`  
2. 本教程会再装：`torch`（CPU 即可）、`gymnasium`。  
3. **验收只占用端口 6008**。启动可视化前请先停掉该端口上的旧 `mjviser` / 旧 CartPole 进程。

### 0.2 本机锁定版本

| 项 | 版本 |
| --- | --- |
| Python | 3.11 |
| mujoco | 3.11.0 |
| mjviser | 0.0.14 |
| torch | 2.13.0+cpu |
| gymnasium | 1.3.0 |
| 实测达标步数 | ≈ **59392**（早停） |
| 贪婪评估 | 20×**500** |

⏱ 本机实测：依赖安装约数分钟；训练约 **30 s**；评估数秒。

### 0.3 端口与磁盘

```bash
df -h /root/autodl-tmp
ss -tlnp | grep 6008 || echo "6008 空闲"
```

停掉占用 6008 的进程（示例）：

```bash
# 查 PID 后结束
ss -tlnp | grep 6008
kill <pid>    # 不行再用 kill -9 <pid>
```

---

## 1. PPO 算法详细介绍

论文：Schulman et al., *Proximal Policy Optimization Algorithms*, 2017  
（常见引用：[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)）

本节目标：弄清 **PPO 在优化什么、为什么比「裸策略梯度」稳、本仓库每一项损失从哪来**。

### 1.1 强化学习最小记号

智能体与环境交互：在状态 \(s_t\) 选动作 \(a_t\)，得奖励 \(r_t\)，进入 \(s_{t+1}\)。

- **策略** \(\pi_\theta(a|s)\)：参数 \(\theta\) 的条件分布（本教程：离散左右两动作的分类器）。  
- **回报（Return）**  
  \[
  G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
  \]
  \(\gamma\in(0,1]\) 为折扣因子（本仓库默认 `gamma=0.99`）。  
- **目标**：最大化期望回报 \(\mathbb{E}_{\tau\sim\pi_\theta}[\sum_t \gamma^t r_t]\)。

CartPole 的「学得好」很直观：杆不倒、车不出轨的步数尽量多 → 每步 +1，满局 500。

### 1.2 为什么需要「策略梯度」而不是监督学习

没有现成的「正确答案动作标签」。只有标量奖励。策略梯度用 **似然比技巧** 把目标对 \(\theta\) 求导，核心形式（REINFORCE）：

\[
\nabla_\theta J(\theta)
\approx
\mathbb{E}_t\Bigl[
\nabla_\theta \log \pi_\theta(a_t|s_t)\, \hat{A}_t
\Bigr]
\]

直觉：

- \(\hat{A}_t > 0\)：这次动作比「平均预期」好 → 增大该动作概率；  
- \(\hat{A}_t < 0\)：比预期差 → 减小概率。

\(\hat{A}_t\) 叫 **优势（Advantage）**：相对「基准」好多少，而不是原始回报本身。用优势通常比用 raw return 方差更小。

### 1.3 Actor-Critic：策略 + 价值网络

纯 REINFORCE 用整条轨迹的 \(G_t\) 当信号，方差大、样本效率差。Actor-Critic 引入 **评论家（Critic）** \(V_\phi(s)\) 估计状态价值：

\[
\hat{A}_t \approx \text{（对未来回报的估计）} - V_\phi(s_t)
\]

本仓库网络结构（`ppo.py` → `ActorCritic`）：

| 头 | 输出 | 作用 |
| --- | --- | --- |
| **Actor** | 两维 logits → `Categorical` | \(\pi_\theta(a\|s)\)，采样或 argmax |
| **Critic** | 标量 \(V(s)\) | 估价值，构造优势与 value loss |

两套 MLP（64-64），正交初始化；Actor 最后一层用很小 `std=0.01`，避免一开始策略过于尖锐。

### 1.4 从 TRPO 到 PPO：信任域思想

朴素策略梯度有个致命问题：**一次更新步子太大**，新策略 \(\pi_{\theta_{\text{new}}}\) 相对旧策略 \(\pi_{\theta_{\text{old}}}\) 偏离过远时：

- 重要性采样比率失真；  
- 性能可能「悬崖式」崩溃，再难爬回来。

**TRPO**（Trust Region Policy Optimization）用约束  
\(\mathbb{E}[\mathrm{KL}(\pi_{\text{old}}\|\pi_{\text{new}})] \le \delta\)  
限制更新幅度，理论漂亮但实现重（共轭梯度、二阶信息）。

**PPO** 用更简单的一阶方法近似同一思想，工业界极常用。最流行的是 **Clipped Surrogate Objective（Clip 目标）**。

### 1.5 PPO-Clip 目标（本仓库使用的核心）

定义 **概率比（probability ratio）**：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
=
\exp\bigl(\log\pi_\theta - \log\pi_{\theta_{\text{old}}}\bigr)
\]

未裁剪的替代目标（类似重要性加权的策略梯度）：

\[
L^{\text{CPI}}(\theta)
=
\mathbb{E}_t\bigl[r_t(\theta)\,\hat{A}_t\bigr]
\]

PPO 将其改成 **裁剪版**：

\[
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t\Bigl[
\min\bigl(
  r_t(\theta)\hat{A}_t,\;
  \mathrm{clip}\bigl(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\hat{A}_t
\bigr)
\Bigr]
\]

本仓库 `clip_eps = 0.2`，即 \(\varepsilon=0.2\)，比率被限制在 \([0.8,\,1.2]\)。

**怎么读这个 \(\min\)：**

| 情况 | 行为 |
| --- | --- |
| \(\hat{A}_t>0\)（好动作） | 希望增大 \(r_t\)；但超过 \(1+\varepsilon\) 后目标不再上升 → **防止概率涨太猛** |
| \(\hat{A}_t<0\)（坏动作） | 希望减小 \(r_t\)；裁剪同样限制一步砍得太狠 |

优化时我们 **最大化** \(L^{\text{CLIP}}\)；代码里写成最小化 `-min(...)`，即 `policy_loss`。

对应实现（概念对齐 `ppo.py` 的 `update`）：

```text
ratio   = exp(new_logprob - old_logprob)
surr1   = ratio * adv
surr2   = clamp(ratio, 1-ε, 1+ε) * adv
policy_loss = -mean( min(surr1, surr2) )
```

### 1.6 优势估计：GAE（Generalized Advantage Estimation）

直接用 \(G_t - V(s_t)\) 可以，但偏置/方差不好折中。本仓库用 **GAE-λ**（`gae_lambda=0.95`）：

先定义 TD 残差：

\[
\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)
\]

（\(d_t=1\) 表示 episode 结束，下一状态价值置 0。）

再递推：

\[
\hat{A}_t = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1}
\]

\[
\hat{R}_t = \hat{A}_t + V(s_t)
\quad\text{（returns，给 Critic 回归）}
\]

| 超参 | 典型值 | 含义 |
| --- | --- | --- |
| \(\gamma\) | 0.99 | 更重视长期；CartPole 需要撑很多步 |
| \(\lambda\) | 0.95 | 越接近 1 越像蒙特卡洛（方差大、偏置小）；越小越像一步 TD |

更新前对 `advantages` 做 **标准化**（减均值除标准差），稳定不同 rollout 尺度。

### 1.7 完整损失：策略 + 价值 + 熵

\[
L(\theta,\phi)
=
\mathbb{E}\Bigl[
  L^{\text{CLIP}}
  + c_v\,(V_\phi(s_t)-\hat{R}_t)^2
  - c_e\,H\bigl[\pi_\theta(\cdot|s_t)\bigr]
\Bigr]
\]

| 项 | 本仓库系数 | 作用 |
| --- | --- | --- |
| Clip 策略损失 | — | 改进策略且限制步长 |
| Value loss | `vf_coef=0.5` | 让 Critic 跟上 returns |
| Entropy bonus | `ent_coef=0.01`（损失里为 **减** 熵） | 鼓励探索，避免过早塌缩成确定性左右摇摆失败 |

另有 `max_grad_norm=0.5` 梯度裁剪，防止一次反传爆炸。

### 1.8 训练循环：Rollout → 多 epoch 小批量更新

PPO 属于 **on-policy**：用来更新的数据必须来自 **当前（或刚收集时）的策略**。流程：

```text
重复直到总环境步数够 / 达标：
  1) 用当前 π 与环境交互 rollout_steps 步（本仓库 2048）
     存：s, a, logπ_old(a|s), r, done, V(s)
  2) 用 GAE 算 advantages / returns
  3) 对这批数据做 update_epochs 轮（默认 10）
     每轮打乱，按 minibatch_size（64）做 Adam 更新
  4) 丢弃旧数据，重新采样（保证近似 on-policy）
```

与 DQN 等 off-policy 对比：PPO **不**依赖巨大 replay；实现简单，超参相对鲁棒，适合本教程这种连续控制/经典控制入门。

### 1.9 离散 CartPole 上 PPO 在学什么

状态 \(s=(x,\dot{x},\theta,\dot{\theta})\)。  
动作 \(a\in\{\text{左力},\text{右力}\}\)。

早期：熵高，左右几乎随机，杆很快倒（回报个位数）。  
中期：Critic 开始分清「危险倾斜」；Actor 在 \(\theta\) 偏向一侧时更常打反向力。  
后期：策略近似「小误差反馈控制」；贪婪评估可稳定满 500 步。

本机日志可见：`avgR100` 从 ~6 爬到数百，熵 `H` 从 ~0.68 降到 ~0.27（仍保留一点随机性；评估用 `--greedy` 取 argmax）。

### 1.10 与相关算法的对照（帮助定位 PPO）

| 算法 | 要点 | 相对 PPO |
| --- | --- | --- |
| REINFORCE | 蒙特卡洛回报 × \(\nabla\log\pi\) | 方差大，无信任域 |
| A2C/A3C | Actor-Critic + 多环境 | 无 clip；同步/异步变体 |
| TRPO | KL 信任域 + 二阶 | 更重；PPO 为其实用近似 |
| SAC / TD3 | 离策略、连续动作常用 | CartPole 离散入门不必上 |
| DQN | 学 Q，ε-greedy | 也很适合 CartPole；本教程选 PPO 为了贯通策略梯度主线 |

### 1.11 本仓库超参速查（与代码默认值一致）

| 符号 / 参数 | 默认 | 位置 |
| --- | --- | --- |
| `lr` | 3e-4 | Adam |
| `gamma` | 0.99 | GAE / 回报 |
| `gae_lambda` | 0.95 | GAE |
| `clip_eps` | 0.2 | PPO-Clip |
| `ent_coef` | 0.01 | 熵奖励 |
| `vf_coef` | 0.5 | 价值损失权重 |
| `update_epochs` | 10 | 每批数据复用轮数 |
| `minibatch_size` | 64 | SGD 小批 |
| `rollout_steps` | 2048 | 每次更新前交互步数 |
| `max_episode_steps` | 500 | 环境截断（满分为 500） |
| 早停 | 最近 20 局均值 ≥ 475 | `train.py` |

---

## 2. 任务与 MuJoCo 建模

### 2.1 MDP 设定

| 项 | 设定 |
| --- | --- |
| 观测 | `[cart_x, cart_vx, pole_theta, pole_omega]`，`float32` |
| 动作 | `Discrete(2)`：0→向左力（ctrl=-1），1→向右力（ctrl=+1） |
| 奖励 | 未失败：每步 **+1**；失败当步 **0** |
| 终止 `terminated` | `\|θ\| > 12°` 或 `\|x\| > 1.4` |
| 截断 `truncated` | 步数 ≥ 500 |
| 物理步长 | XML 中 `timestep="0.02"`（50 Hz） |

约定：MuJoCo 铰链角 **θ=0 表示竖直向上**（与部分 Gym 经典实现「θ=0 为下垂」不同，以本仓库 XML 为准）。

### 2.2 XML 要点（`assets/cartpole.xml`）

- `slider`：小车沿 x 滑动，限位约 ±1.5 m。  
- `hinge`：杆绕 y 轴转动。  
- `motor`：作用在 slider 上，`gear="100"`，`ctrlrange="-1 1"`。  
- 杆用 capsule + tip 小球，便于 mjviser 里辨认。

环境封装：`env.py` → `CartPoleMuJoCoEnv`（Gymnasium API：`reset` / `step`）。

---

## 3. 工程结构

```text
/root/autodl-tmp/cartpole_rl/
  assets/cartpole.xml       # MuJoCo 模型
  env.py                    # 环境
  ppo.py                    # ActorCritic + PPO-Clip + GAE
  train.py                  # 训练与早停
  evaluate.py               # 无界面评估
  visualize_mjviser.py      # 策略 + mjviser，默认端口 6008
  checkpoints/              # *.pt 权重
  runs/train_history.json   # 训练曲线
  README.md                 # 短说明
```

与 §1 的对应关系：

- GAE / Clip / 熵 → `ppo.py`  
- Rollout 循环、早停、存盘 → `train.py`  
- `step_fn` 里用策略写 `ctrl` 再步进 → `visualize_mjviser.py`

**全部源码（含逐段中文注释）见文末 [附录 A](#附录-a-本项目全部源码附中文注释)。**

---

## 4. 环境准备

### 4.1 激活已有 conda 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
```

若尚未安装 MuJoCo / mjviser，先完成《MuJoCo_mjviser_安装教程》。

### 4.2 安装训练依赖（AutoDL 建议先关 turbo 代理）

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset REQUESTS_CA_BUNDLE SSL_CERT_FILE

# CartPole 很小，CPU 版 torch 足够
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium
```

验收：

```bash
python - <<'EOF'
import torch, gymnasium, mujoco
print("torch", torch.__version__)
print("gymnasium", gymnasium.__version__)
print("mujoco", mujoco.__version__)
EOF
```

### 4.3 进入项目目录

```bash
cd /root/autodl-tmp/cartpole_rl
ls assets/cartpole.xml env.py ppo.py train.py evaluate.py visualize_mjviser.py
```

---

## 5. 训练

```bash
cd /root/autodl-tmp/cartpole_rl
python train.py --total-steps 80000 --seed 42
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--total-steps` | 最大环境交互步数上限 |
| `--rollout-steps` | 每次 PPO 更新前采集步数（默认 2048） |
| `--solved-reward` | 早停阈值（默认 475） |
| `--solved-window` | 用最近多少局均值判断达标（默认 20） |
| `--device` | 默认 `cpu` |

### 5.1 日志怎么读

示例行：

```text
step= 59392  avgR100=  316.7  avgLen= 317.4  pi=-0.004  v=35.687  H=0.272  t=27.3s
Solved: mean return over last 20 eps = 481.6 >= 475.0
```

| 字段 | 含义 |
| --- | --- |
| `step` | 累计环境步 |
| `avgR100` | 最近最多 100 局平均回报（含早期低分，故达标时可能仍 <475） |
| `pi` / `v` / `H` | 策略损失、价值损失、熵（见 §1.7） |
| `Solved` | **最近 20 局**均值 ≥475 → 早停 |

本机实测约 **29** 次更新、**≈59k** 步触发 Solved；墙钟约 **半分钟**。

### 5.2 产出文件

| 路径 | 说明 |
| --- | --- |
| `checkpoints/cartpole_ppo_best.pt` | 训练过程/结束时保存的策略 |
| `checkpoints/cartpole_ppo_final.pt` | 结束时权重 |
| `runs/train_history.json` | 每次更新的指标，可自行画曲线 |

---

## 6. 评估（无界面）

```bash
cd /root/autodl-tmp/cartpole_rl
python evaluate.py \
  --checkpoint checkpoints/cartpole_ppo_best.pt \
  --episodes 20 \
  --greedy
```

- `--greedy`：每步 `argmax`，结果更稳、便于验收。  
- 不加则按随机策略采样，回报方差更大。

**本机验收期望：**

```text
mean=500.0  std=0.0  min=500  max=500
```

若均值明显低于 400：检查是否加载错权重、是否未用 `--greedy`、XML/角度阈值是否被改过。

---

## 7. mjviser 可视化（必须端口 6008）

### 7.1 先释放 6008

```bash
ss -tlnp | grep 6008 || echo "空闲"
# 若有占用：
kill <pid>
# 必要时：
kill -9 <pid>
```

同时若误开了 6009 上的旧 CartPole，一并停掉，避免混淆。

### 7.2 启动

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
cd /root/autodl-tmp/cartpole_rl

# 前台（看日志）
python visualize_mjviser.py --port 6008

# 或后台
mkdir -p /root/autodl-tmp/logs
nohup python visualize_mjviser.py --port 6008 \
  > /root/autodl-tmp/logs/cartpole_mjviser_6008.log 2>&1 &
sleep 2
ss -tlnp | grep 6008
tail -n 20 /root/autodl-tmp/logs/cartpole_mjviser_6008.log
```

成功时应看到：

```text
Loaded policy: .../cartpole_ppo_best.pt
╭────── viser (listening *:6008) ───────╮
│   HTTP      │ http://localhost:6008   │
╰───────────────────────────────────────╯
```

### 7.3 浏览器

| 场景 | 地址 |
| --- | --- |
| 本机 / SSH 隧道 | `http://localhost:6008` |
| AutoDL 自定义服务 | 映射实例端口 **6008** |

页面可 **Pause / Step / Reset / 调速**。默认贪婪策略；若要看随机策略：

```bash
python visualize_mjviser.py --port 6008 --stochastic
```

### 7.4 实现要点（与 PPO 的衔接）

`visualize_mjviser.py` 向 `mjviser.Viewer` 传入自定义 `step_fn`：

1. 用当前观测问 Actor 要动作（greedy 或 sample）；  
2. 调用 `env.step(action)`（内部写 `data.ctrl` 并 `mj_step`）；  
3. 失败或截断则 `env.reset()`，打印该局回报。

注意：`step_fn` **不要**再 `time.sleep`——Viser 已按仿真时间预算调用 `step_fn`，额外 sleep 会导致卡顿与 `[CAPPED]`。

---

## 8. 从零对照：算法 ↔ 代码地图

| 概念（§1） | 代码位置 |
| --- | --- |
| \(\pi_\theta\) / \(V_\phi\) | `ppo.py` → `ActorCritic` |
| 采样 \(a,\log\pi,V\) | `ActorCritic.act` |
| GAE | `PPO.compute_gae` |
| \(r_t(\theta)\) 与 Clip | `PPO.update` 中 `ratio` / `clamp` / `min` |
| 熵奖励 | `- ent_coef * entropy` |
| Rollout + 多 epoch | `train.py` 主循环 |
| 环境转移与奖励 | `env.py` → `CartPoleMuJoCoEnv.step` |
| 部署 | `visualize_mjviser.py` → `step_fn` |

建议阅读顺序：`env.py` → `ppo.py`（对照 §1.5–1.7）→ `train.py` → `evaluate.py` → `visualize_mjviser.py`。

---

## 9. 坑点与排障

### 坑 A：6008 被旧 mjviser 占用

- **现象：** `Address already in use` 或 Autodl 打开仍是旧场景。  
- **处理：** §7.1 结束占用进程后再启 CartPole。

### 坑 B：评估分数低但训练显示 Solved

- 查是否评估了更早的 `best`（若中途 avgR100 虚高）。以 `final` 或训练结束时写入的 `best` 为准；用 `--greedy`。

### 坑 C：可视化里杆乱晃 / 立刻倒

- 确认日志有 `Loaded policy: ...cartpole_ppo_best.pt`。  
- 无权重时会警告并使用随机策略。

### 坑 D：`torch` 安装失败 / 系统盘满

- 使用 CPU 索引安装；pip 缓存与环境放在 `/root/autodl-tmp`（见安装教程）。

### 坑 E：角度阈值理解错误

- 本仓库 θ=0 为竖直；失败阈约 ±12°。改 XML 质量/阻尼后需 **重新训练**。

### 坑 F：想「再稳一点」

- 略增 `total-steps`；评估与可视化用 greedy；或减小 `ent_coef`（探索变少，可能更早收敛也可能更易局部）。

---

## 10. 验收清单

- [ ] 能激活 `.../conda-envs/mjviser`，`import mujoco, torch, gymnasium` 成功  
- [ ] `python train.py` 出现 `Solved` 或 `avgR100` 明显上升并生成 `checkpoints/*.pt`  
- [ ] `python evaluate.py --greedy` 平均回报接近 **500**  
- [ ] `ss -tlnp | grep 6008` 显示 CartPole 可视化在听  
- [ ] 浏览器打开 **6008** 能看到小车维持杆平衡  
- [ ] （可选）能口述 PPO-Clip 中 \(r_t(\theta)\)、\(\varepsilon\)、GAE、熵项各自的作用（§1）

全部勾选即完成本教程。

---

## 11. 扩展（可选，不在最小验收内）

- 把动作改成连续力 + Beta/Gaussian 策略（仍可用 PPO-Clip）。  
- 换 `stable-baselines3` 的 `PPO` 对比自写实现。  
- 记录视频：`env.render_mode="rgb_array"` + imageio。  
- 画 `runs/train_history.json` 中 `avg_return_100` 曲线。

---

## 12. 参考

- Schulman et al., PPO, 2017: <https://arxiv.org/abs/1707.06347>  
- Schulman et al., GAE, 2015: <https://arxiv.org/abs/1506.02438>  
- Schulman et al., TRPO, 2015: <https://arxiv.org/abs/1502.05477>  
- MuJoCo 文档: <https://mujoco.readthedocs.io/>  
- mjviser: <https://github.com/mujocolab/mjviser>  
- Gymnasium: <https://gymnasium.farama.org/>  
- 本机短说明: `/root/autodl-tmp/cartpole_rl/README.md`  
- 环境安装篇: `/root/autodl-tmp/MuJoCo_mjviser_安装教程.md`

---

## 附录 A. 本项目全部源码（附中文注释）

> 路径根目录：`/root/autodl-tmp/cartpole_rl/`  
> 下列代码与仓库文件一一对应；注释在「可运行逻辑」之上补充说明，便于对照 §1 PPO 与 §8 代码地图阅读。  
> 若磁盘上的源文件与附录有细微差异，**以仓库文件为准**；附录注释不改变算法语义。

文件一览：

| 附录小节 | 文件 |
| --- | --- |
| A.1 | `assets/cartpole.xml` |
| A.2 | `env.py` |
| A.3 | `ppo.py` |
| A.4 | `train.py` |
| A.5 | `evaluate.py` |
| A.6 | `visualize_mjviser.py` |

---

### A.1 `assets/cartpole.xml` — MuJoCo 平衡车模型

```xml
<!--
  经典 CartPole / 倒立摆。
  约定：铰链角 θ=0 表示杆竖直向上；小车沿世界系 x 轴滑动。
  强化学习里离散动作左右，会映射到 motor 的 ctrl ∈ [-1, 1]。
-->
<mujoco model="cartpole">
  <!-- angle=radian：关节角用弧度；inertiafromgeom：由几何自动推惯性 -->
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <!-- 仿真步长 0.02s → 50Hz；RK4 积分；重力向下 -->
  <option timestep="0.02" gravity="0 0 -9.81" integrator="RK4"/>

  <default>
    <!-- 关节阻尼与转子惯量，略增稳定性 -->
    <joint damping="0.05" armature="0.01"/>
    <geom friction="0.5 0.01 0.001" rgba="0.75 0.75 0.8 1"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <!-- 地面与导轨仅作可视化，contype/conaffinity=0 表示不参与碰撞 -->
    <geom name="floor" type="plane" size="3 1 0.05" rgba="0.92 0.92 0.95 1"
          contype="0" conaffinity="0"/>
    <geom name="rail" type="capsule" fromto="-1.5 0 0 1.5 0 0" size="0.02"
          rgba="0.35 0.35 0.4 1" contype="0" conaffinity="0"/>

    <!-- 小车：一维滑动关节 slider → qpos[0], qvel[0] -->
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"
             limited="true" range="-1.5 1.5"/>
      <geom name="cart" type="box" size="0.12 0.08 0.06" mass="1.0"
            rgba="0.2 0.45 0.85 1"/>
      <!-- 杆：铰链 hinge → qpos[1], qvel[1]；嵌在小车上方 -->
      <body name="pole" pos="0 0 0.06">
        <joint name="hinge" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6" size="0.03"
              mass="0.1" rgba="0.9 0.35 0.25 1"/>
        <geom name="tip" type="sphere" pos="0 0 0.6" size="0.04" mass="0.01"
              rgba="0.95 0.8 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- gear 放大控制量到力；env 里写 data.ctrl[0] ∈ [-1,1] -->
    <motor name="slide" joint="slider" gear="100"
           ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>

  <!-- sensor 块便于调试；本教程观测直接读 qpos/qvel，不必经 sensor -->
  <sensor>
    <jointpos name="cart_pos" joint="slider"/>
    <jointvel name="cart_vel" joint="slider"/>
    <jointpos name="pole_angle" joint="hinge"/>
    <jointvel name="pole_angvel" joint="hinge"/>
  </sensor>
</mujoco>
```

---

### A.2 `env.py` — Gymnasium 环境封装

```python
"""Gymnasium 兼容的 MuJoCo CartPole（平衡车 / 倒立摆）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# XML 与本文件相对定位，保证从任意 cwd 启动都能找到模型
ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_XML = ASSET_DIR / "cartpole.xml"

# qpos / qvel 下标：0=小车滑动，1=杆铰链（与 XML 中关节声明顺序一致）
CART_POS, POLE_ANGLE = 0, 1
CART_VEL, POLE_ANGVEL = 0, 1


class CartPoleMuJoCoEnv(gym.Env):
    """经典 CartPole，物理由 MuJoCo 积分。

    观测: [cart_x, cart_vx, pole_theta, pole_omega]
      - pole_theta = 0 表示竖直向上
    动作: Discrete(2) → 左力 (-1) / 右力 (+1)
    奖励: 杆未倒且车未出轨时每步 +1
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,          # 满分为 500
        angle_threshold: float = 12 * np.pi / 180,  # |θ| 超过约 12° 判失败
        x_threshold: float = 1.4,              # |x| 过大判失败（略小于导轨限位）
        force_mag: float = 1.0,                # |ctrl| 幅值
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.angle_threshold = float(angle_threshold)
        self.x_threshold = float(x_threshold)
        self.force_mag = float(force_mag)

        # MjModel=静态模型；MjData=动态状态（位置、速度、控制等）
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # 观测空间上界：位置放宽到 2*x_threshold；角速度不设有限上界
        high = np.array(
            [self.x_threshold * 2, np.finfo(np.float32).max, np.pi, np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._step_count = 0
        self._renderer: Optional[mujoco.Renderer] = None

    def _get_obs(self) -> np.ndarray:
        """从 MuJoCo 状态拼出 4 维观测向量。"""
        return np.array(
            [
                self.data.qpos[CART_POS],
                self.data.qvel[CART_VEL],
                self.data.qpos[POLE_ANGLE],
                self.data.qvel[POLE_ANGVEL],
            ],
            dtype=np.float32,
        )

    def _is_failed(self) -> bool:
        """是否触发 terminated（真正失败，而非到步数截断）。"""
        x = float(self.data.qpos[CART_POS])
        theta = float(self.data.qpos[POLE_ANGLE])
        return abs(x) > self.x_threshold or abs(theta) > self.angle_threshold

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)  # 初始化 self.np_random
        mujoco.mj_resetData(self.model, self.data)

        # 在竖直附近加小噪声，避免策略过拟合单一初值
        self.data.qpos[CART_POS] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[POLE_ANGLE] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[CART_VEL] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[POLE_ANGVEL] = self.np_random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)  # 由 qpos 推导相关量

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"invalid action {action}"
        # 离散动作 → 连续控制量（actuator ctrl）
        self.data.ctrl[0] = (-self.force_mag) if action == 0 else self.force_mag
        mujoco.mj_step(self.model, self.data)  # 推进一个物理步长
        self._step_count += 1

        obs = self._get_obs()
        terminated = self._is_failed()
        truncated = self._step_count >= self.max_episode_steps
        # 失败当步给 0，其余 +1（鼓励尽可能存活）
        reward = 0.0 if terminated else 1.0
        return obs, reward, terminated, truncated, {}

    def render(self):
        """可选离屏渲染；本教程主可视化走 mjviser，一般不用这里。"""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(seed: Optional[int] = None, **kwargs) -> CartPoleMuJoCoEnv:
    """工厂函数：可选地用 seed 做一次 reset。"""
    env = CartPoleMuJoCoEnv(**kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
```

---

### A.3 `ppo.py` — Actor-Critic 与 PPO-Clip

```python
"""面向离散 CartPole 的精简 PPO（Actor-Critic + Clip + GAE）。对照教程 §1。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """正交初始化权重（RL 里常见，利于训练稳定）。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """共享观测、分叉为策略头与价值头。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        # Actor：输出各动作 logits → Categorical 策略 π(a|s)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            # 最后一层用很小 std，避免初始策略过于尖锐
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        # Critic：标量 V(s)，用于 GAE / value loss
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return Categorical(logits=logits), value

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """采样动作，并返回 logπ、V（供 PPO 存旧策略概率与 GAE）。"""
        with torch.no_grad():
            dist, value = self.forward(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def greedy(self, obs: np.ndarray) -> int:
        """评估 / 可视化用：取概率最大的动作（argmax）。"""
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())


@dataclass
class RolloutBatch:
    """一次 rollout 收集、并算好 advantage/return 后的训练批次。"""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor   # 旧策略 logπ_old(a|s)
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor     # 旧 Critic 的 V(s)
    advantages: torch.Tensor
    returns: torch.Tensor    # advantage + value，给 Critic 回归


class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,          # 折扣 γ
        gae_lambda: float = 0.95,     # GAE 的 λ
        clip_eps: float = 0.2,        # PPO-Clip 的 ε
        ent_coef: float = 0.01,       # 熵奖励系数
        vf_coef: float = 0.5,         # 价值损失权重
        max_grad_norm: float = 0.5,   # 梯度裁剪
        update_epochs: int = 10,      # 每批数据复用轮数
        minibatch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

    def compute_gae(
        self,
        rewards: List[float],
        dones: List[bool],
        values: List[float],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """广义优势估计（从后向前递推）。见教程 §1.6。

        δ_t = r_t + γ V(s_{t+1}) (1-d_t) - V(s_t)
        A_t = δ_t + γλ (1-d_t) A_{t+1}
        R_t = A_t + V(s_t)
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - float(dones[t])  # episode 结束则下一价值断开
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """PPO-Clip 多 epoch 小批量更新。见教程 §1.5、§1.7。"""
        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        old_logprobs = batch.logprobs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        # 标准化优势，降低不同 rollout 尺度差异
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = obs.shape[0]
        idxs = np.arange(n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                mb = idxs[start : start + self.minibatch_size]
                dist, values = self.net(obs[mb])
                logprobs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                # r_t(θ) = π_θ / π_old = exp(logπ_new - logπ_old)
                ratio = torch.exp(logprobs - old_logprobs[mb])
                adv = advantages[mb]
                surr1 = ratio * adv
                # clip 把比率限制在 [1-ε, 1+ε]，抑制过大更新
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                # 最大化 E[min(surr1,surr2)] ≡ 最小化其相反数
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - returns[mb]) ** 2).mean()
                # 总损失：策略 + c_v * 价值 - c_e * 熵（减熵 = 鼓励探索）
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in metrics.items()}

    def save(self, path: str) -> None:
        torch.save({"model": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
```

---

### A.4 `train.py` — 采集 Rollout、更新、早停与存盘

```python
#!/usr/bin/env python3
"""在 MuJoCo CartPole 上训练 PPO。"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from env import CartPoleMuJoCoEnv
from ppo import PPO, RolloutBatch

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO CartPole with MuJoCo")
    p.add_argument("--total-steps", type=int, default=80_000)   # 环境步上限
    p.add_argument("--rollout-steps", type=int, default=2048)   # 每次更新前交互长度
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--log-dir", type=Path, default=ROOT / "runs")
    # 最近 solved_window 局平均回报 ≥ solved_reward 则早停
    p.add_argument("--solved-reward", type=float, default=475.0)
    p.add_argument("--solved-window", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CartPoleMuJoCoEnv()
    obs, _ = env.reset(seed=args.seed)
    agent = PPO(
        obs_dim=env.observation_space.shape[0],  # 4
        act_dim=env.action_space.n,              # 2
        lr=args.lr,
        device=args.device,
    )

    # 滑动窗口统计，用于日志与早停（与 avgR100 窗口可不同）
    ep_returns: deque[float] = deque(maxlen=100)
    ep_lens: deque[int] = deque(maxlen=100)
    recent_for_solve: deque[float] = deque(maxlen=args.solved_window)
    history = []

    global_step = 0
    ep_ret = 0.0
    ep_len = 0
    best_avg = -1e9
    t0 = time.time()

    print(
        f"Train CartPole | total_steps={args.total_steps} rollout={args.rollout_steps} "
        f"device={args.device}"
    )

    while global_step < args.total_steps:
        # ---------- 1) 用当前策略采集一段 on-policy 轨迹 ----------
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        for _ in range(args.rollout_steps):
            action, logprob, value = agent.net.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logprob)       # 旧策略 logπ，更新时算 ratio
            rew_buf.append(float(reward))
            done_buf.append(bool(done))
            val_buf.append(value)

            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1
            global_step += 1

            if done:
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                recent_for_solve.append(ep_ret)
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_len = 0

            if global_step >= args.total_steps:
                break

        # rollout 末尾若未 done，需要 bootstrap：用 V(s_last) 补全 GAE
        with torch.no_grad():
            _, last_value = agent.net.forward(torch.as_tensor(obs, dtype=torch.float32))
            last_value = float(last_value.item())

        # ---------- 2) GAE → 3) PPO update ----------
        advantages, returns = agent.compute_gae(rew_buf, done_buf, val_buf, last_value)
        batch = RolloutBatch(
            obs=torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32),
            actions=torch.as_tensor(np.asarray(act_buf), dtype=torch.int64),
            logprobs=torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32),
            rewards=torch.as_tensor(np.asarray(rew_buf), dtype=torch.float32),
            dones=torch.as_tensor(np.asarray(done_buf), dtype=torch.float32),
            values=torch.as_tensor(np.asarray(val_buf), dtype=torch.float32),
            advantages=torch.as_tensor(advantages, dtype=torch.float32),
            returns=torch.as_tensor(returns, dtype=torch.float32),
        )
        metrics = agent.update(batch)

        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        row = {
            "step": global_step,
            "avg_return_100": avg_ret,
            "avg_len_100": avg_len,
            "episodes": len(ep_returns),
            **metrics,
        }
        history.append(row)
        elapsed = time.time() - t0
        print(
            f"step={global_step:6d}  avgR100={avg_ret:7.1f}  avgLen={avg_len:6.1f}  "
            f"pi={metrics['policy_loss']:.3f}  v={metrics['value_loss']:.3f}  "
            f"H={metrics['entropy']:.3f}  t={elapsed:.1f}s"
        )

        # 按「最近 100 局均值」存一份过程最优（早期局会拉低该均值）
        if avg_ret > best_avg and len(ep_returns) >= 10:
            best_avg = avg_ret
            best_path = args.save_dir / "cartpole_ppo_best.pt"
            agent.save(str(best_path))

        # 早停：看最近 window 局是否已经接近满分
        if len(recent_for_solve) == args.solved_window and np.mean(recent_for_solve) >= args.solved_reward:
            print(
                f"Solved: mean return over last {args.solved_window} eps "
                f"= {np.mean(recent_for_solve):.1f} >= {args.solved_reward}"
            )
            agent.save(str(args.save_dir / "cartpole_ppo_best.pt"))
            best_avg = float(np.mean(recent_for_solve))
            break

    final_path = args.save_dir / "cartpole_ppo_final.pt"
    agent.save(str(final_path))
    # 结束时再写一份 best，与最终策略对齐，便于 evaluate / 可视化默认加载
    agent.save(str(args.save_dir / "cartpole_ppo_best.pt"))
    log_path = args.log_dir / "train_history.json"
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    env.close()
    print(f"Saved final → {final_path}")
    print(f"Saved best  → {args.save_dir / 'cartpole_ppo_best.pt'} (avgR={best_avg:.1f})")
    print(f"History     → {log_path}")


if __name__ == "__main__":
    main()
```

---

### A.5 `evaluate.py` — 无界面评估

```python
#!/usr/bin/env python3
"""加载 checkpoint，在无渲染环境下跑多局，打印回报统计。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import CartPoleMuJoCoEnv
from ppo import PPO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cartpole_ppo_best.pt",
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="使用 argmax，方差更小，适合验收")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = CartPoleMuJoCoEnv()
    agent = PPO(obs_dim=4, act_dim=2)
    agent.load(str(args.checkpoint))

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)  # 每局不同种子，覆盖不同初值
        done = False
        ep_ret = 0.0
        while not done:
            if args.greedy:
                action = agent.net.greedy(obs)
            else:
                action, _, _ = agent.net.act(obs)  # 仍按随机策略采样
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
        print(f"episode {ep+1:02d}: return={ep_ret:.0f}")

    print(
        f"mean={np.mean(returns):.1f}  std={np.std(returns):.1f}  "
        f"min={np.min(returns):.0f}  max={np.max(returns):.0f}"
    )
    env.close()


if __name__ == "__main__":
    main()
```

---

### A.6 `visualize_mjviser.py` — 策略驱动 + 端口 6008 网页可视化

```python
#!/usr/bin/env python3
"""用 mjviser 在浏览器中观看训练好的平衡策略。

默认端口 6008（本教程验收端口）。启动前请确保该端口空闲。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import viser
from mjviser import Viewer

from env import DEFAULT_XML, CartPoleMuJoCoEnv
from ppo import PPO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CartPole PPO with mjviser")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cartpole_ppo_best.pt",
    )
    p.add_argument("--port", type=int, default=6008)
    p.add_argument("--host", type=str, default="0.0.0.0")  # 便于 AutoDL 映射访问
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="按策略分布采样；默认贪婪 argmax",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 可视化时把单局步数上限放大，避免「撑满 500 就截断」打断观看
    env = CartPoleMuJoCoEnv(xml_path=DEFAULT_XML, max_episode_steps=100_000)
    agent = PPO(obs_dim=4, act_dim=2)
    if args.checkpoint.exists():
        agent.load(str(args.checkpoint))
        print(f"Loaded policy: {args.checkpoint}")
    else:
        print(f"WARNING: checkpoint not found ({args.checkpoint}), using random policy")

    obs, _ = env.reset(seed=args.seed)
    # 必须把同一份 model/data 交给 Viewer，这样策略写的 ctrl 与画面一致
    model, data = env.model, env.data
    episode = 0
    ep_ret = 0.0

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """替换 Viewer 默认的 mujoco.mj_step：先问策略再 env.step。

        mjviser 按仿真时间预算反复调用本函数；不要在这里 sleep。
        """
        nonlocal obs, episode, ep_ret
        if args.stochastic:
            action, _, _ = agent.net.act(obs)
        else:
            action = agent.net.greedy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += float(reward)
        if terminated or truncated:
            print(f"[episode {episode}] return={ep_ret:.0f}")
            episode += 1
            ep_ret = 0.0
            obs, _ = env.reset()

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """GUI 点 Reset 时同步重置环境与回合统计。"""
        nonlocal obs, ep_ret
        obs, _ = env.reset()
        ep_ret = 0.0
        print("Reset from viewer GUI")

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"mjviser CartPole → http://localhost:{args.port}")
    Viewer(model, data, server=server, step_fn=step_fn, reset_fn=reset_fn).run()


if __name__ == "__main__":
    main()
```

---

### A.7 附录阅读建议

1. 先扫 **A.1 XML**，建立「关节下标 ↔ 观测分量」心智模型。  
2. 读 **A.2 `step/reset`**，对齐奖励与终止条件。  
3. 带着 §1 公式读 **A.3 `compute_gae` / `update`**。  
4. 用 **A.4** 看清「采集 → GAE → 更新 → 早停」主循环。  
5. **A.5 / A.6** 是同一策略的两种消费方式：统计验收 vs 端口 **6008** 观看。

---

*文档依据 2026-08-21 AutoDL 本机实测（PPO 早停 ≈59k 步、评估 20×500、可视化端口 6008）整理；附录源码注释版与 `cartpole_rl/` 目录同步。*
