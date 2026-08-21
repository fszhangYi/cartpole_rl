# MuJoCo 平衡车（CartPole）强化学习

经典倒立摆 / 平衡车：物理 **MuJoCo**，算法由 **`config.yaml`** 选择（**PPO** 或 **Q-Learning**），可视化 **mjviser**。

**详细教程：** [CartPole_PPO_手把手教程.md](./CartPole_PPO_手把手教程.md)

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
cd /root/autodl-tmp/cartpole_rl
pip install pyyaml   # 若尚未安装
```

## 用配置切换算法（推荐）

编辑 `config.yaml` 顶部：

```yaml
algorithm: ppo         # 或 q_learning
```

大改动集中在代码里的 **`match/case`**（类似 switch）：

| 位置 | 作用 |
| --- | --- |
| `algorithms/factory.py` → `create_agent` | 创建 PPO / QLearningAgent |
| `train.py` → `main` | 选择 `train_ppo` / `train_q_learning` |
| `config_loader.py` → `resolve_checkpoint_path` | `.pt` vs `.npz` 权重后缀 |

评估与可视化 **复用** 统一接口 `agent.select_action(obs, greedy=...)`，不再写死 PPO。

```bash
# 按配置训练
python train.py --config config.yaml

# 命令行覆盖算法（不改文件）
python train.py --algorithm ppo
python train.py --algorithm q_learning

# 评估 / 可视化（默认读 config 里的 algorithm 与对应权重）
python evaluate.py --greedy
python visualize_mjviser.py --port 6008
```

权重命名：`checkpoints/cartpole_{algorithm}_best.pt|.npz`。

## 任务设定

| 项 | 内容 |
| --- | --- |
| 状态 | `[x, vx, θ, ω]`（θ=0 为竖直） |
| 动作 | 离散 2：左 / 右 |
| 奖励 | 存活每步 +1 |
| 终止 | `|θ| > 12°` 或 `|x| > 1.4`，或满 500 步 |

## 目录

```text
cartpole_rl/
  config.yaml              # 算法与超参
  config_loader.py         # 加载 / 规范化算法名
  algorithms/              # ★ 算法包（可插拔）
    __init__.py
    base.py                # Agent 协议
    factory.py             # match/case 创建智能体
    ppo.py
    qlearning.py
  env.py / assets/         # 环境（算法无关，复用）
  train.py                 # match/case 训练入口
  evaluate.py / visualize_mjviser.py
  checkpoints/ runs/
```
