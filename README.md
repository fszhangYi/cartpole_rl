# MuJoCo 平衡车（CartPole）强化学习

经典倒立摆 / 平衡车：物理 **MuJoCo**，算法由 **`config.yaml`** 选择，可视化 **mjviser**。

## 文档（`docs/`）

| 文档 | 说明 |
| --- | --- |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | 零基础算法详解（逐算法） |
| [docs/CartPole_PPO_手把手教程.md](docs/CartPole_PPO_手把手教程.md) | CartPole + PPO 手把手教程 |
| [docs/MuJoCo_mjviser_安装教程.md](docs/MuJoCo_mjviser_安装教程.md) | MuJoCo / mjviser 环境安装 |

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
algorithm: ppo         # 详见 docs/ALGORITHMS.md
```

大改动集中在代码里的 **`match/case`**（类似 switch）：

| 位置 | 作用 |
| --- | --- |
| `algorithms/factory.py` → `create_agent` | 按算法名创建智能体 |
| `train.py` → `main` | 选择对应训练循环 |
| `config_loader.py` → `resolve_checkpoint_path` | `.pt` vs `.npz` 权重后缀 |

评估与可视化 **复用** 统一接口 `agent.select_action(obs, greedy=...)`。

```bash
python train.py --config config.yaml
python train.py --algorithm ppo
python train.py --algorithm q_learning
python evaluate.py --greedy
python visualize_mjviser.py --port 6008
```

权重命名：`checkpoints/cartpole_{algorithm}_best.pt|.npz`。

## 任务设定

| 项 | 内容 |
| --- | --- |
| 状态 | `[x, vx, θ, ω]`（θ=0 为竖直） |
| 动作 | 离散 2：左 / 右（连续算法见文档） |
| 奖励 | 存活每步 +1 |
| 终止 | `|θ| > 12°` 或 `|x| > 1.4`，或满 500 步 |

## 目录

```text
cartpole_rl/
  README.md
  docs/                    # Markdown 教程与算法说明
    ALGORITHMS.md
    CartPole_PPO_手把手教程.md
    MuJoCo_mjviser_安装教程.md
  config.yaml
  algorithms/              # 算法包
  training_loops.py
  env.py / assets/
  train.py / evaluate.py / visualize_mjviser.py
  checkpoints/ runs/
```
