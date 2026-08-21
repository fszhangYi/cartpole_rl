# MuJoCo 平衡车（CartPole）强化学习

经典倒立摆 / 平衡车：小车左右施力，保持杆竖直。物理用 **MuJoCo**，算法为 **PPO**，可视化用 **mjviser**。

**详细教程（含 PPO 算法讲解）：** `./CartPole_PPO_手把手教程.md`

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
cd /root/autodl-tmp/cartpole_rl
```

依赖（若尚未安装）：`mujoco`、`mjviser`、`gymnasium`、`torch`（CPU 即可）。

## 任务设定

| 项 | 内容 |
| --- | --- |
| 状态 | `[x, vx, θ, ω]` 小车位置/速度、杆角/角速度（θ=0 为竖直） |
| 动作 | 离散 2：向左 / 向右恒力 |
| 奖励 | 杆未倒且小车未出轨：每步 +1 |
| 终止 | `|θ| > 12°` 或 `|x| > 1.4`，或满 500 步 |
| 模型 | `assets/cartpole.xml` |

## 训练

```bash
python train.py --total-steps 80000
```

达标：最近 20 局平均回报 ≥ 475 时提前结束。权重写入：

- `checkpoints/cartpole_ppo_best.pt`
- `checkpoints/cartpole_ppo_final.pt`
- `runs/train_history.json`

## 评估（无界面）

```bash
python evaluate.py --checkpoint checkpoints/cartpole_ppo_best.pt --episodes 20 --greedy
```

期望：多数回合回报接近 **500**。

## mjviser 可视化（默认端口 6008）

```bash
python visualize_mjviser.py --port 6008
```

浏览器打开 `http://localhost:6008`（AutoDL 映射自定义服务端口 **6008**）。  
页面可 Pause / Reset / 调速；策略默认贪婪（argmax）。

## 目录

```text
cartpole_rl/
  assets/cartpole.xml      # MuJoCo 模型
  env.py                   # Gymnasium 环境
  ppo.py                   # PPO Actor-Critic
  train.py                 # 训练
  evaluate.py              # 评估
  visualize_mjviser.py     # 网页可视化
  checkpoints/             # 策略权重
  runs/                    # 训练曲线 JSON
```
