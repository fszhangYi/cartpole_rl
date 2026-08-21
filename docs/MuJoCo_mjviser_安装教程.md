# MuJoCo + mjviser 手把手安装教程（本机实测细化版）

> 组件：[MuJoCo](https://github.com/google-deepmind/mujoco)（物理仿真）+ [mjviser](https://github.com/mujocolab/mjviser)（基于 [Viser](https://github.com/viser-project/viser) 的网页版查看器）  
> PyPI：[`mujoco`](https://pypi.org/project/mujoco/) · [`mjviser`](https://pypi.org/project/mjviser/)  
> 实测机：AutoDL；数据盘 `/root/autodl-tmp`（约 150G）+ 系统盘 `/`（约 30G）  
> **本教程验收端口：`6008`**  
> **进度（2026-08-21）：** conda 环境创建 → pip 安装 mujoco / mjviser → 本机示例 XML 在 `6008` 启动并通过验收。

---

## 这项工作是什么：你装完能做什么

读操作步骤之前，先弄清三件事：

| 组件 | 是什么 | 本教程用途 |
| --- | --- | --- |
| **MuJoCo** | DeepMind 开源的刚体接触物理引擎；Python 绑定可加载 `.xml` / `.mjb` 模型并步进仿真 | 作为仿真后端，被 mjviser 调用 |
| **mjviser** | 把 MuJoCo 场景挂到浏览器里的交互查看器（关节滑条、接触可视化、相机跟踪等） | 在网页打开模型；本机验收用端口 **6008** |
| **robot_descriptions**（可选） | 一键按名字拉取常见机器人 MuJoCo 模型（如 `go1`、`franka_emika_panda`） | 没有自备 XML 时用来加载公开模型 |

验收标准很具体：**环境能 `import mujoco`，能执行 `mjviser ... --port 6008`，浏览器打开 `http://localhost:6008`（或 AutoDL 自定义服务映射）能看到场景。**

---

## 读前须知

1. **本教程只覆盖「conda 环境 + MuJoCo + mjviser 网页查看」**，不涉及强化学习训练、mjx / mujoco-warp、或原生 OpenGL 桌面 viewer（`mujoco.viewer`）。
2. **分盘是硬要求（AutoDL）：** 系统盘常只有几 GB 空闲。环境与缓存请放到数据盘 `/root/autodl-tmp`，否则 `conda create` / `pip install` 中途磁盘满会失败。
3. **网络分流（AutoDL）：**  
   - 装 conda / pip 包：一般 **关掉** `source /etc/network_turbo` 代理，用清华等国内源更快。  
   - 若用 `robot_descriptions` 从 GitHub 拉模型：再 **打开** turbo 或自行配置代理。
4. **不要用系统 Python 硬装。** 独立 conda 环境可避免与 ComfyUI / 其他项目抢依赖。
5. **mjviser 需要一个 MODEL 参数**（XML 路径、模糊名、或 `robot_descriptions` 名）。首次验收建议用 MuJoCo 自带的 `testdata/model.xml`，无需联网下载。

### 本机实测版本（写教程时锁定）

| 项 | 版本 / 路径 |
| --- | --- |
| conda | 24.4.0（Miniconda：`/root/miniconda3`） |
| Python | 3.11.15 |
| 环境路径 | `/root/autodl-tmp/conda-envs/mjviser` |
| mujoco | 3.11.0 |
| mjviser | 0.0.14 |
| viser | 1.1.0 |
| robot_descriptions | 3.1.0（可选，已装） |
| 验收端口 | **6008** |
| 验收模型 | `.../site-packages/mujoco/testdata/model.xml` |

⏱ 本机实测总墙钟：约 **5–10 分钟**（已有 conda、清华 pip 源正常时）。

---

## 0. 前置检查

在开始前跑一遍，避免装到一半才发现盘满或端口占用。

### 0.1 磁盘

```bash
df -h / /root/autodl-tmp
```

建议：

- 系统盘 `/`：至少留 **2GB+**（conda 临时文件、系统库）。
- 数据盘 `/root/autodl-tmp`：至少留 **3GB+**（本环境实测大约占 **0.5–1GB** 量级；留足余量更稳）。

### 0.2 conda 是否可用

```bash
which conda
conda --version
```

若没有 conda，先安装 Miniconda（略），或改用 `venv` + `pip`（见文末「备选方案」）。

### 0.3 端口 6008 是否空闲

```bash
ss -tlnp | grep 6008 || echo "6008 空闲"
```

若已被占用，换端口（如 `6009`）或先结束旧进程：

```bash
# 查看占用进程后按需结束
pgrep -af mjviser
# kill <pid>
```

### 0.4（可选）清理失效的 conda 镜像频道

本机曾出现清华 **`anaconda/pkgs/free`** 返回 **HTTP 404**，导致 `conda create` 直接失败。先看频道：

```bash
conda config --show channels
```

若列表里有 `.../anaconda/pkgs/free`，本教程推荐 **不改全局配置**，创建环境时用 `--override-channels` 临时绕过（见 §1）。也可以自行从 `.condarc` 去掉该频道。

---

## 1. 创建 conda 环境（放到数据盘）

### 1.1 准备目录与环境变量

```bash
mkdir -p /root/autodl-tmp/conda-envs /root/autodl-tmp/conda-pkgs /root/autodl-tmp/logs

export CONDA_ENVS_DIRS=/root/autodl-tmp/conda-envs
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs
```

说明：

- `CONDA_ENVS_DIRS`：环境装到数据盘，避免撑爆系统盘。
- `CONDA_PKGS_DIRS`：下载包缓存也放数据盘，方便复用。

若希望长期生效，可写入 `~/.bashrc`（可选）：

```bash
echo 'export CONDA_ENVS_DIRS=/root/autodl-tmp/conda-envs' >> ~/.bashrc
echo 'export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs' >> ~/.bashrc
```

### 1.2 创建环境（Python 3.11）

MuJoCo 官方 wheel 对较新的 Python（如 3.14）可能尚未提供；**3.10 / 3.11 / 3.12** 通常最稳。本教程用 **3.11**。

**推荐命令（绕过失效的 free 频道）：**

```bash
conda create -y -p /root/autodl-tmp/conda-envs/mjviser python=3.11 \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c defaults
```

激活：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
python --version   # 期望：Python 3.11.x
which python       # 应指向 .../conda-envs/mjviser/bin/python
```

> 若你的 Miniconda 不在 `/root/miniconda3`，把 `source` 路径改成你机器上的 `conda.sh`。

### 1.3 若 `conda create` 报 `UnavailableInvalidChannel: .../pkgs/free`

原因：`.condarc` 里配置了已下线/404 的 `anaconda/pkgs/free`。  
处理：继续用上一节的 `--override-channels`；或编辑 `~/.condarc` 删除该行后再创建。

---

## 2. 安装 MuJoCo 与 mjviser

**确认已激活环境后再装包。** AutoDL 上若开了 turbo，先关掉代理再 pip：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset REQUESTS_CA_BUNDLE SSL_CERT_FILE
```

### 2.1 升级 pip（可选但推荐）

```bash
pip install -U pip
```

本机 pip 默认索引多为清华：`https://pypi.tuna.tsinghua.edu.cn/simple`。若没有镜像，可临时指定：

```bash
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.2 安装核心包

最小安装（够用）：

```bash
pip install mujoco mjviser
```

推荐安装（便于按名字加载公开机器人模型）：

```bash
pip install mujoco mjviser robot_descriptions
```

本机实测会拉到大致如下依赖链（版本随时间浮动，以你机器 `pip show` 为准）：

- `mujoco`（含 `glfw` / `pyopengl` / `numpy` 等）
- `mjviser` → `viser` → `websockets` / `trimesh` / `imageio` 等
- （可选）`robot_descriptions` → `GitPython` / `tqdm`

### 2.3 安装验收（import）

```bash
python - <<'EOF'
import mujoco
import mjviser
import importlib.metadata as m

print("mujoco", mujoco.__version__)
print("mjviser", m.version("mjviser"))
print("mjviser CLI:", __import__("shutil").which("mjviser"))
EOF
```

期望输出类似：

```text
mujoco 3.11.0
mjviser 0.0.14
mjviser CLI: /root/autodl-tmp/conda-envs/mjviser/bin/mjviser
```

再确认 CLI：

```bash
mjviser -h
```

应看到：

```text
usage: mjviser [-h] [--port PORT] MODEL
...
  --port PORT  port to bind the Viser server to (default: 8080)
```

---

## 3. 启动 mjviser（端口 6008）

### 3.1 选一个模型（三种方式）

#### 方式 A：用 MuJoCo 自带示例（**推荐首次验收，无需联网**）

```bash
MODEL=/root/autodl-tmp/conda-envs/mjviser/lib/python3.11/site-packages/mujoco/testdata/model.xml
ls -la "$MODEL"
```

#### 方式 B：你自己的 XML

```bash
MODEL=/绝对路径/你的模型.xml
```

注意：XML 里若 `meshdir` / `file=` 引用了相对路径资源，请在模型所在目录启动，或保证资源路径正确。

#### 方式 C：`robot_descriptions` 名字（需联网，首次可能 clone 仓库，较慢）

```bash
# 示例：go1 / franka_emika_panda / aloha 等
mjviser go1 --port 6008
```

> 实测：对 `humanoid` 这类名字，`robot_descriptions` 可能去 clone 完整 `google-deepmind/mujoco` 仓库，耗时长、占磁盘。首次验收优先用方式 A。

### 3.2 前台启动（方便看日志）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser

MODEL=/root/autodl-tmp/conda-envs/mjviser/lib/python3.11/site-packages/mujoco/testdata/model.xml
mjviser "$MODEL" --port 6008
```

成功时终端会出现类似：

```text
╭────── viser (listening *:6008) ───────╮
│             ╷                         │
│   HTTP      │ http://localhost:6008   │
│   Websocket │ ws://localhost:6008     │
│             ╵                         │
╰───────────────────────────────────────╯
```

Viser 默认监听 **`0.0.0.0`**，因此 AutoDL「自定义服务」映射端口后，外网/控制台跳转也能访问。

### 3.3 后台启动（推荐长期挂着）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser

MODEL=/root/autodl-tmp/conda-envs/mjviser/lib/python3.11/site-packages/mujoco/testdata/model.xml
mkdir -p /root/autodl-tmp/logs

nohup mjviser "$MODEL" --port 6008 \
  > /root/autodl-tmp/logs/mjviser_6008.log 2>&1 &

echo "PID=$!"
sleep 2
ss -tlnp | grep 6008
tail -n 30 /root/autodl-tmp/logs/mjviser_6008.log
```

期望：`ss` 中看到 `0.0.0.0:6008` 且进程名为 `mjviser`。

### 3.4 在浏览器打开

| 场景 | URL |
| --- | --- |
| 本机浏览器 / SSH 本地转发 | `http://localhost:6008` |
| AutoDL 控制台「自定义服务」 | 把实例端口填 **6008**，用平台给出的访问链接 |

页面里应能看到 MuJoCo 场景，以及仿真控制、关节/执行器相关 GUI（具体控件随 mjviser 版本略有差异）。

### 3.5 停止服务

```bash
pgrep -af mjviser
kill <pid>          # 温和结束
# kill -9 <pid>     # 仍不退出时再用
ss -tlnp | grep 6008 || echo "已停止"
```

---

## 4. 用 Python API 打开（可选）

除 CLI 外，也可在脚本里启动（同样可指定端口）：

```python
import mujoco
import viser
from mjviser import Viewer

MODEL = "/root/autodl-tmp/conda-envs/mjviser/lib/python3.11/site-packages/mujoco/testdata/model.xml"

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

server = viser.ViserServer(host="0.0.0.0", port=6008)
Viewer(model, data, server=server).run()
```

保存为例如 `/root/autodl-tmp/run_mjviser_6008.py` 后：

```bash
conda activate /root/autodl-tmp/conda-envs/mjviser
python /root/autodl-tmp/run_mjviser_6008.py
```

---

## 5. 常用命令速查

```bash
# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser

# 查看版本
python -c "import mujoco; import importlib.metadata as m; print(mujoco.__version__, m.version('mjviser'))"

# 启动（示例模型）
mjviser /root/autodl-tmp/conda-envs/mjviser/lib/python3.11/site-packages/mujoco/testdata/model.xml --port 6008

# 启动（自有模型）
mjviser /path/to/robot.xml --port 6008

# 启动（robot_descriptions 名，需已安装该包）
mjviser go1 --port 6008

# 看日志 / 端口
tail -f /root/autodl-tmp/logs/mjviser_6008.log
ss -tlnp | grep 6008
```

---

## 6. 坑点与排障

### 坑 A：`UnavailableInvalidChannel: .../anaconda/pkgs/free`（HTTP 404）

- **现象：** `conda create` 在 Collecting package metadata 阶段失败。  
- **原因：** 频道失效但仍写在 `~/.condarc`。  
- **处理：** 创建时加 `--override-channels` 并只用 `pkgs/main` + `defaults`（见 §1.2）。

### 坑 B：系统盘写满

- **现象：** `No space left on device`，或 conda/pip 解压失败。  
- **处理：** 环境与 `CONDA_PKGS_DIRS` 放到 `/root/autodl-tmp`；清理 `~/.cache/pip`、旧环境。

### 坑 C：`mjviser: command not found`

- **原因：** 未激活环境，或装到了别的 Python。  
- **处理：** `conda activate .../mjviser` 后 `which mjviser`；确认路径在该环境的 `bin/` 下。

### 坑 D：端口已被占用

- **现象：** 启动报 address already in use，或 `ss` 显示 6008 已被别的进程监听。  
- **处理：** `pgrep -af mjviser` 后 `kill`，或改 `--port`。

### 坑 E：浏览器打不开 / 白屏

1. 确认进程仍在：`ss -tlnp | grep 6008`。  
2. AutoDL 是否在控制台正确映射了 **6008**。  
3. 看日志：`/root/autodl-tmp/logs/mjviser_6008.log`。  
4. 本机可用 SSH 隧道：`ssh -L 6008:127.0.0.1:6008 user@host`。

### 坑 F：`robot_descriptions` 卡住很久

- **现象：** 日志出现 `Cloning https://github.com/...`，长时间无端口监听。  
- **原因：** 首次按名加载会 git clone 上游仓库。  
- **处理：** 开 turbo / 代理；或改用本地已有 XML（§3.1 方式 A/B）。

### 坑 G：AutoDL turbo 导致 pip 极慢或 SSL 错

- **处理：** 装包前 `unset` 全部代理与 `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`（见 §2）；用清华 PyPI。

### 坑 H：Python 版本过新，MuJoCo 无 wheel

- **现象：** pip 尝试从源码编译并失败。  
- **处理：** 环境改用 Python 3.11 或 3.12 重建。

---

## 7. 卸载 / 重建

只删环境（不影响其他项目）：

```bash
conda deactivate
conda env remove -p /root/autodl-tmp/conda-envs/mjviser
# 或直接：
# rm -rf /root/autodl-tmp/conda-envs/mjviser
```

然后从 §1 重新创建即可。

---

## 8. 备选方案：不用 conda，用 venv

若坚持不用 conda：

```bash
mkdir -p /root/autodl-tmp/venvs
python3.11 -m venv /root/autodl-tmp/venvs/mjviser
source /root/autodl-tmp/venvs/mjviser/bin/activate
pip install -U pip
pip install mujoco mjviser robot_descriptions
mjviser /path/to/model.xml --port 6008
```

（需系统已有 `python3.11`；否则仍建议用 conda 创建指定版本。）

---

## 9. 一键脚本（可选）

把下列内容保存为 `/root/autodl-tmp/setup_mjviser.sh`，可复现本教程主路径：

```bash
#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX=/root/autodl-tmp/conda-envs/mjviser
PKGS_DIR=/root/autodl-tmp/conda-pkgs
LOG_DIR=/root/autodl-tmp/logs
PORT=6008

mkdir -p "$(dirname "$ENV_PREFIX")" "$PKGS_DIR" "$LOG_DIR"
export CONDA_ENVS_DIRS="$(dirname "$ENV_PREFIX")"
export CONDA_PKGS_DIRS="$PKGS_DIR"

# 装包阶段建议关代理（AutoDL）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
unset REQUESTS_CA_BUNDLE SSL_CERT_FILE || true

source /root/miniconda3/etc/profile.d/conda.sh

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.11 \
    --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    -c defaults
fi

conda activate "$ENV_PREFIX"
pip install -U pip
pip install mujoco mjviser robot_descriptions

MODEL="$ENV_PREFIX/lib/python3.11/site-packages/mujoco/testdata/model.xml"
test -f "$MODEL"

# 若端口已有 mjviser 在听，则跳过启动
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已在监听，跳过启动。"
else
  nohup mjviser "$MODEL" --port "$PORT" > "$LOG_DIR/mjviser_${PORT}.log" 2>&1 &
  sleep 2
fi

echo "---- 验收 ----"
python -c "import mujoco, importlib.metadata as m; print('mujoco', mujoco.__version__, 'mjviser', m.version('mjviser'))"
ss -tlnp | grep ":${PORT}" || true
echo "打开: http://localhost:${PORT}"
```

使用：

```bash
chmod +x /root/autodl-tmp/setup_mjviser.sh
bash /root/autodl-tmp/setup_mjviser.sh
```

---

## 10. 验收清单（打勾即完成）

- [ ] `conda activate /root/autodl-tmp/conda-envs/mjviser` 成功  
- [ ] `python -c "import mujoco; print(mujoco.__version__)"` 有版本号  
- [ ] `mjviser -h` 能显示 `--port`  
- [ ] `ss -tlnp | grep 6008` 显示 `0.0.0.0:6008`  
- [ ] 浏览器打开 `http://localhost:6008`（或 AutoDL 映射链接）能看到场景  

全部勾上后，本教程安装目标即达成。

---

## 参考链接

- MuJoCo 文档：<https://mujoco.readthedocs.io/>  
- mjviser 仓库：<https://github.com/mujocolab/mjviser>  
- Viser：<https://github.com/viser-project/viser>  
- robot_descriptions：<https://github.com/robot-descriptions/robot_descriptions.py>  
- MuJoCo Python 绑定说明：模型用 `mujoco.MjModel.from_xml_path`，数据用 `mujoco.MjData`  

---

*文档依据 2026-08-21 本机 AutoDL 实测步骤整理；依赖小版本号会随 PyPI 更新，以你环境中 `pip show mujoco mjviser` 为准。*
