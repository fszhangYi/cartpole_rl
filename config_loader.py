"""算法名规范化、连续动作判定、权重后缀。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, FrozenSet

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# 已实现
IMPLEMENTED = (
    "ppo",
    "q_learning",
    "sarsa",
    "sarsa_lambda",
    "dyna_q",
    "monte_carlo",
    "value_iteration",
    "policy_iteration",
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "rainbow",
    "reinforce",
    "a2c",
    "trpo",
    "ddpg",
    "td3",
    "sac",
    "mpc",
)

# 明确不实现 / 仅文档说明
NOT_IMPLEMENTED = {
    "a3c": "需要异步多进程 Actor-Learner；本仓库提供同步版 A2C 作为单机替代。",
    "impala": "面向大规模分布式（V-trace）；单机 CartPole 无意义且工程量过大。",
    "td0": "TD(0) 是更新规则而非完整控制算法；请用 sarsa（控制）或价值预测教程。",
    "rainbow_full": "完整 Rainbow 含 C51/NoisyNet 等；本仓库 rainbow 为 Double+Dueling+n-step+PER 精简版。",
}

CONTINUOUS_ALGOS: FrozenSet[str] = frozenset({"ddpg", "td3", "sac", "mpc"})
TABULAR_NPZ: FrozenSet[str] = frozenset(
    {
        "q_learning",
        "sarsa",
        "sarsa_lambda",
        "dyna_q",
        "monte_carlo",
        "value_iteration",
        "policy_iteration",
        "mpc",
    }
)

ALIASES = {
    "q": "q_learning",
    "ql": "q_learning",
    "qlearning": "q_learning",
    "sarsa_l": "sarsa_lambda",
    "td_lambda": "sarsa_lambda",
    "tdlambda": "sarsa_lambda",
    "dynaq": "dyna_q",
    "mc": "monte_carlo",
    "vi": "value_iteration",
    "pi": "policy_iteration",
    "double": "double_dqn",
    "dueling": "dueling_dqn",
    "rainbow_lite": "rainbow",
}


def normalize_algorithm(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = ALIASES.get(key, key)
    if key in NOT_IMPLEMENTED:
        raise ValueError(f"算法 {name!r} 未实现：{NOT_IMPLEMENTED[key]}")
    if key not in IMPLEMENTED:
        raise ValueError(
            f"不支持的算法: {name!r}\n可选: {', '.join(IMPLEMENTED)}\n"
            f"未实现说明: {NOT_IMPLEMENTED}"
        )
    return key


def needs_continuous_action(algorithm: str) -> bool:
    return algorithm in CONTINUOUS_ALGOS


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {cfg_path}")
    cfg["algorithm"] = normalize_algorithm(str(cfg.get("algorithm", "ppo")))
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def checkpoint_stem(algorithm: str, tag: str = "best") -> str:
    return f"cartpole_{algorithm}_{tag}"


def resolve_checkpoint_path(cfg: dict[str, Any], tag: str = "best") -> Path:
    viz = cfg.get("visualize") or {}
    explicit = viz.get("checkpoint")
    save_dir = ROOT / str((cfg.get("train") or {}).get("save_dir", "checkpoints"))
    algo = cfg["algorithm"]
    if explicit:
        return Path(explicit)
    ext = ".npz" if algo in TABULAR_NPZ else ".pt"
    return save_dir / f"{checkpoint_stem(algo, tag)}{ext}"


SUPPORTED_ALGORITHMS = IMPLEMENTED  # 兼容旧名
