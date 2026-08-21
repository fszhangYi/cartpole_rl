"""加载 YAML 配置；命令行可覆盖 algorithm 等字段。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# 统一算法名
ALG_PPO = "ppo"
ALG_Q_LEARNING = "q_learning"
SUPPORTED_ALGORITHMS = (ALG_PPO, ALG_Q_LEARNING)


def normalize_algorithm(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    match key:
        case "ppo":
            return ALG_PPO
        case "q_learning" | "qlearning" | "q" | "ql":
            return ALG_Q_LEARNING
        case _:
            raise ValueError(
                f"不支持的算法: {name!r}，可选: {', '.join(SUPPORTED_ALGORITHMS)}"
            )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {cfg_path}")
    cfg["algorithm"] = normalize_algorithm(str(cfg.get("algorithm", ALG_PPO)))
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def checkpoint_stem(algorithm: str, tag: str = "best") -> str:
    """例如 cartpole_ppo_best / cartpole_q_learning_best。"""
    return f"cartpole_{algorithm}_{tag}"


def resolve_checkpoint_path(cfg: dict[str, Any], tag: str = "best") -> Path:
    """按算法解析权重路径（PPO→.pt，Q-Learning→.npz）。"""
    viz = cfg.get("visualize") or {}
    explicit = viz.get("checkpoint")
    save_dir = ROOT / str((cfg.get("train") or {}).get("save_dir", "checkpoints"))
    algo = cfg["algorithm"]

    if explicit:
        return Path(explicit)

    match algo:
        case "ppo":
            return save_dir / f"{checkpoint_stem(algo, tag)}.pt"
        case "q_learning":
            return save_dir / f"{checkpoint_stem(algo, tag)}.npz"
        case _:
            raise ValueError(algo)
