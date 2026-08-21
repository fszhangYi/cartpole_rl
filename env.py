"""Gymnasium-compatible MuJoCo CartPole（离散或连续力）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_XML = ASSET_DIR / "cartpole.xml"

CART_POS, POLE_ANGLE = 0, 1
CART_VEL, POLE_ANGVEL = 0, 1

ActionType = Union[int, float, np.ndarray]


class CartPoleMuJoCoEnv(gym.Env):
    """CartPole：discrete=左右恒力；continuous=ctrl∈[-1,1]。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        angle_threshold: float = 12 * np.pi / 180,
        x_threshold: float = 1.4,
        force_mag: float = 1.0,
        continuous: bool = False,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.angle_threshold = float(angle_threshold)
        self.x_threshold = float(x_threshold)
        self.force_mag = float(force_mag)
        self.continuous = bool(continuous)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        high = np.array(
            [self.x_threshold * 2, np.finfo(np.float32).max, np.pi, np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        if self.continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)

        self._step_count = 0
        self._renderer: Optional[mujoco.Renderer] = None

    def _get_obs(self) -> np.ndarray:
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
        x = float(self.data.qpos[CART_POS])
        theta = float(self.data.qpos[POLE_ANGLE])
        return abs(x) > self.x_threshold or abs(theta) > self.angle_threshold

    def _apply_action(self, action: ActionType) -> None:
        if self.continuous:
            a = float(np.asarray(action).reshape(-1)[0])
            a = float(np.clip(a, -1.0, 1.0))
            self.data.ctrl[0] = a * self.force_mag
        else:
            action_i = int(action)
            assert self.action_space.contains(action_i), f"invalid action {action_i}"
            self.data.ctrl[0] = (-self.force_mag) if action_i == 0 else self.force_mag

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[CART_POS] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[POLE_ANGLE] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[CART_VEL] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[POLE_ANGVEL] = self.np_random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: ActionType) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._get_obs()
        terminated = self._is_failed()
        truncated = self._step_count >= self.max_episode_steps
        reward = 0.0 if terminated else 1.0
        return obs, reward, terminated, truncated, {}

    def render(self):
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
    env = CartPoleMuJoCoEnv(**kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
