"""Gymnasium-compatible MuJoCo CartPole (平衡车 / 倒立摆)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_XML = ASSET_DIR / "cartpole.xml"

# Indices in qpos / qvel: slider, hinge
CART_POS, POLE_ANGLE = 0, 1
CART_VEL, POLE_ANGVEL = 0, 1


class CartPoleMuJoCoEnv(gym.Env):
    """Classic CartPole with MuJoCo physics.

    Observation: [cart_x, cart_vx, pole_theta, pole_omega]
      - pole_theta = 0 means upright (MuJoCo hinge zero)
    Action: Discrete(2) → force left (-1) or right (+1)
    Reward: +1 every step the pole stays up and cart stays on track
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        angle_threshold: float = 12 * np.pi / 180,  # ~12 degrees
        x_threshold: float = 1.4,
        force_mag: float = 1.0,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.angle_threshold = float(angle_threshold)
        self.x_threshold = float(x_threshold)
        self.force_mag = float(force_mag)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        high = np.array(
            [self.x_threshold * 2, np.finfo(np.float32).max, np.pi, np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Small random init around upright (theta≈0)
        self.data.qpos[CART_POS] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[POLE_ANGLE] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[CART_VEL] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[POLE_ANGVEL] = self.np_random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"invalid action {action}"
        # 0 → left (-force), 1 → right (+force)
        self.data.ctrl[0] = (-self.force_mag) if action == 0 else self.force_mag
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
