from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyBullet is required for this environment. Install with: pip install pybullet"
    ) from e


@dataclass
class PandaGraspConfig:
    render: bool = False
    max_steps: int = 200
    control_dt: float = 1.0 / 60.0
    sim_substeps: int = 4
    action_scale_xyz: float = 0.05

    # workspace box for end-effector target
    ee_min: np.ndarray = np.array([0.35, -0.30, 0.02], dtype=np.float32)
    ee_max: np.ndarray = np.array([0.75, 0.30, 0.55], dtype=np.float32)

    # success definition
    lift_height_success: float = 0.12


class PandaGraspEnv(gym.Env):
    """
    Minimal PyBullet grasp environment (dense reward) for demo/training.

    - Robot: Franka Panda
    - Object: small cube on table
    - Action (4D): delta_xyz in [-1,1]^3 and gripper command in [-1,1]
    - Observation: ee_pos(3), cube_pos(3), ee_to_cube(3), gripper_open(1), has_contact(1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: PandaGraspConfig | None = None):
        super().__init__()
        self.cfg = cfg or PandaGraspConfig()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self._cid: int | None = None
        self._panda_id: int | None = None
        self._cube_id: int | None = None

        self._ee_link = 11
        self._arm_joints = list(range(7))
        self._finger_joints = [9, 10]

        self._ee_target = np.array([0.55, 0.0, 0.25], dtype=np.float32)
        self._step = 0

    def _connect(self):
        if self._cid is not None:
            return
        self._cid = p.connect(p.GUI if self.cfg.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.cfg.control_dt / self.cfg.sim_substeps)

    def _disconnect(self):  # pragma: no cover
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None

    def close(self):  # pragma: no cover
        self._disconnect()

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._connect()
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, -0.65])

        self._panda_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

        # Small cube near the robot
        cube_xyz = np.array([0.62, self.np_random.uniform(-0.08, 0.08), 0.02], dtype=np.float32)
        self._cube_id = p.loadURDF("cube.urdf", cube_xyz, globalScaling=0.05)
        p.changeDynamics(self._cube_id, -1, mass=0.1, lateralFriction=1.0, spinningFriction=0.1)

        # Initial arm pose (from `demo_scripted_grasp.py`)
        init = [-0.785, -0.785, 0.785, -1.57, -0.785, 1.57, 0.785]
        for j, q in enumerate(init):
            p.resetJointState(self._panda_id, j, q)

        # Open gripper initially
        self._set_gripper(open_gripper=True)

        ee_pos = np.array(p.getLinkState(self._panda_id, self._ee_link)[0], dtype=np.float32)
        self._ee_target = np.clip(ee_pos, self.cfg.ee_min, self.cfg.ee_max)

        self._step = 0
        obs = self._get_obs()
        info = {"is_success": False}
        return obs, info

    def _set_gripper(self, open_gripper: bool):
        assert self._panda_id is not None
        target = 0.04 if open_gripper else 0.0
        for j in self._finger_joints:
            p.setJointMotorControl2(
                self._panda_id, j, p.POSITION_CONTROL, targetPosition=target, force=100
            )

    def _apply_ee_target(self, ee_xyz: np.ndarray):
        assert self._panda_id is not None
        ee_xyz = np.clip(ee_xyz.astype(np.float32), self.cfg.ee_min, self.cfg.ee_max)
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        q = p.calculateInverseKinematics(self._panda_id, self._ee_link, ee_xyz, orn)
        for i, j in enumerate(self._arm_joints):
            p.setJointMotorControl2(
                self._panda_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(q[i]),
                force=500,
                maxVelocity=1.0,
            )

    def _has_contact(self) -> bool:
        assert self._panda_id is not None and self._cube_id is not None
        pts = p.getContactPoints(self._panda_id, self._cube_id)
        return len(pts) > 0

    def _get_obs(self) -> np.ndarray:
        assert self._panda_id is not None and self._cube_id is not None
        ee_pos = np.array(p.getLinkState(self._panda_id, self._ee_link)[0], dtype=np.float32)
        cube_pos = np.array(p.getBasePositionAndOrientation(self._cube_id)[0], dtype=np.float32)
        rel = cube_pos - ee_pos

        # Approximate gripper openness from joint pos
        j0 = p.getJointState(self._panda_id, self._finger_joints[0])[0]
        gripper_open = np.clip(float(j0) / 0.04, 0.0, 1.0)

        contact = 1.0 if self._has_contact() else 0.0
        return np.concatenate([ee_pos, cube_pos, rel, [gripper_open, contact]]).astype(np.float32)

    def step(self, action: np.ndarray):
        assert self._panda_id is not None and self._cube_id is not None
        self._step += 1

        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        dxyz = action[:3] * self.cfg.action_scale_xyz
        grip = float(action[3])

        self._ee_target = np.clip(self._ee_target + dxyz, self.cfg.ee_min, self.cfg.ee_max)
        self._apply_ee_target(self._ee_target)
        self._set_gripper(open_gripper=(grip > 0.0))

        for _ in range(self.cfg.sim_substeps):
            p.stepSimulation()

        obs = self._get_obs()

        ee_pos = obs[0:3]
        cube_pos = obs[3:6]
        dist = float(np.linalg.norm(cube_pos - ee_pos))
        contact = bool(obs[-1] > 0.5)

        # Dense shaping
        reward = -dist
        if contact:
            reward += 0.5
        if cube_pos[2] > self.cfg.lift_height_success:
            reward += 2.0

        terminated = cube_pos[2] > self.cfg.lift_height_success and contact
        truncated = self._step >= self.cfg.max_steps
        if terminated:
            reward += 10.0

        info = {"is_success": bool(terminated), "distance": dist, "has_contact": contact}
        return obs, float(reward), bool(terminated), bool(truncated), info


def make_env(render: bool = False, max_steps: int = 200) -> PandaGraspEnv:
    return PandaGraspEnv(PandaGraspConfig(render=render, max_steps=max_steps))



