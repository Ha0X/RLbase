## PyBullet Reacher（连续控制）

本目录包含 **PPO-Clip + GAE** 实现，用于 **`ReacherBulletEnv-v0`** 环境。

### 运行

```bash
python -m scripts.pybullet.reacher_ppo_clip
```

### 说明

- `arm_new.py`: 完整的 PPO-Clip + GAE 实现（自包含）
- 使用观测标准化、GAE 进行优势估计、PPO 裁剪


