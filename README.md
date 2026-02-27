# RLbase

一个用于**强化学习算法学习/实现**与**在 PyBullet 环境中做训练实验**的代码仓库（monorepo）。

## 目标与组织方式

- **算法主线**：在尽量不绑定具体环境的前提下，实现/练习常见 RL 算法（DQN / PG / A2C / PPO 等）。
- **实验主线**：在 PyBullet（通过 Gymnasium/Bullet 环境）里进行训练、可视化、调参、复现实验。

当前仓库由原 `rl_learn` 演进而来，并通过 `git subtree` 导入了原 `pybullet` 仓库到 `pybullet/` 目录（历史已保留）。

## 目录概览（现状）

- `pybullet/`: 从原 `pybullet` 仓库导入的内容（例如 `robotarm.py` 等脚本）
- `arm/`: 与 Bullet Reacher 等连续控制环境相关的实验代码
- `pg_reinforce/`: policy gradient/REINFORCE 相关练习
- 根目录的一些脚本：`ppo.py`、`actor-critic.py`、`DQN-cartpole.py` 等

> 后续推荐逐步整理成更清晰的结构：`rl/`（算法）、`envs/`（环境封装）、`train/`（训练入口）、`experiments/`（实验配置/复现说明）。

## 快速开始（最少依赖）

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install gymnasium numpy
```

如要跑 Bullet 环境（例如 `ReacherBulletEnv-v0`），通常需要额外安装 Bullet 环境注册包（你的代码里使用的是 `pybullet_envs_gymnasium`）：

```bash
pip install pybullet pybullet-envs-gymnasium
```

## 重要说明

- `runs/`、`models/`、各类缓存与大文件建议不要提交到 git（见 `.gitignore`）。
- 如果你准备把 GitHub 仓库也改名为 `RLbase`，可以在 GitHub 上重命名仓库后，再本地更新 `origin` 的 URL。


