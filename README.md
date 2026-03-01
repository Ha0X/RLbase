# RLbase


1) **强化学习基础算法**（DQN / SAC / PPO 等）
2) **在 PyBullet 环境中用强化学习训练机械臂抓取**



## 目录结构

- `scripts/`: 
  - `scripts/basics/`: 强化学习基础算法（Q-learning / DQN / Actor-Critic / REINFORCE 等）
  - `scripts/pybullet/`: PyBullet 相关训练入口（抓取 / 连续控制）
- `arm/`: PyBullet Reacher 连续控制（PPO-Clip + GAE）
- `projects/pybullet_panda_grasp/`: **PyBullet Panda 抓取**（Gym 环境 + PPO 训练入口 + scripted demo）

## 环境

建议使用虚拟环境：

```bash
pip install -U pip
pip install -r requirements.txt
pip install pybullet pybullet-envs-gymnasium
```


## 运行示例

推荐入口：

```bash
python -m scripts.basics.qlearning_frozenlake
python -m scripts.basics.dqn_cartpole
python -m scripts.basics.actor_critic_cartpole
python -m scripts.basics.reinforce_cartpole
```

如需使用 REINFORCE 的完整 CLI 选项（支持 baseline、不同环境等），可进入 `scripts/basics/reinforce/` 目录运行：

```bash
cd scripts/basics/reinforce
python main.py --env CartPole-v1 --baseline --total-steps 150000
```

### PyBullet 抓取（PPO）

```bash
python -m scripts.pybullet.panda_grasp_ppo
python -m scripts.pybullet.panda_grasp_ppo --gui
```

### PyBullet 连续控制（Reacher, PPO-Clip）

```bash
python -m scripts.pybullet.reacher_ppo_clip
```





