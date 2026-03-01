## PyBullet Panda 抓取（PPO 基线）



### 安装

```bash
pip install -r requirements.txt
```

需要安装：`pybullet`、`gymnasium`、`stable-baselines3`。

### 训练（PPO）

无界面模式：

```bash
python -m projects.pybullet_panda_grasp.train_ppo
```

带 GUI 模式：

```bash
python -m projects.pybullet_panda_grasp.train_ppo --gui
```

### 脚本化演示（非 RL）

作为参考的一个 IK + 抓取演示：

```bash
python projects/pybullet_panda_grasp/demo_scripted_grasp.py
```

使用逆运动学来程序化控制机械臂（不使用 RL）。可作为完整性检查，并帮助理解任务设置。


