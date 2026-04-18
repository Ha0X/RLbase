# RLbase

Reinforcement learning algorithms implemented from scratch, with robot-arm experiments in PyBullet simulation.

---

## Contents

| Path | Description |
|---|---|
| [`scripts/basics/`](./scripts/basics/) | Classic RL algorithms on Gym environments |
| [`scripts/pybullet/`](./scripts/pybullet/) | Robot control experiments in PyBullet |
| [`projects/pybullet_panda_grasp/`](./projects/pybullet_panda_grasp/) | Panda grasping environment + PPO training + scripted demo |
| [`arm/`](./arm/) | Self-contained PPO-Clip + GAE for PyBullet Reacher |
| [`note.md`](./note.md) | Study notes on RL fundamentals |

---

## Algorithms

### Classic Control (Gym)

All implementations are self-contained with no RL framework dependency:

| Script | Algorithm | Environment |
|---|---|---|
| `scripts/basics/qlearning_frozenlake.py` | Tabular Q-learning | FrozenLake-v1 |
| `scripts/basics/dqn_cartpole.py` | DQN (replay buffer + target network) | CartPole-v1 |
| `scripts/basics/reinforce_cartpole.py` | REINFORCE | CartPole-v1 |
| `scripts/basics/actor_critic_cartpole.py` | Actor-Critic (TD) | CartPole-v1 |
| `scripts/basics/reinforce/main.py` | REINFORCE + baseline, full CLI | CartPole / LunarLander |

### Robot Arm (PyBullet)

**Panda Grasping** ([`projects/pybullet_panda_grasp/`](./projects/pybullet_panda_grasp/)):
- Custom Gymnasium environment with Franka Panda arm
- PPO training entry: `train_ppo.py`
- Scripted IK demo as a sanity-check baseline: `demo_scripted_grasp.py`

**Reacher Continuous Control** ([`arm/arm_new.py`](./arm/arm_new.py)):
- PPO-Clip + GAE on `ReacherBulletEnv-v0`
- Observation normalization, entropy regularization, minibatch updates

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyBullet environments:

```bash
pip install pybullet pybullet-envs-gymnasium
```

---

## Quick Start

```bash
# Tabular and deep RL basics
python -m scripts.basics.qlearning_frozenlake
python -m scripts.basics.dqn_cartpole
python -m scripts.basics.actor_critic_cartpole
python -m scripts.basics.reinforce_cartpole

# REINFORCE with full CLI (baseline, different envs)
cd scripts/basics/reinforce
python main.py --env CartPole-v1 --baseline --total-steps 150000

# Panda grasping (headless / GUI)
python -m scripts.pybullet.panda_grasp_ppo
python -m scripts.pybullet.panda_grasp_ppo --gui

# Reacher continuous control
python -m scripts.pybullet.reacher_ppo_clip
```
