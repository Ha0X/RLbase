import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_SIZE = 100_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_EVERY = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30_000
MAX_STEPS = 200_000
EVAL_EVERY = 10_000


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done: bool):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(-1),
            torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            torch.tensor(s2, dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buf)


def select_action(q_net: QNetwork, state, step: int, action_dim: int):
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY_STEPS)
    if random.random() < eps:
        return random.randrange(action_dim), eps
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = q_net(s)
        action = int(q_values.argmax(dim=1).item())
        return action, eps


@torch.no_grad()
def compute_target_y(
    target_q_net: QNetwork,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    next_q_max = target_q_net(next_states).max(dim=1, keepdim=True)[0]
    return rewards + GAMMA * (1.0 - dones) * next_q_max


def compute_loss(q_net: QNetwork, states: torch.Tensor, actions: torch.Tensor, target_y: torch.Tensor):
    q_pred = q_net(states).gather(dim=1, index=actions)
    return F.smooth_l1_loss(q_pred, target_y)


@torch.no_grad()
def evaluate(q_net: QNetwork, env: gym.Env, episodes: int = 5) -> float:
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        terminated = False
        truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = int(q_net(s_t).argmax(dim=1).item())
            s, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
        total += ep_ret
    return total / episodes


def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_q_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    replay = ReplayBuffer(REPLAY_SIZE)

    s, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        a = env.action_space.sample()
        s2, r, done, truncated, _ = env.step(a)
        replay.push(s, a, r, s2, done or truncated)
        s = s2 if not (done or truncated) else env.reset()[0]

    s, _ = env.reset()
    step, best_eval = 0, -1e9
    while step < MAX_STEPS:
        a, eps = select_action(q_net, s, step, action_dim)
        s2, r, done, truncated, _ = env.step(a)
        d = done or truncated
        replay.push(s, a, r, s2, d)

        s = s2
        step += 1

        states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
        target_y = compute_target_y(target_q_net, rewards, next_states, dones)
        loss = compute_loss(q_net, states, actions, target_y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        if step % TARGET_UPDATE_EVERY == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        if d:
            s, _ = env.reset()

        if step % EVAL_EVERY == 0:
            eval_env = gym.make("CartPole-v1")
            eval_ret = evaluate(q_net, eval_env)
            eval_env.close()
            best_eval = max(best_eval, eval_ret)
            print(
                f"step={step}  eps={eps:.3f}  eval_return={eval_ret:.1f}  best={best_eval:.1f}  loss={loss.item():.4f}"
            )

    env.close()


def main():
    train()


if __name__ == "__main__":
    main()



