from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class Config:
    env_id: str = "FrozenLake-v1"
    is_slippery: bool = False
    gamma: float = 0.99
    alpha: float = 0.8
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000
    train_episodes: int = 20000
    eval_episodes: int = 100


def make_env(cfg: Config):
    return gym.make(cfg.env_id, is_slippery=cfg.is_slippery)


def init_q_table(env):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    return np.zeros((n_s, n_a), dtype=np.float32)


def epsilon_greedy(q: np.ndarray, s: int, epsilon: float, n_a: int) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_a))
    return int(np.argmax(q[s]))


def td_update(q: np.ndarray, s: int, a: int, r: float, s2: int, cfg: Config):
    q[s, a] = q[s, a] + cfg.alpha * (r + cfg.gamma * np.max(q[s2]) - q[s, a])


def run_episode(env, q: np.ndarray, cfg: Config, epsilon: float):
    ep_ret, ep_len = 0.0, 0
    s, _ = env.reset()
    n_a = env.action_space.n
    while True:
        a = epsilon_greedy(q, s, epsilon, n_a)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        td_update(q, s, a, float(r), s2, cfg)
        s = s2
        ep_ret += float(r)
        ep_len += 1
        if done:
            return ep_ret, ep_len


def evaluate_policy(env, q: np.ndarray, episodes: int = 100):
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = int(np.argmax(q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
            done = terminated or truncated
        total += ep_ret
    return total / episodes


def main():
    cfg = Config()
    env = make_env(cfg)
    q = init_q_table(env)

    eps = cfg.eps_start
    rewards = []
    for ep in range(1, cfg.train_episodes + 1):
        t = min(ep / cfg.eps_decay_episodes, 1.0)
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t

        ep_ret, _ = run_episode(env, q, cfg, eps)
        rewards.append(ep_ret)

        if ep % 100 == 0:
            avg = float(np.mean(rewards[-100:]))
            print(f"[train] ep={ep:4d} avg_return(100ep)={avg:.3f} eps={eps:.3f}")

    avg_eval = evaluate_policy(env, q, cfg.eval_episodes)
    print(f"[eval] avg_return over {cfg.eval_episodes} episodes: {avg_eval:.3f}")


if __name__ == "__main__":
    main()



