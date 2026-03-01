from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from projects.pybullet_panda_grasp.env import make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gui", action="store_true", help="Enable PyBullet GUI rendering.")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs/pybullet_panda_grasp_ppo")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    def _train():
        env = make_env(render=args.gui, max_steps=200)
        env = Monitor(env)
        return env

    train_env = DummyVecEnv([_train])
    train_env = VecMonitor(train_env)

    def _eval():
        env = make_env(render=False, max_steps=200)
        env = Monitor(env)
        return env

    eval_env = DummyVecEnv([_eval])
    eval_env = VecMonitor(eval_env)

    checkpoint_cb = CheckpointCallback(save_freq=50_000, save_path=args.logdir, name_prefix="ppo_ckpt")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.98,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, eval_cb])
    model.save(os.path.join(args.logdir, "ppo_final"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()



