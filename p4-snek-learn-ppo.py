import os
import time

import gymnasium as gym
from stable_baselines3 import PPO, A2C
import snakeenv


models_dir = f"models/A2C-{int(time.time())}"
logdir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("snake-game-v0")
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 100):
    # model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO-{i}", reset_num_timesteps=False, saved_path=models_dir)
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO-{i}", reset_num_timesteps=False)
    model.save(f"{models_dir}/PPO-{i}")











