# from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env

from snakeenv import SnekEnv

env = SnekEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
