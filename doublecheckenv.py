from snakeenv import SnekEnv

env = SnekEnv()
episodes = 50

for epose in range(episodes):
    done = False
    obs, info = env.reset()
    # while True:
    while not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, truncated, info = env.step(random_action)
        print("reward", reward)
