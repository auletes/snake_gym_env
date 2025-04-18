import random
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Generic, SupportsFloat, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

SNAKE_LEN_GOAL = 30

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id="snake-game-v0",  # call it whatever you want
    entry_point="snakeenv:SnekEnv",  # module_name:class_name
)


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnekEnv(gym.Env):

    def __init__(self):
        super(SnekEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500, shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float32)

        # JEB: Copy paste from reset to intialize types
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]
        self.reward = 0
        self.total_reward = 0
        self.prev_reward = 0
        self.done: bool = False
        self.prev_actions = 0

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.prev_actions.append(action)
        cv2.imshow("a", self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]), (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype="uint8")
            cv2.putText(self.img, "Your Score is {}".format(self.score), (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("a", self.img)
            self.done = True

        self.total_reward = len(self.snake_position) - 3  # default length is 3
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.done:
            self.reward = -10
        else:
            # self.reward  = self.score # hm apples eaten ?
            self.reward = self.score * 10  # hm apples eaten ?
        info: Dict[str, Any] = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        # Return observation, reward, terminated, truncated (not used), info
        return observation, self.reward, self.done, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        super().reset(seed=seed)  # gym requires this call to control randomness and reproduce scenarios.

        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.reward = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.prev_reward = 0

        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        # Additional info to return. For debugging or whatever.
        info: Dict[str, Any] = {}

        # Return observation and info
        return observation, info


# For unit testing
if __name__ == "__main__":
    env = gym.make("snake-game-v0")

    # Use this to check our custom environment
    print("Check environment begin")
    # check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while True:
        rand_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(rand_action)

        if terminated:
            obs = env.reset()[0]
