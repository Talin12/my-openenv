import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.max_steps = 100
        self.current_step = 0

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([20, 20, 20, 20, 1], dtype=np.int32),
            dtype=np.int32
        )

        self.state = np.zeros(5, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        cars = np.random.randint(0, 11, size=4)
        light_state = 0
        self.state = np.array([*cars, light_state], dtype=np.int32)

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        cars_n, cars_s, cars_e, cars_w, light = self.state

        # 1. Apply Action
        if action == 1:
            light = 1 - light

        # 2. Move Cars (Green light lets up to 3 cars pass per step)
        cars_to_pass = 3
        if light == 0:  # N-S Green
            cars_n = max(0, cars_n - cars_to_pass)
            cars_s = max(0, cars_s - cars_to_pass)
        else:  # E-W Green
            cars_e = max(0, cars_e - cars_to_pass)
            cars_w = max(0, cars_w - cars_to_pass)

        # 3. Add Incoming Traffic and clip to observation space bounds
        cars_n += np.random.randint(0, 3)
        cars_s += np.random.randint(0, 3)
        cars_e += np.random.randint(0, 3)
        cars_w += np.random.randint(0, 3)

        cars_n, cars_s, cars_e, cars_w = np.clip(
            [cars_n, cars_s, cars_e, cars_w], 0, 20
        )

        self.state = np.array([cars_n, cars_s, cars_e, cars_w, light], dtype=np.int32)

        # 4. Calculate Reward
        total_cars = int(cars_n) + int(cars_s) + int(cars_e) + int(cars_w)
        reward = -total_cars

        # 5. Episode Boundaries
        terminated = total_cars >= 40  # Gridlock threshold
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return self.state.copy()