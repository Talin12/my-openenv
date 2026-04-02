import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # State: [cars_N, cars_S, cars_E, cars_W, light_state]
        # light_state: 0 = N-S green, 1 = E-W green

        # Action space: 0 = Keep light, 1 = Switch light
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # - cars_N, cars_S, cars_E, cars_W: each in range [0, 20]
        # - light_state: 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([20, 20, 20, 20, 1], dtype=np.int32),
            dtype=np.int32
        )

        self.state = np.zeros(5, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly initialize cars on each road (0 to 10)
        cars = np.random.randint(0, 11, size=4)
        light_state = 0  # Default to N-S green
        self.state = np.array([*cars, light_state], dtype=np.int32)

        return self._get_obs(), {}

    def step(self, action):
        # Action: 0 = Keep light, 1 = Switch light
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

        # 3. Add Incoming Traffic (0 to 2 new cars arrive at each lane per step)
        cars_n += np.random.randint(0, 3)
        cars_s += np.random.randint(0, 3)
        cars_e += np.random.randint(0, 3)
        cars_w += np.random.randint(0, 3)

        self.state = np.array([cars_n, cars_s, cars_e, cars_w, light], dtype=np.int32)

        # 4. Calculate Reward (Negative sum of waiting cars = penalty for congestion)
        reward = -(int(cars_n) + int(cars_s) + int(cars_e) + int(cars_w))

        # terminated and truncated are False for now (no episode end condition yet)
        terminated = False
        truncated = False
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Return a copy to prevent accidental state mutation outside the class
        return self.state.copy()