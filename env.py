import numpy as np

class TrafficEnv:
    def __init__(self):
        # State: [cars_N, cars_S, cars_E, cars_W, light_state]
        # light_state: 0 = N-S green, 1 = E-W green
        self.state = np.zeros(5, dtype=int)
        
    def reset(self):
        # Randomly initialize cars on each road (0 to 10)
        cars = np.random.randint(0, 11, size=4)
        light_state = 0 # Default to N-S green
        self.state = np.array([*cars, light_state])
        return self._get_obs()
        
    def step(self, action):
        # Action: 0 = Keep light, 1 = Switch light
        cars_n, cars_s, cars_e, cars_w, light = self.state
        
        # 1. Apply Action
        if action == 1:
            light = 1 - light
            
        # 2. Move Cars (Green light lets up to 3 cars pass per step)
        cars_to_pass = 3
        if light == 0: # N-S Green
            cars_n = max(0, cars_n - cars_to_pass)
            cars_s = max(0, cars_s - cars_to_pass)
        else: # E-W Green
            cars_e = max(0, cars_e - cars_to_pass)
            cars_w = max(0, cars_w - cars_to_pass)
            
        # 3. Add Incoming Traffic (0 to 2 new cars arrive at each lane per step)
        cars_n += np.random.randint(0, 3)
        cars_s += np.random.randint(0, 3)
        cars_e += np.random.randint(0, 3)
        cars_w += np.random.randint(0, 3)
        
        self.state = np.array([cars_n, cars_s, cars_e, cars_w, light])
        
        # 4. Calculate Reward (Negative sum of waiting cars = penalty for congestion)
        reward = -(cars_n + cars_s + cars_e + cars_w)
        
        return self._get_obs(), reward
        
    def _get_obs(self):
        # Return a copy to prevent accidental state mutation outside the class
        return self.state.copy()