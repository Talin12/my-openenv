import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrafficEnv(gym.Env):
    """
    A single 4-way intersection traffic control environment for RL research.

    Compliant with the Gymnasium API (https://gymnasium.farama.org/).

    Observation Space:
        Box(low=[0,0,0,0,0], high=[20,20,20,20,1], dtype=int32)
        Index 0 — Cars waiting: North lane (0–20)
        Index 1 — Cars waiting: South lane (0–20)
        Index 2 — Cars waiting: East  lane (0–20)
        Index 3 — Cars waiting: West  lane (0–20)
        Index 4 — Light state: 0 = North-South green, 1 = East-West green

    Action Space:
        Discrete(2)
        0 — Keep current light state
        1 — Switch light state

    Reward:
        -(total waiting cars across all lanes) per step.
        The agent is incentivised to minimise congestion.

    Episode End:
        Terminated: total waiting cars >= 40 (gridlock threshold).
        Truncated:  step count reaches max_steps (default 100).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        """
        Initialise the environment.

        Args:
            render_mode (str | None): Rendering mode. Supports "ansi" for a
                formatted string representation of the intersection state.
        """
        super().__init__()

        self.render_mode = render_mode
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
        """
        Reset the environment to a new initial state.

        Car queues are randomly initialised in [0, 10]. The light defaults
        to North-South green. Step counter is reset to zero.

        Args:
            seed (int | None): Seed for reproducibility.
            options (dict | None): Unused; included for API compliance.

        Returns:
            obs (np.ndarray): Initial observation.
            info (dict): Empty info dict.
        """
        super().reset(seed=seed)

        self.current_step = 0

        cars = np.random.randint(0, 11, size=4)
        light_state = 0
        self.state = np.array([*cars, light_state], dtype=np.int32)

        return self._get_obs(), {}

    def step(self, action):
        """
        Advance the environment by one timestep.

        Dynamics (in order):
            1. Apply action — optionally flip the light state.
            2. Move cars   — up to 3 cars clear the green-light lanes.
            3. Arrive cars — 0–2 new cars join each of the 4 lanes randomly.
            4. Clip queues — each lane is capped at 20 cars.
            5. Compute reward — negative sum of all waiting cars.
            6. Check episode end conditions.

        Args:
            action (int): 0 = keep light, 1 = switch light.

        Returns:
            obs        (np.ndarray): Updated observation.
            reward     (float):      Congestion penalty (negative total cars).
            terminated (bool):       True if total cars >= 40 (gridlock).
            truncated  (bool):       True if step count >= max_steps.
            info       (dict):       Empty info dict.
        """
        self.current_step += 1

        cars_n, cars_s, cars_e, cars_w, light = self.state

        # 1. Apply Action
        if action == 1:
            light = 1 - light

        # 2. Move Cars (green light clears up to 3 cars per lane per step)
        cars_to_pass = 3
        if light == 0:  # N-S Green
            cars_n = max(0, cars_n - cars_to_pass)
            cars_s = max(0, cars_s - cars_to_pass)
        else:           # E-W Green
            cars_e = max(0, cars_e - cars_to_pass)
            cars_w = max(0, cars_w - cars_to_pass)

        # 3. Add Incoming Traffic
        cars_n += np.random.randint(0, 3)
        cars_s += np.random.randint(0, 3)
        cars_e += np.random.randint(0, 3)
        cars_w += np.random.randint(0, 3)

        # 4. Clip to observation space bounds
        cars_n, cars_s, cars_e, cars_w = np.clip(
            [cars_n, cars_s, cars_e, cars_w], 0, 20
        )

        self.state = np.array([cars_n, cars_s, cars_e, cars_w, light], dtype=np.int32)

        # 5. Calculate Reward
        total_cars = int(cars_n) + int(cars_s) + int(cars_e) + int(cars_w)
        reward = -total_cars

        # 6. Episode Boundaries
        terminated = total_cars >= 40
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """
        Render the current intersection state.

        Only active when render_mode="ansi". Returns a bordered ASCII string
        showing the current step, light state, per-lane car counts, and total
        waiting cars. Returns None for unsupported render modes.

        Returns:
            str | None: Formatted intersection string, or None.
        """
        if self.render_mode != "ansi":
            return

        cars_n, cars_s, cars_e, cars_w, light = self.state
        light_str = "N-S Green 🟢" if light == 0 else "E-W Green 🟢"
        total = int(cars_n) + int(cars_s) + int(cars_e) + int(cars_w)

        return (
            f"┌─── Step {self.current_step:>3} ──────────────────┐\n"
            f"│  🚦 Light: {light_str:<22}│\n"
            f"│                                   │\n"
            f"│  🚗 North: {cars_n:>2}   South: {cars_s:>2}          │\n"
            f"│  🚗 East:  {cars_e:>2}   West:  {cars_w:>2}          │\n"
            f"│                                   │\n"
            f"│  Total Waiting: {total:>2}                │\n"
            f"└───────────────────────────────────┘"
        )

    def _get_obs(self):
        return self.state.copy()