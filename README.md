---
title: OpenEnv Traffic Control
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Smart City Traffic Control - RL Environment

This repository contains a simple, functional reinforcement learning (RL) environment designed for the OpenEnv × MetAI hackathon. It simulates a 4-way traffic intersection to test congestion-reduction strategies.

## Environment Architecture

- **Framework**: Fully compliant with the [`gymnasium`](https://gymnasium.farama.org/) interface (`gymnasium.Env`).
- **State Space**: A `Box` observation array of 5 integers: `[Cars North, Cars South, Cars East, Cars West, Light State]`.
  - Car queues: bounded `[0, 20]` per lane.
  - `Light State`: `0` (North-South is Green), `1` (East-West is Green).
- **Action Space**: `Discrete(2)`.
  - `0`: Keep current light state.
  - `1`: Switch light state.
- **Reward Function**: The agent receives a penalty (negative reward) equal to the total number of waiting cars at the intersection. The objective is to maximise the reward by minimising overall congestion.
- **Dynamics**: During a `step()`, a green light allows up to 3 cars to pass. Simultaneously, 0–2 new cars arrive at each of the 4 queues randomly. Car counts are clipped to a maximum of 20 per lane.

## Episode Boundaries

- **Truncation**: The episode ends after **100 steps** (`truncated = True`).
- **Termination**: The episode ends early if total waiting cars across all lanes reaches **40 or more**, representing a critical gridlock condition (`terminated = True`).

## Rendering

The environment supports `render_mode="ansi"`, which returns a formatted ASCII string of the intersection state. Pass it at instantiation:
```python
env = TrafficEnv(render_mode="ansi")
obs, info = env.reset()
print(env.render())
```

## Running the Test Agents

`test_agent.py` validates the environment mechanics by running two back-to-back episodes and comparing their performance:

- **Episode 1 — Random Agent**: Selects actions uniformly at random via `env.action_space.sample()`.
- **Episode 2 — Heuristic Agent**: A greedy baseline that switches the light whenever the currently-red direction has more waiting cars than the green direction. This should consistently outperform the random agent, confirming that the reward function and dynamics correctly incentivise sensible traffic management.
```bash
python test_agent.py
```

The script prints the intersection state and reward at every step, then concludes with a side-by-side summary of total reward and episode length for both agents.

## Files Overview
- `env.py`: The core `gymnasium`-compliant environment containing `TrafficEnv`, `reset()`, `step()`, and `render()`.
- `app.py`: A Gradio web interface to interactively simulate and test the environment logic.
- `test_agent.py`: Compares a Random Agent against a Baseline Heuristic Agent to validate environment mechanics.
- `requirements.txt`: Python dependencies.

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the UI: `python app.py`
3. Open your browser to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).