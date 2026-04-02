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
- **Reward Function**: The agent receives a penalty (negative reward) equal to the total number of waiting cars at the intersection. The objective is to maximize the reward by minimizing overall congestion.
- **Dynamics**: During a `step()`, a green light allows up to 3 cars to pass. Simultaneously, 0–2 new cars arrive at each of the 4 queues randomly. Car counts are clipped to a maximum of 20 per lane.

## Episode Boundaries

- **Truncation**: The episode ends after **100 steps** (`truncated = True`).
- **Termination**: The episode ends early if total waiting cars across all lanes reaches **40 or more**, representing a critical gridlock condition (`terminated = True`).

## Files Overview
- `env.py`: The core `gymnasium`-compliant environment containing `TrafficEnv`, `reset()`, and `step()`.
- `app.py`: A Gradio web interface to interactively simulate and test the environment logic.
- `requirements.txt`: Python dependencies.

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the UI: `python app.py`
3. Open your browser to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).