# Smart City Traffic Control - RL Environment

This repository contains a simple, functional reinforcement learning (RL) environment designed for the OpenEnv × MetAI hackathon. It simulates a 4-way traffic intersection to test congestion-reduction strategies.

## Environment Architecture

- **State Space**: An array of 5 integers representing the environment: `[Cars North, Cars South, Cars East, Cars West, Light State]`.
  - `Light State`: `0` (North-South is Green), `1` (East-West is Green).
- **Action Space**: Discrete binary actions.
  - `0`: Keep current light state.
  - `1`: Switch light state.
- **Reward Function**: The agent receives a penalty (negative reward) equal to the total number of waiting cars at the intersection. The objective is to maximize the reward by minimizing overall congestion.
- **Dynamics**: During a `step()`, a green light allows up to 3 cars to pass. Simultaneously, 0-2 new cars arrive at each of the 4 queues randomly.

## Files Overview
- `env.py`: The core environment logic containing the `reset()` and `step()` functions.
- `app.py`: A Gradio web interface to interactively simulate and test the environment logic.
- `requirements.txt`: Python dependencies.

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the UI: `python app.py`
3. Open your browser to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).