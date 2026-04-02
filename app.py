import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from env import TrafficEnv

# Initialize FastAPI and environment
app = FastAPI()
env = TrafficEnv()
episode_over = False


# --- Pydantic Models ---

class ActionRequest(BaseModel):
    action: int


# --- REST API Endpoints ---

@app.post("/reset")
def api_reset():
    global episode_over
    episode_over = False
    obs, info = env.reset()
    return {"observation": obs.tolist(), "info": info}


@app.post("/step")
def api_step(req: ActionRequest):
    global episode_over
    obs, reward, terminated, truncated, info = env.step(req.action)
    episode_over = terminated or truncated
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


# --- Gradio UI ---

def format_state(state):
    n, s, e, w, light = state
    light_str = "🟢 North-South" if light == 0 else "🟢 East-West"
    return f"🚦 Green Light: {light_str}\n\n🚗 Cars Waiting:\n- North: {n}\n- South: {s}\n- East: {e}\n- West: {w}"

def reset_simulation():
    global episode_over
    episode_over = False
    state, info = env.reset()
    return format_state(state), "Environment Reset", 0, "🟡 Running"

def step_simulation(action_type):
    global episode_over

    if episode_over:
        return gr.skip(), "Episode ended. Please reset.", gr.skip(), gr.skip()

    action = 0 if action_type == "Keep Light" else 1
    state, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

    if terminated:
        status = "🔴 Terminated (Gridlock!)"
    elif truncated:
        status = "🟠 Truncated (Max Steps Reached)"
    else:
        status = "🟡 Running"

    action_log = f"Agent Action Taken: {action_type} (Action {action})"
    return format_state(state), action_log, reward, status

with gr.Blocks(title="Smart City Traffic RL Environment") as demo:
    gr.Markdown("# 🚦 Traffic Control RL Environment")
    gr.Markdown("A real-world simulation environment for the OpenEnv × MetAI hackathon. Act as the AI agent below to control the intersection.")

    with gr.Row():
        with gr.Column():
            state_display = gr.Textbox(label="Current Environment State", lines=6, interactive=False)
            action_log = gr.Textbox(label="Event Log", interactive=False)
            reward_display = gr.Number(label="Last Step Reward (Congestion Penalty)", interactive=False)
            status_display = gr.Textbox(label="Episode Status", interactive=False)

        with gr.Column():
            gr.Markdown("### Agent Controls")
            btn_keep = gr.Button("Step: Keep Light (Action 0)", variant="primary")
            btn_switch = gr.Button("Step: Switch Light (Action 1)", variant="stop")
            gr.Markdown("---")
            btn_reset = gr.Button("Reset Environment")

    btn_keep.click(fn=lambda: step_simulation("Keep Light"), outputs=[state_display, action_log, reward_display, status_display])
    btn_switch.click(fn=lambda: step_simulation("Switch Light"), outputs=[state_display, action_log, reward_display, status_display])
    btn_reset.click(fn=reset_simulation, outputs=[state_display, action_log, reward_display, status_display])
    demo.load(fn=reset_simulation, outputs=[state_display, action_log, reward_display, status_display])


# --- Mount Gradio onto FastAPI ---
app = gr.mount_gradio_app(app, demo, path="/")

# --- OpenEnv Server Entry Point ---
def start_server():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)